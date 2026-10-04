"""
CUDA Graph 学習経路 (cuda_graph_step.py / data_utils.StaticShapeTable / checkpoints) の CPU テスト。

GPU が無くても動く範囲で、次の点を実モデル (幅だけ縮小した SynthesizerTrn / MultiPeriodDiscriminator) で確認する:
  1. train_step_body が従来の学習ループと同じ損失・同じパラメータ更新になること
  2. GraphedTrainStep の状態遷移 (eager → capture → replay)・静的バッファ・失敗時のフォールバック
  3. バケット上限までパディングしても (MAS ノイズ std を揃えれば) 損失・勾配・更新が変わらないこと
  4. StaticShapeTable / collate の形状
  5. チェックポイントの optimizer state が従来形式に戻ること

実行: python -m pytest tests/test_cuda_graph_cpu.py -x -q
(CUDA Graph そのもの (capture / replay) は GPU でしか試せない。GPU では tools/profile_step.py を使う)
"""

import contextlib
import importlib.util
import os
import sys
import types
from unittest import mock

import numpy as np
import pytest
import torch
from torch.nn import functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
os.chdir(REPO)

# losses.py が import する重い依存 / CUDA 専用の MAS カーネルは (無ければ) 差し替える
# (pytest 経由なら tests/conftest.py が先に同じことをしている。単体で import された場合のための保険)
def _is_missing(name):
    if name in sys.modules:
        return False
    try:
        return importlib.util.find_spec(name) is None
    except ValueError:
        return False


if _is_missing("torchaudio"):
    sys.modules.setdefault("torchaudio", types.ModuleType("torchaudio"))
if _is_missing("transformers"):
    _tr = types.ModuleType("transformers")
    _tr.AutoModel = object
    sys.modules.setdefault("transformers", _tr)
if _is_missing("super_monotonic_align"):
    _sma = types.ModuleType("super_monotonic_align")
    _sma.maximum_path = lambda *a, **k: None
    sys.modules.setdefault("super_monotonic_align", _sma)

import cuda_graph_step as CG  # noqa: E402
from data_utils import StaticShapeTable, TextAudioSpeakerCollate  # noqa: E402
from losses import discriminator_loss, feature_loss, generator_loss, kl_loss  # noqa: E402
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch  # noqa: E402
from style_bert_vits2.models import commons, monotonic_alignment  # noqa: E402
from style_bert_vits2.models import models_jp_extra as MJ  # noqa: E402
from style_bert_vits2.models.hyper_parameters import HyperParameters  # noqa: E402
from style_bert_vits2.models.utils import checkpoints  # noqa: E402
from style_bert_vits2.nlp.symbols import NUM_LANGUAGES, NUM_TONES, SYMBOLS  # noqa: E402

torch.set_num_threads(1)
HPS = HyperParameters.load_from_json(os.path.join(REPO, "configs/config_jp_extra.json"))
HOP = HPS.data.hop_length
SEG = 2048  # テストでは短いセグメント (4 フレーム) にして計算を軽くする


def _maximum_path_cpu(neg_cent, mask):
    """CUDA/Triton の MAS の代わりに numba 版 (monotonic_alignment.py の最初の定義) を使う。"""
    device, dtype = neg_cent.device, neg_cent.dtype
    nc = neg_cent.data.cpu().numpy().astype(np.float32)
    path = np.zeros(nc.shape, dtype=np.int32)
    t_t = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
    t_s = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
    getattr(monotonic_alignment, "__maximum_path_jit")(path, nc, t_t, t_s)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


monotonic_alignment.maximum_path = _maximum_path_cpu


@contextlib.contextmanager
def _scaled_convs(div):
    """D の Conv 幅だけ縮める (演算の個数は変えない)。"""
    import torch.nn as nn

    def wrap(cls):
        def mk(cin, cout, *a, **k):
            g = k.get("groups", 1)
            cin2 = cin if cin == 1 else max(1, cin // div)
            cout2 = cout if cout == 1 else max(1, cout // div)
            if g > 1:
                g2 = min(g, cin2, cout2)
                while cin2 % g2 or cout2 % g2:
                    g2 -= 1
                k["groups"] = g2
            return cls(cin2, cout2, *a, **k)

        return mk

    old = (MJ.Conv1d, MJ.Conv2d)
    MJ.Conv1d, MJ.Conv2d = wrap(nn.Conv1d), wrap(nn.Conv2d)
    try:
        yield
    finally:
        MJ.Conv1d, MJ.Conv2d = old


def build_models(seed=0):
    torch.manual_seed(seed)
    m = HPS.model
    kw = dict(
        use_spk_conditioned_encoder=m.use_spk_conditioned_encoder,
        use_noise_scaled_mas=m.use_noise_scaled_mas,
        use_mel_posterior_encoder=m.use_mel_posterior_encoder,
        use_duration_discriminator=False,
        use_wavlm_discriminator=False,
        inter_channels=32,
        hidden_channels=32,
        filter_channels=64,
        n_heads=m.n_heads,
        n_layers=m.n_layers,
        kernel_size=m.kernel_size,
        p_dropout=m.p_dropout,
        resblock=m.resblock,
        resblock_kernel_sizes=m.resblock_kernel_sizes,
        resblock_dilation_sizes=m.resblock_dilation_sizes,
        upsample_rates=m.upsample_rates,
        upsample_initial_channel=64,
        upsample_kernel_sizes=m.upsample_kernel_sizes,
        n_layers_q=m.n_layers_q,
        use_spectral_norm=m.use_spectral_norm,
        gin_channels=32,
        slm=m.slm,
    )
    g = MJ.SynthesizerTrn(
        len(SYMBOLS),
        HPS.data.filter_length // 2 + 1,
        SEG // HOP,
        n_speakers=8,
        mas_noise_scale_initial=0.01,
        noise_scale_delta=2e-6,
        **kw,
    )
    with _scaled_convs(8):
        d = MJ.MultiPeriodDiscriminator(HPS.model.use_spectral_norm)
    return g, d


def make_batch(seed, B=2, t_txt=41, t_spec=96, lens_txt=None, lens_spec=None, pad_to=None):
    """lens_* を与えると各サンプルの有効長 (残りはゼロ埋め)。pad_to=(text, spec) でさらにゼロ埋めして静的形状にする。"""
    gen = torch.Generator().manual_seed(seed)
    lens_txt = lens_txt or [t_txt] * B
    lens_spec = lens_spec or [t_spec] * B
    T_txt, T_spec = max(lens_txt), max(lens_spec)
    if pad_to is not None:
        T_txt, T_spec = pad_to
    x = torch.zeros(B, T_txt, dtype=torch.long)
    tone = torch.zeros(B, T_txt, dtype=torch.long)
    lang = torch.zeros(B, T_txt, dtype=torch.long)
    bert = torch.zeros(B, 1024, T_txt)
    spec = torch.zeros(B, HPS.data.filter_length // 2 + 1, T_spec)
    y = torch.zeros(B, 1, T_spec * HOP)
    for i in range(B):
        n, s = lens_txt[i], lens_spec[i]
        x[i, :n] = torch.randint(1, len(SYMBOLS), (n,), generator=gen)
        tone[i, :n] = torch.randint(0, NUM_TONES, (n,), generator=gen)
        lang[i, :n] = torch.randint(0, NUM_LANGUAGES, (n,), generator=gen)
        bert[i, :, :n] = torch.randn(1024, n, generator=gen)
        spec[i, :, :s] = torch.rand(spec.shape[1], s, generator=gen) * 0.1 + 1e-3
        y[i, :, : s * HOP] = (torch.rand(1, s * HOP, generator=gen) - 0.5) * 0.2
    return (
        x,
        torch.tensor(lens_txt),
        spec,
        torch.tensor(lens_spec),
        y,
        torch.tensor([s * HOP for s in lens_spec]),
        torch.zeros(B, dtype=torch.long),
        tone,
        lang,
        bert,
        torch.randn(B, 256, generator=gen),
    )


def make_cfg(amp):
    return CG.StepConfig(
        segment_size=SEG,
        hop_length=HOP,
        win_length=HPS.data.win_length,
        filter_length=HPS.data.filter_length,
        n_mel_channels=HPS.data.n_mel_channels,
        sampling_rate=HPS.data.sampling_rate,
        mel_fmin=HPS.data.mel_fmin,
        mel_fmax=HPS.data.mel_fmax,
        c_mel=HPS.train.c_mel,
        c_kl=HPS.train.c_kl,
        amp_enabled=amp,
        amp_dtype=torch.bfloat16 if amp else torch.float32,
        device_type="cpu",
    )


def make_opts(g, d):
    kw = dict(betas=HPS.train.betas, eps=HPS.train.eps)
    return (
        torch.optim.AdamW(g.parameters(), 1e-4, **kw),
        torch.optim.AdamW(d.parameters(), 1e-4, **kw),
    )


def original_step(g, d, og, od, batch, amp, mas=0.01):
    """train_ms_jp_extra.py::train_and_evaluate の 1 step (従来コード) を CPU / DDP 無しで再現したもの。"""
    x, x_len, spec, spec_len, y, y_len, sid, tone, lang, bert, style = batch
    if g.use_noise_scaled_mas:
        g.current_mas_noise_scale = mas
    ac = lambda: torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp)  # noqa: E731
    with ac():
        (y_hat, l_length, attn, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q), (hx, logw, logw_), gg) = g(
            x, x_len, spec, spec_len, sid, tone, lang, bert, style
        )
        mel = spec_to_mel_torch(spec, HPS.data.filter_length, HPS.data.n_mel_channels, HPS.data.sampling_rate, HPS.data.mel_fmin, HPS.data.mel_fmax)
        y_mel = commons.slice_segments(mel, ids_slice, SEG // HOP)
        y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1).float(), HPS.data.filter_length, HPS.data.n_mel_channels, HPS.data.sampling_rate, HOP, HPS.data.win_length, HPS.data.mel_fmin, HPS.data.mel_fmax)
        ys = commons.slice_segments(y, ids_slice * HOP, SEG)
        y_d_hat_r, y_d_hat_g, _, _ = d(ys, y_hat.detach())
        with ac():
            loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
    od.zero_grad()
    loss_disc.backward()
    if amp:
        torch.nn.utils.clip_grad_norm_(d.parameters(), max_norm=200)
    gnd = commons.clip_grad_value_(d.parameters(), None)
    od.step()
    with ac():
        y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = d(ys, y_hat)
        with ac():
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * HPS.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * HPS.train.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _ = generator_loss(y_d_hat_g)
            loss_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
    og.zero_grad()
    loss_all.backward()
    if amp:
        torch.nn.utils.clip_grad_norm_(g.parameters(), max_norm=500)
    gng = commons.clip_grad_value_(g.parameters(), None)
    og.step()
    return {
        "loss_disc_all": float(loss_disc),
        "loss_gen_all": float(loss_all),
        "loss_fm": float(loss_fm),
        "loss_mel": float(loss_mel),
        "loss_dur": float(loss_dur),
        "loss_kl": float(loss_kl),
        "grad_norm_d": gnd,
        "grad_norm_g": gng,
    }


def params_of(*mods):
    return [p.detach().clone() for m in mods for p in m.parameters()]


def max_rel_diff(a_list, b_list, floor_ratio=1e-4):
    """テンソルごとの最大相対誤差の最大値。ただし「全体で見て無視できるほど小さい」テンソル
    (例: softmax のシフト不変性で勾配が理論上 0 になる attention の key bias) は分母に下限を置いて除外する。"""
    gmax = max(float(a.abs().max()) for a in a_list)
    worst = 0.0
    for a, b in zip(a_list, b_list):
        denom = max(float(a.abs().max()), float(b.abs().max()), floor_ratio * gmax, 1e-12)
        worst = max(worst, float((a - b).abs().max()) / denom)
    return worst


def l2_rel_diff(a_list, b_list):
    num = sum(float(((a - b) ** 2).sum()) for a, b in zip(a_list, b_list))
    den = sum(float((a**2).sum()) for a in a_list)
    return (num / max(den, 1e-30)) ** 0.5


def assert_vals_close(a, b, rtol):
    for k in a:
        if k in b:
            assert abs(a[k] - b[k]) <= rtol * max(abs(a[k]), abs(b[k]), 1e-6) + 1e-9, (k, a[k], b[k])


# ----------------------------------------------------------------------------------------
# 1. train_step_body == 従来ループ
# ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("amp", [False, True])
def test_step_body_matches_original_loop(amp):
    batches = [make_batch(10 + i) for i in range(2)]

    g1, d1 = build_models()
    og1, od1 = make_opts(g1, d1)
    ref = []
    for i, b in enumerate(batches):
        torch.manual_seed(500 + i)
        ref.append(original_step(g1, d1, og1, od1, b, amp))

    g2, d2 = build_models()
    og2, od2 = make_opts(g2, d2)
    cfg = make_cfg(amp)
    new = []
    for i, b in enumerate(batches):
        torch.manual_seed(500 + i)
        outs = CG.train_step_body(g2, d2, og2, od2, b, 0.01, cfg)
        new.append(outs.to_floats())

    rtol = 1e-5 if not amp else 2e-3
    for r, n in zip(ref, new):
        assert_vals_close(r, n, rtol)
    assert max_rel_diff(params_of(g1, d1), params_of(g2, d2)) < (1e-5 if not amp else 5e-3)


# ----------------------------------------------------------------------------------------
# 2. GraphedTrainStep の状態遷移 (FakeBackend)
# ----------------------------------------------------------------------------------------
SHAPES = [(41, 96), (45, 112), (37, 96)]


def _sequence():
    seq = []
    for visit in range(3):
        for si, (t_txt, t_spec) in enumerate(SHAPES):
            seq.append(make_batch(100 * visit + si, t_txt=t_txt, t_spec=t_spec))
    return seq


def test_graphed_step_state_machine_matches_eager():
    amp = False
    seq = _sequence()
    cfg = make_cfg(amp)

    g1, d1 = build_models()
    og1, od1 = make_opts(g1, d1)
    ref = []
    for i, b in enumerate(seq):
        torch.manual_seed(900 + i)
        ref.append(CG.train_step_body(g1, d1, og1, od1, b, 0.01, cfg).to_floats())

    g2, d2 = build_models()
    og2, od2 = make_opts(g2, d2)
    backend = CG.FakeBackend()
    runner = CG.GraphedTrainStep(g2, d2, og2, od2, cfg, backend=backend, validate="off")
    got = []
    for i, b in enumerate(seq):
        torch.manual_seed(900 + i)
        got.append(runner.run(b, 0.01, want_norms=False).to_floats())

    for r, n in zip(ref, got):
        assert_vals_close(r, n, 1e-5)
    assert max_rel_diff(params_of(g1, d1), params_of(g2, d2)) < 1e-5
    assert backend.captures == len(SHAPES)  # 形状ごとに 1 回だけ capture
    assert all(e.state == "ready" for e in runner.entries.values())
    # 1 回目は eager (warm-up)、2 回目は capture して replay、3 回目は replay
    assert runner.n_replay == 2 * len(SHAPES)
    assert runner.n_eager == len(SHAPES)
    assert not runner.disabled


@pytest.mark.parametrize("max_ahead", [0, 1, 2, 5])
def test_run_ahead_is_bounded(max_ahead):
    """CPU が GPU より max_ahead step 以上先へ進まない (N step 前のイベントの完了を待つ)。0 なら何もしない。"""
    cfg = make_cfg(False)
    g, d = build_models()
    og, od = make_opts(g, d)
    backend = CG.FakeBackend()
    runner = CG.GraphedTrainStep(g, d, og, od, cfg, backend=backend, validate="off", max_ahead=max_ahead)
    seq = _sequence()
    for b in seq:
        runner.run(b, 0.01, want_norms=False)
    n = len(seq)
    if max_ahead == 0:
        assert backend.event_syncs == 0 and len(runner._events) == 0
    else:
        # 毎 step イベントを 1 つ記録し、溜まった分が max_ahead を超えるたびに最古を 1 つ待つ
        assert len(runner._events) == min(n, max_ahead)
        assert backend.event_syncs == max(0, n - max_ahead)


def test_graphed_step_self_check_passes_and_training_continues():
    seq = _sequence()
    g, d = build_models()
    og, od = make_opts(g, d)
    backend = CG.FakeBackend()
    runner = CG.GraphedTrainStep(g, d, og, od, make_cfg(False), backend=backend, validate="first")
    for i, b in enumerate(seq):
        torch.manual_seed(900 + i)
        vals = runner.run(b, 0.01, want_norms=True).to_floats()
        assert all(np.isfinite(v) for v in vals.values())
    assert runner.validated and not runner.disabled
    assert backend.captures == len(SHAPES)


class _RaisingBackend(CG.FakeBackend):
    def capture(self, fn, pool, guard=None):
        raise RuntimeError("synthetic capture failure")


def test_self_check_runs_for_every_shape_by_default():
    """既定 (validate=all) では形状ごとの初回 capture で eager と突き合わせる。全部通れば学習は続行する。"""
    cfg = make_cfg(False)
    g, d = build_models()
    og, od = make_opts(g, d)
    backend = CG.FakeBackend()
    runner = CG.GraphedTrainStep(g, d, og, od, cfg, backend=backend)
    assert runner.validate_mode == "all"
    seq = _sequence()
    calls = []
    orig_compare = runner._compare

    def counting_compare(*a, **k):
        calls.append(1)
        return orig_compare(*a, **k)

    runner._compare = counting_compare
    for b in seq:
        outs = runner.run(b, 0.01, want_norms=True).to_floats()
        assert all(v == v for v in outs.values())
    assert len(calls) == len(SHAPES)  # 形状ごとに 1 回
    assert runner.n_graphs_ok == len(SHAPES) and not runner.disabled
    assert all(e.state == "ready" for e in runner.entries.values())


def test_capture_failure_falls_back_to_eager_without_corrupting_state():
    seq = _sequence()
    cfg = make_cfg(False)
    g1, d1 = build_models()
    og1, od1 = make_opts(g1, d1)
    for i, b in enumerate(seq):
        torch.manual_seed(900 + i)
        CG.train_step_body(g1, d1, og1, od1, b, 0.01, cfg)

    g2, d2 = build_models()
    og2, od2 = make_opts(g2, d2)
    runner = CG.GraphedTrainStep(g2, d2, og2, od2, cfg, backend=_RaisingBackend(), validate="first")
    for i, b in enumerate(seq):
        torch.manual_seed(900 + i)
        runner.run(b, 0.01, want_norms=False)
    assert runner.disabled
    assert max_rel_diff(params_of(g1, d1), params_of(g2, d2)) < 1e-5


class _CorruptingBackend(CG.FakeBackend):
    """replay の後にパラメータを壊す → 自己検証が失敗して巻き戻されるはず。"""

    def __init__(self, params):
        super().__init__()
        self._params = params

    def replay(self, graph):
        super().replay(graph)
        with torch.no_grad():
            for p in self._params:
                p.add_(0.05 * torch.randn_like(p))


def test_self_check_detects_wrong_replay_and_restores_state():
    seq = _sequence()[: len(SHAPES) * 2]  # 各形状 2 回 (= capture まで)
    cfg = make_cfg(False)
    g1, d1 = build_models()
    og1, od1 = make_opts(g1, d1)
    for i, b in enumerate(seq):
        torch.manual_seed(900 + i)
        CG.train_step_body(g1, d1, og1, od1, b, 0.01, cfg)

    g2, d2 = build_models()
    og2, od2 = make_opts(g2, d2)
    backend = _CorruptingBackend(list(g2.parameters()) + list(d2.parameters()))
    runner = CG.GraphedTrainStep(g2, d2, og2, od2, cfg, backend=backend, validate="first")
    for i, b in enumerate(seq):
        torch.manual_seed(900 + i)
        runner.run(b, 0.01, want_norms=False)
    assert runner.disabled, "壊れた replay を自己検証が見逃した"
    # 失敗時は検証前の状態に戻してから eager でやり直すので、最初から eager で回した結果と一致する
    assert max_rel_diff(params_of(g1, d1), params_of(g2, d2)) < 1e-5


# ----------------------------------------------------------------------------------------
# 3. 静的形状 (バケット上限までパディング) でも結果が変わらない
# ----------------------------------------------------------------------------------------
_ORIG_RANDN, _ORIG_RAND = torch.randn, torch.rand


class DetRandom:
    """乱数を「呼び出し順ごとの固定テーブルの先頭スライス」にする。パディング量が違っても有効領域の乱数が同じになる。"""

    def __init__(self):
        self.k = 0
        self.tables = {}

    def _table(self, kind, ndim):
        key = (self.k, kind)
        if key not in self.tables:
            g = torch.Generator().manual_seed(9000 + self.k)
            shape = [8] + [1100] * (ndim - 1)
            self.tables[key] = (
                _ORIG_RANDN(shape, generator=g) if kind == "n" else _ORIG_RAND(shape, generator=g)
            )
        return self.tables[key]

    def _take(self, kind, size, device, dtype):
        size = tuple(int(s) for s in size)
        t = self._table(kind, len(size))
        self.k += 1
        return t[tuple(slice(0, s) for s in size)].to(device=device, dtype=dtype or torch.float32).clone()

    @staticmethod
    def _size(args):
        return tuple(args[0]) if len(args) == 1 and isinstance(args[0], (list, tuple)) else tuple(args)

    def randn(self, *args, device=None, dtype=None, **kw):
        return self._take("n", self._size(args), device, dtype)

    def rand(self, *args, device=None, dtype=None, **kw):
        return self._take("u", self._size(args), device, dtype)

    def randn_like(self, t, **kw):
        return self._take("n", t.shape, t.device, t.dtype)


def _run_det(batch, amp, static, mas=1.0):
    g, d = build_models()
    g.eval()
    d.eval()  # dropout を無効にして乱数を固定テーブルに寄せる (それ以外の計算は学習時と同じ)
    g.mas_std_over_batch_max = static
    # AdamW の 1 step 目は ±lr (勾配の符号) になり、勾配が 0 付近の要素で符号が反転して比較が不安定になる。
    # 勾配そのものを比べたいので、ここでは SGD (更新量 = lr × 勾配) を使う。
    og = torch.optim.SGD(g.parameters(), lr=1e-2)
    od = torch.optim.SGD(d.parameters(), lr=1e-2)
    before = params_of(g, d)
    det = DetRandom()
    with mock.patch.object(torch, "randn", det.randn), mock.patch.object(
        torch, "rand", det.rand
    ), mock.patch.object(torch, "randn_like", det.randn_like):
        outs = CG.train_step_body(g, d, og, od, batch, mas, make_cfg(amp))
    deltas = [a - b for a, b in zip(params_of(g, d), before)]
    return outs.to_floats(), deltas


@pytest.mark.parametrize("amp", [False, True])
def test_static_padding_does_not_change_the_step(amp):
    lens_txt, lens_spec = [41, 33], [96, 80]  # サンプルごとに長さが違う (従来 collate はバッチ内最大へパディング)
    dyn = make_batch(7, lens_txt=lens_txt, lens_spec=lens_spec)
    stat = make_batch(7, lens_txt=lens_txt, lens_spec=lens_spec, pad_to=(56, 128))
    assert dyn[2].shape[-1] == 96 and stat[2].shape[-1] == 128 and stat[0].shape[-1] == 56
    # fp32 では MAS ノイズを大きく (1.0) して std の補正が効いていることまで確認する。
    # bf16 は行列積の形状で丸めが変わり、ノイズが大きいと離散的な MAS の経路が変わりうるので実学習と同じ 0.01 で見る
    mas = 1.0 if not amp else 0.01
    v_dyn, d_dyn = _run_det(dyn, amp, static=False, mas=mas)
    v_stat, d_stat = _run_det(stat, amp, static=True, mas=mas)
    assert_vals_close(v_dyn, v_stat, 1e-4 if not amp else 5e-3)
    # パラメータの更新量 (= 勾配) が一致すること。bf16 は行列積の形状が変わると丸めが変わるので
    # (同じ入力を 2 回流した結果は完全一致)、要素ごとではなく全体の L2 誤差で見る
    if not amp:
        assert max_rel_diff(d_dyn, d_stat) < 1e-3
    assert l2_rel_diff(d_dyn, d_stat) < (1e-4 if not amp else 2e-2)


def test_padding_sensitivity_is_detectable():
    """上のテストが「何でも通る」テストでないことの確認: MAS ノイズ std の補正を切ると差が出る。"""
    lens_txt, lens_spec = [41, 33], [96, 80]
    dyn = make_batch(7, lens_txt=lens_txt, lens_spec=lens_spec)
    stat = make_batch(7, lens_txt=lens_txt, lens_spec=lens_spec, pad_to=(56, 128))
    v_dyn, _ = _run_det(dyn, False, static=False, mas=1.0)
    v_stat_wrong, _ = _run_det(stat, False, static=False, mas=1.0)  # 補正なし (torch.std を余白込みで計算)
    v_stat_ok, _ = _run_det(stat, False, static=True, mas=1.0)
    err_wrong = max(abs(v_dyn[k] - v_stat_wrong[k]) / max(abs(v_dyn[k]), 1e-6) for k in ("loss_gen_all", "loss_mel", "loss_kl", "loss_fm", "loss_dur"))
    err_ok = max(abs(v_dyn[k] - v_stat_ok[k]) / max(abs(v_dyn[k]), 1e-6) for k in ("loss_gen_all", "loss_mel", "loss_kl", "loss_fm", "loss_dur"))
    print(f"rel err without correction={err_wrong:.3g}, with correction={err_ok:.3g}")
    assert err_ok < 1e-4
    assert err_wrong > err_ok


def test_std_helper_equals_torch_std_on_batch_max_block():
    torch.manual_seed(0)
    B, T_y, T_x = 3, 50, 20
    neg = torch.randn(B, T_y, T_x)
    x_len, y_len = torch.tensor([20, 15, 12]), torch.tensor([50, 44, 30])
    ref = torch.std(neg)
    assert torch.allclose(MJ._std_over_batch_max(neg, x_len, y_len), ref, rtol=1e-5)
    padded = torch.randn(B, 80, 32)
    padded[:, :T_y, :T_x] = neg
    assert torch.allclose(MJ._std_over_batch_max(padded, x_len, y_len), ref, rtol=1e-5)


# ----------------------------------------------------------------------------------------
# 4. StaticShapeTable / collate
# ----------------------------------------------------------------------------------------
class _FakeSampler:
    def __init__(self, boundaries):
        self.boundaries = boundaries


class _FakeDataset:
    def __init__(self, lengths, text_lengths):
        self.lengths, self.text_lengths = lengths, text_lengths


def test_static_shape_table_and_collate():
    lengths = [40, 120, 250, 301, 350, 420, 480, 700]
    text_lengths = [11, 25, 49, 61, 70, 90, 101, 160]
    table = StaticShapeTable.from_sampler(
        _FakeDataset(lengths, text_lengths), _FakeSampler([32, 300, 400, 500, 800]), HOP
    )
    assert table.spec_bounds == [300, 400, 500, 800]
    # 累積最大 (b+1 以下) を 8 の倍数に切り上げ
    assert table.text_pad == {300: 64, 400: 72, 500: 104, 800: 160}
    assert table.resolve(30, 280, 280 * HOP + 5) == (64, 300, 300 * HOP)
    assert table.resolve(80, 300, 300 * HOP) == (80, 300, 300 * HOP)  # テキストが表の値を超えたらそのまま
    assert table.resolve(10, 301, 301 * HOP)[1] == 400
    assert table.resolve(10, 900, 900 * HOP) == (10, 900, 900 * HOP) and table.n_dynamic == 1

    def sample(n_txt, n_spec):
        return (
            torch.randint(1, 5, (n_txt,)),
            torch.rand(513, n_spec),
            torch.rand(1, n_spec * HOP + 7),
            torch.zeros(1, dtype=torch.long),
            torch.zeros(n_txt, dtype=torch.long),
            torch.zeros(n_txt, dtype=torch.long),
            torch.rand(1024, n_txt),
            torch.rand(256),
        )

    batch = [sample(20, 310), sample(30, 380)]
    dyn = TextAudioSpeakerCollate(use_jp_extra=True)(batch)
    sta = TextAudioSpeakerCollate(use_jp_extra=True, static_table=table)(batch)
    assert dyn[2].shape[-1] == 380 and dyn[0].shape[-1] == 30
    assert sta[2].shape[-1] == 400 and sta[0].shape[-1] == 72 and sta[4].shape[-1] == 400 * HOP
    # 有効領域は同じ内容で、余白は 0、長さは実長のまま
    assert torch.equal(sta[2][:, :, :380], dyn[2]) and float(sta[2][:, :, 380:].abs().sum()) == 0
    assert torch.equal(sta[1], dyn[1]) and torch.equal(sta[3], dyn[3]) and torch.equal(sta[5], dyn[5])
    assert torch.equal(sta[9][:, :, :30], dyn[9]) and float(sta[9][:, :, 30:].abs().sum()) == 0


# ----------------------------------------------------------------------------------------
# 5. optimizer state の互換 (checkpoint)
# ----------------------------------------------------------------------------------------
def _via_torch_save(obj):
    """実際のチェックポイントと同じく save → load して、テンソルの別名参照を切る。"""
    import io

    buf = io.BytesIO()
    torch.save(obj, buf)
    buf.seek(0)
    return torch.load(buf, map_location="cpu")


def test_portable_optimizer_state_roundtrip():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(5))
    opt = torch.optim.AdamW([p], 1e-3, betas=(0.8, 0.99), eps=1e-9)
    for _ in range(3):
        p.grad = torch.randn(5)
        opt.step()
    plain = checkpoints.portable_optimizer_state_dict(opt)
    assert plain["param_groups"][0]["lr"] == 1e-3 and not plain["param_groups"][0]["capturable"]

    # グラフ用に変換した状態 (capturable / テンソル lr / step は float32 テンソル)
    lr_t = torch.tensor(1e-3)
    CG.prepare_optimizer_for_capture(opt, lr_t)
    assert opt.param_groups[0]["capturable"] and opt.param_groups[0]["lr"] is lr_t
    saved = checkpoints.portable_optimizer_state_dict(opt)
    assert isinstance(saved["param_groups"][0]["lr"], float)
    assert saved["param_groups"][0]["capturable"] is False
    assert saved["state"][0]["step"].device.type == "cpu"

    # 従来の optimizer にそのまま読み込めて、同じ更新になる
    p2 = torch.nn.Parameter(p.detach().clone())
    opt2 = torch.optim.AdamW([p2], 5e-5, betas=(0.9, 0.9), eps=1e-3)
    opt2.load_state_dict(_via_torch_save(saved))
    # lr は float32 テンソル経由なので 1e-3 から 5e-8 (相対) ずれる。スケジューラが毎エポック再計算するので影響しない
    assert abs(opt2.param_groups[0]["lr"] - 1e-3) < 1e-9 and not opt2.param_groups[0]["capturable"]
    ref = torch.nn.Parameter(p.detach().clone())
    opt_ref = torch.optim.AdamW([ref], 1e-3, betas=(0.8, 0.99), eps=1e-9)
    opt_ref.load_state_dict(_via_torch_save(plain))
    grad = torch.randn(5)
    p2.grad, ref.grad = grad.clone(), grad.clone()
    opt2.step()
    opt_ref.step()
    assert torch.allclose(p2, ref, atol=1e-8)

    # scheduler が lr を float で上書きしても rebind_lr がテンソルへ戻す
    opt.param_groups[0]["lr"] = 2e-4
    CG.rebind_lr(opt, lr_t)
    assert opt.param_groups[0]["lr"] is lr_t and abs(float(lr_t) - 2e-4) < 1e-9


def test_resume_from_checkpoint_matches_uninterrupted_training(tmp_path):
    """途中で保存した checkpoint (従来形式) から再開して CUDA Graph 経路に入っても、止めずに eager で続けた場合と
    同じ更新になる。さらに graph 学習中の状態を保存し直しても従来形式 (float の lr / capturable=False / CPU の step) のまま。
    (Modal / Lightning ではコンテナの再起動で自動再開することが多いので、この経路は実運用で必ず通る)"""
    cfg = make_cfg(False)
    head = [make_batch(300 + i) for i in range(2)]
    tail = _sequence()[: 2 * len(SHAPES)]  # 各形状 2 回 (= warm-up と capture + replay まで通る)

    # 止めずに学習: head で保存 → tail を eager で続ける
    g1, d1 = build_models()
    og1, od1 = make_opts(g1, d1)
    for i, b in enumerate(head):
        torch.manual_seed(700 + i)
        CG.train_step_body(g1, d1, og1, od1, b, 0.01, cfg)
    checkpoints.save_checkpoint(g1, og1, 1e-4, 2, tmp_path / "G_2.pth")
    checkpoints.save_checkpoint(d1, od1, 1e-4, 2, tmp_path / "D_2.pth")
    for i, b in enumerate(tail):
        torch.manual_seed(900 + i)
        CG.train_step_body(g1, d1, og1, od1, b, 0.01, cfg)

    # 再開: モデルも optimizer も作り直して load してから graph runner (train_ms_jp_extra.py と同じ順序)
    g2, d2 = build_models(seed=123)  # 初期値は別 (load で上書きされる)
    og2, od2 = make_opts(g2, d2)
    checkpoints.load_checkpoint(tmp_path / "G_2.pth", g2, og2)
    checkpoints.load_checkpoint(tmp_path / "D_2.pth", d2, od2)
    runner = CG.GraphedTrainStep(g2, d2, og2, od2, cfg, backend=CG.FakeBackend(), validate="off")
    for i, b in enumerate(tail):
        torch.manual_seed(900 + i)
        runner.run(b, 0.01, want_norms=False)
    assert not runner.disabled and runner.n_graphs_ok == len(SHAPES)
    assert max_rel_diff(params_of(g1, d1), params_of(g2, d2)) < 1e-5

    # graph 学習中の状態を保存し直しても、従来のコード (capturable=False の AdamW) でそのまま読める形式
    checkpoints.save_checkpoint(g2, og2, 1e-4, 8, tmp_path / "G_8.pth")
    saved = torch.load(tmp_path / "G_8.pth", map_location="cpu")["optimizer"]
    assert isinstance(saved["param_groups"][0]["lr"], float)
    assert saved["param_groups"][0]["capturable"] is False
    assert all(st["step"].device.type == "cpu" for st in saved["state"].values())
    g3, _ = build_models(seed=7)
    og3 = torch.optim.AdamW(g3.parameters(), 1e-3, betas=HPS.train.betas, eps=HPS.train.eps)
    checkpoints.load_checkpoint(tmp_path / "G_8.pth", g3, og3)
    assert not og3.param_groups[0]["capturable"] and isinstance(og3.param_groups[0]["lr"], float)
