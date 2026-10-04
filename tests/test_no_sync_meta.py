"""
train_step_body (G 順伝播 → D 更新 → G 更新 + 勾配ノルム) が、GPU では同期や CPU<->GPU 転送になる演算を
1 つも含まないことの検査 (GPU 不要)。

方法: `meta` デバイス (形状だけを持ち計算しないテンソル) 上に実モデル (幅だけ縮小) を作り、1 step を動かしながら
tests/dispatch_probe.py の SyncProbe で全 aten 演算を見張る。meta テンソルは値を持たないので、`.item()` のような
読み出しは (プローブが無くても) その場でエラーになる。つまりこのテストが通る = 値の読み出しも転送も無い。

「検査が空振りしていない」ことの確認として、旧コードの典型パターン (CPU で zeros を作って device に移す Flip、
マスク添字 + min/max の定義域チェックのスプライン) を同じプローブに通すと検出されることも確認する。
元コードの生成器の順伝播だけで同じプローブが 101 箇所 (転送 13・マスク添字 72・値の読み出し 16) を検出し、
修正後は 0 になることは手元で確認済み (元ツリーはリポジトリに含まれないのでテストには入れていない)。

注意:
  * aten 演算レベルの検査であり、MAS の Triton カーネル本体 (super_monotonic_align) は meta では動かせないので、
    ラッパーが行うホスト側のテンソル演算だけを写した stand-in に差し替えている。カーネル自体は
    `tl.load(t_x + batch)` のようにデバイス上の長さを読むだけで、ホストへの読み出しは無い (ソース確認済み)。
  * CUDA ランタイムが内部で行う同期 (cudaMalloc の初回、cuDNN benchmark など) は見えない。
    それらは warm-up 実行 (各形状の 1 回目) で済ませる設計。
"""

import dataclasses

import pytest
import torch
import torch.fx.experimental._config as exp_config

import cuda_graph_step as CG
from tests import test_cuda_graph_cpu as T
from tests.dispatch_probe import SyncProbe


def _stand_in_maximum_path(neg_cent, mask):
    """super_monotonic_align のラッパー (monotonic_alignment.maximum_path) が行うホスト側のテンソル演算を写した版。
    Triton カーネルの起動部分は、結果と同形の path をそのまま返す。"""
    dtype = neg_cent.dtype
    value = neg_cent.transpose(1, 2).float()
    mask_t = mask.transpose(1, 2).to(torch.int32)
    value = value.contiguous()
    value = value.masked_fill_(mask_t.logical_not(), 0)
    path = torch.zeros_like(value, dtype=torch.float32)
    _t_x_max = mask_t.sum(1)[:, 0].to(torch.int32)
    _t_y_max = mask_t.sum(2)[:, 0].to(torch.int32)
    return path.transpose(1, 2).to(dtype)


class _NoopOptimizer(torch.optim.Optimizer):
    """meta パラメータ用。step は何もしない (AdamW の演算は別途 GPU で確認する)。"""

    def __init__(self, params):
        super().__init__(params, {})

    def step(self, closure=None):
        return None


@pytest.fixture()
def meta_models(monkeypatch):
    monkeypatch.setattr(T.monotonic_alignment, "maximum_path", _stand_in_maximum_path)
    with torch.device("meta"):
        g, d = T.build_models()
    return g, d


def _meta_batch():
    return [t.to("meta") for t in T.make_batch(1)]


# ---------------------------------------------------------------------------------------------------------
# 1. プローブ自体が旧コードのパターンを検出できる (= 空振りしない)
# ---------------------------------------------------------------------------------------------------------
def test_probe_flags_the_old_patterns():
    with exp_config.patch(meta_nonzero_assume_all_nonzero=True):
        x = torch.randn(2, 6, 8, device="meta")
        probe = SyncProbe(scalar_placeholder=True)
        with probe:
            # 旧 Flip: CPU で zeros を作ってから device へ
            torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            # 旧スプライン: bool マスクでの代入・取り出し
            inside = (x >= -1) & (x <= 1)
            out = torch.zeros_like(x)
            out[~inside] = x[~inside]
            # 旧スプライン: 定義域チェック (min/max を Python の if に使う)
            if torch.min(x) < -1 or torch.max(x) > 1:
                pass
            # データ依存形状
            torch.nonzero(x)
    kinds = set(probe.counts())
    assert {"transfer", "mask_indexing", "scalar_read", "data_dependent_shape"} <= kinds, probe.findings


def test_probe_is_quiet_on_plain_tensor_math():
    x = torch.randn(2, 6, 8, device="meta")
    probe = SyncProbe()
    with probe:
        y = torch.where(x > 0, x, torch.zeros_like(x))
        y = y.masked_fill(x < -1, 0.0).clamp(-1, 1)
        torch.zeros(2, 3, device=x.device).add_(1).sum()
        torch.cumsum(y, -1)[..., 1:].mean()
    assert probe.n_ops > 0
    assert probe.findings == []


# ---------------------------------------------------------------------------------------------------------
# 2. 実モデルの 1 step に同期 / 転送が無い
# ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("static_shapes", [False, True], ids=["eager-std", "graph-static-std"])
@pytest.mark.parametrize("amp", [False, True], ids=["fp32", "bf16"])
def test_train_step_body_has_no_sync_or_transfer(meta_models, amp, static_shapes):
    g, d = meta_models
    g.mas_std_over_batch_max = static_shapes
    og, od = _NoopOptimizer(g.parameters()), _NoopOptimizer(d.parameters())
    cfg = dataclasses.replace(T.make_cfg(amp), device_type="cpu")
    batch = _meta_batch()
    mas = torch.tensor(0.01, device="meta")

    # 1 回目は一度きりのキャッシュ (マスクのバッファ等) を温める。meta では値の読み出しがあればここで例外になる
    CG.train_step_body(g, d, og, od, batch, mas, cfg)

    probe = SyncProbe()
    with probe:
        outs = CG.train_step_body(g, d, og, od, batch, mas, cfg)
    assert probe.n_ops > 1000, "step が実際に演算を発行していない (検査が空振り)"
    assert probe.findings == [], "\n".join(str(f) for f in probe.findings[:20])
    # 返り値はすべてデバイス上のテンソル (ログ時にだけ 1 回まとめて読み出す設計)
    for name in ("loss_disc_all", "loss_gen_all", "loss_fm", "loss_mel", "loss_dur", "loss_kl", "grad_norm_d", "grad_norm_g"):
        assert isinstance(getattr(outs, name), torch.Tensor), name


def test_generator_forward_with_static_std_has_no_sync(meta_models):
    """学習ループ外 (推論側の共通部品) ではなく、学習時の生成器 forward だけを単独でも確認する。"""
    g, _ = meta_models
    g.mas_std_over_batch_max = True
    g.train()
    x, x_len, spec, spec_len, y, y_len, sid, tone, lang, bert, style = _meta_batch()
    g.current_mas_noise_scale = torch.tensor(0.01, device="meta")
    g(x, x_len, spec, spec_len, sid, tone, lang, bert, style)  # warm-up
    probe = SyncProbe()
    with probe:
        g(x, x_len, spec, spec_len, sid, tone, lang, bert, style)
    assert probe.findings == [], "\n".join(str(f) for f in probe.findings[:20])
