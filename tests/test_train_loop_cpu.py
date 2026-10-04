"""
train_ms_jp_extra.py の train_and_evaluate (CUDA Graph 経路) を CPU 上で数 step 回す結合テスト。

GPU 用の学習スクリプトは import しただけでは何も検査されないため、ここで
  * 改造したループ本体 (graph_runner を呼び、ログ用の値を取り出し、TensorBoard 用の辞書を作る部分)
  * ログ step / 非ログ step の切り替え
  * global_step の進み方、エポック末の後始末 (gc / empty_cache の間引き)
が最後まで動くことを確認する。CUDA Graph の capture / replay そのものは偽バックエンド (CPU) で代用する。
(実際の GPU での確認は tools/profile_step.py を参照。)
"""

import importlib
import importlib.util
import math
import sys
import types

import pytest
import torch

import cuda_graph_step as CG
from style_bert_vits2.models.hyper_parameters import HyperParameters
from tests import test_cuda_graph_cpu as T


def _importable(name):
    """本物が import できるか。types.ModuleType で作った偽物 (__spec__ が None) は「無い」扱い。"""
    mod = sys.modules.get(name)
    if mod is not None:
        return getattr(mod, "__spec__", None) is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ValueError, ImportError):
        return False


def _stub(monkeypatch, name, **attrs):
    """環境に無い (重い / GPU 専用の) 依存だけを空のモジュールに差し替える。実物があれば何もしない。"""
    if _importable(name):
        return
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)


@pytest.fixture()
def train_module(monkeypatch):
    class _HfApi:  # huggingface_hub.HfApi の代わり (このテストでは使わない)
        pass

    _stub(monkeypatch, "huggingface_hub", HfApi=_HfApi)
    if not _importable("tensorboard"):  # torch.utils.tensorboard は tensorboard が無いと import できない
        tb = types.ModuleType("torch.utils.tensorboard")
        tb.SummaryWriter = object  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", tb)
    _stub(monkeypatch, "transformers.trainer_pt_utils", DistributedLengthGroupedSampler=object)
    sys.modules.pop("train_ms_jp_extra", None)
    mod = importlib.import_module("train_ms_jp_extra")
    yield mod
    sys.modules.pop("train_ms_jp_extra", None)


class _Loader:
    """DataLoader の代わり: 同じ形状のバッチを並べただけ。"""

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class _Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, k, v, step):
        self.scalars.append((k, v, step))


def _setup(log_interval, amp=False):
    hps = HyperParameters.load_from_json(T.os.path.join(T.REPO, "configs/config_jp_extra.json"))
    hps.train.segment_size = T.SEG
    hps.train.log_interval = log_interval
    hps.train.eval_interval = 10**9
    hps.train.bf16_run = amp
    hps.train.fp16_run = False
    hps.model_dir = "unused"
    hps.speedup = False
    hps.repo_id = None
    g, d = T.build_models()
    g.mas_std_over_batch_max = True
    net_g, net_d = CG.SingleProcessWrapper(g), CG.SingleProcessWrapper(d)
    og, od = T.make_opts(g, d)
    sg = torch.optim.lr_scheduler.ExponentialLR(og, 0.99)
    sd = torch.optim.lr_scheduler.ExponentialLR(od, 0.99)
    cfg = CG.StepConfig.from_hps(hps, amp_enabled=amp, amp_dtype=torch.bfloat16 if amp else torch.float32, device_type="cpu")
    runner = CG.GraphedTrainStep(net_g, net_d, og, od, cfg, backend=CG.FakeBackend(), device="cpu")
    return hps, net_g, net_d, og, od, sg, sd, runner


def _run_epoch(TR, hps, net_g, net_d, og, od, sg, sd, runner, writer, n_batches, epoch=1, logger=None):
    loader = _Loader([tuple(T.make_batch(100 + i)) for i in range(n_batches)])
    TR.train_and_evaluate(
        0,
        0,
        epoch,
        hps,
        [net_g, net_d, None, None, None],
        [og, od, None, None],
        [sg, sd, None, None],
        TR.GradScaler(enabled=False),
        [loader, None],
        logger if logger is not None else TR.logger,
        [writer, None],
        None,
        0,
        runner,
    )


def test_graph_branch_runs_logs_and_advances_steps(train_module):
    TR = train_module
    TR.global_step = 0
    TR._last_cleanup_step = 0
    hps, net_g, net_d, og, od, sg, sd, runner = _setup(log_interval=1)
    writer = _Writer()
    _run_epoch(TR, hps, net_g, net_d, og, od, sg, sd, runner, writer, n_batches=4)

    assert TR.global_step == 4
    # 1 回目 eager (warm-up) → 2 回目 capture (+自己検証) → 3 回目以降 replay
    assert runner.n_graphs_ok == 1 and not runner.disabled, runner.summary()
    assert runner.n_replay >= 2, runner.summary()

    by_key = {}
    for k, v, step in writer.scalars:
        by_key.setdefault(k, []).append((step, v))
    for key in ("loss/g/total", "loss/d/total", "loss/g/fm", "loss/g/mel", "loss/g/dur", "loss/g/kl", "grad_norm_d", "grad_norm_g", "learning_rate"):
        assert key in by_key, f"{key} がログされていない: {sorted(by_key)}"
        assert [s for s, _ in by_key[key]] == [0, 1, 2, 3], key
        for _, v in by_key[key]:
            assert isinstance(v, float) and math.isfinite(v), (key, v)
    # D / G の識別器ごとの損失もログされる
    assert any(k.startswith("loss/g/") and k.split("/")[-1].isdigit() for k in by_key)
    assert any(k.startswith("loss/d_r/") for k in by_key) and any(k.startswith("loss/d_g/") for k in by_key)


def test_graph_branch_skips_logging_on_other_steps(train_module):
    TR = train_module
    TR.global_step = 0
    TR._last_cleanup_step = 0
    hps, net_g, net_d, og, od, sg, sd, runner = _setup(log_interval=3)
    writer = _Writer()
    _run_epoch(TR, hps, net_g, net_d, og, od, sg, sd, runner, writer, n_batches=5)
    steps = sorted({s for _, _, s in writer.scalars})
    assert steps == [0, 3], steps
    assert TR.global_step == 5
    # 2 つ目のエポック: global_step は引き継がれ、スケジューラ後に lr を再束縛しても壊れない
    sg.step()
    sd.step()
    runner.after_lr_schedule()
    _run_epoch(TR, hps, net_g, net_d, og, od, sg, sd, runner, writer, n_batches=3, epoch=2)
    assert TR.global_step == 8
    assert sorted({s for _, _, s in writer.scalars}) == [0, 3, 6]
    lrs = [v for k, v, _ in writer.scalars if k == "learning_rate"]
    assert lrs[-1] < lrs[0], lrs  # ExponentialLR で下がっている
    assert not runner.disabled


def test_epoch_end_cleanup_is_throttled(train_module, monkeypatch):
    """gc.collect() / empty_cache() は 100 step 以上空いたときだけ (1 エポック 7〜8 step の小データでの毎エポック実行を避ける)。"""
    TR = train_module
    calls = []
    monkeypatch.setattr(TR.gc, "collect", lambda *a, **k: calls.append("gc"))
    monkeypatch.setattr(TR.torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    TR.global_step = 0
    TR._last_cleanup_step = 0
    hps, net_g, net_d, og, od, sg, sd, runner = _setup(log_interval=1000)
    writer = _Writer()
    _run_epoch(TR, hps, net_g, net_d, og, od, sg, sd, runner, writer, n_batches=3)
    assert calls == []  # 3 step しか進んでいない
    TR.global_step = 150  # 100 step 以上進んだ状態から 1 エポック
    _run_epoch(TR, hps, net_g, net_d, og, od, sg, sd, runner, writer, n_batches=3, epoch=2)
    assert calls == ["gc", "empty_cache"], calls
    assert TR._last_cleanup_step == TR.global_step


class _RecordingLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg, *a, **k):
        self.messages.append(str(msg))

    warning = error = debug = info


def test_perf_line_is_logged_on_log_steps(train_module):
    """ログ step ごとに、同期込みの実効 s/step と DataLoader 待ちの割合が出る (CUDA Graph 後の確認用)。"""
    TR = train_module
    TR.global_step = 0
    TR._last_cleanup_step = 0
    TR._perf.reset()
    hps, net_g, net_d, og, od, sg, sd, runner = _setup(log_interval=5)
    rec = _RecordingLogger()
    _run_epoch(TR, hps, net_g, net_d, og, od, sg, sd, runner, _Writer(), n_batches=11, logger=rec)
    perf = [m for m in rec.messages if m.startswith("[perf]")]
    assert len(perf) == 2, rec.messages  # step 5 と 10 (step 0 は区間が短いので出さない)
    for m in perf:
        assert "s/step" in m and "DataLoader 待ち" in m and "5 step" in m, m
