"""
学習 1 ステップ (G 順伝播 → D 更新 → G 更新) を CUDA Graph 1 回の replay にまとめる。

■ なぜ必要か
  SBV2 (JP-Extra) の 1 ステップは約 12 万個の aten 演算 (CUDA カーネル起動は約 3 万回) を
  Python 1 スレッドから順番に発行する「ホスト (CPU) 律速」の負荷になっている。
  GPU が十分速い (H200 / B200 / B300 など) と、GPU は常に次のカーネルを待つ状態になり、
  1 ステップの時間は GPU の性能ではなく「CPU のシングルスレッド性能 × 演算数」で決まる。
  → GPU を速くしても 1 step ≒ 1 秒前後から縮まらない / むしろ CPU が遅いホストでは遅くなる。
  CUDA Graph にすると、この約 3 万回の起動が `graph.replay()` 1 回になり、
  step 時間はほぼ GPU の実計算時間になる。

■ 仕組み (要点)
  * 形状は静的 (data_utils.StaticShapeTable がバケット境界までパディング) →
    形状ごとに 1 つのグラフ。同じ形状が 2 回目に現れた時に capture する
    (1 回目は eager で実行 = ウォームアップ。cuDNN / cuFFT / Triton(MAS) の初期化を済ませる)。
  * capture するのは「順伝播 + 逆伝播 + optimizer.step() + 勾配ノルム」まで全部。
    入力は形状ごとの静的バッファへ copy_ し、学習率 / MAS ノイズ係数は 0 次元 CUDA テンソルを
    in-place 更新する (Python float のままだとグラフに焼き込まれてしまうため)。
  * 勾配バッファは最初の eager ステップで確保したものを使い回し (zero_grad(set_to_none=False))、
    形状の異なるグラフ同士でメモリプールを共有する。
  * 形状ごとの最初の capture 時に「eager で 1 step → 状態を巻き戻す → capture → replay」を行い、
    eager と結果 (loss / 勾配ノルム / パラメータ更新量) が一致することを確認する (1 形状あたり数秒)。
    一致しない / capture に失敗した場合は警告を出して従来どおり eager で続行する。
  * fp16 (GradScaler) / DDP (複数 GPU) / duration・WavLM discriminator は対象外 (従来経路)。

■ 環境変数
  SBV2_CUDA_GRAPH            auto(既定: sm_90 以上のみ有効) | on | off
  SBV2_CUDA_GRAPH_MAX_SHAPES 保持するグラフ形状の上限 (既定 24)
  SBV2_CUDA_GRAPH_VALIDATE   all(既定: 形状ごとの初回 capture で検証) | first(最初の 1 形状だけ) | off
  SBV2_CUDA_GRAPH_RTOL       自己検証の許容相対誤差 (既定 0.1)
  SBV2_CUDA_GRAPH_MAX_AHEAD  CPU が GPU より先に投入してよい step 数の上限 (既定 3)。
                             graph 化すると CPU の投入は一瞬で終わるため、上限が無いと
                             DataLoader のバッチ (ピン留めメモリ) が何十 step 分も溜まり、
                             tqdm の s/it も実際の速度より速く見えてしまう。0 で無効
"""

from __future__ import annotations

import contextlib
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch
from torch.nn import functional as F

from losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from style_bert_vits2.logging import logger
from style_bert_vits2.models import commons


def _log(msg: str) -> None:
    logger.info(f"[cuda-graph] {msg}")


def _warn(msg: str) -> None:
    logger.warning(f"[cuda-graph] {msg}")


# --------------------------------------------------------------------------------------
# 設定 / 出力
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class StepConfig:
    segment_size: int
    hop_length: int
    win_length: int
    filter_length: int
    n_mel_channels: int
    sampling_rate: int
    mel_fmin: float
    mel_fmax: Optional[float]
    c_mel: float
    c_kl: float
    amp_enabled: bool = False
    amp_dtype: torch.dtype = torch.float32
    device_type: str = "cuda"
    clip_norm_d: float = 200.0
    clip_norm_g: float = 500.0
    # G 更新用の D 順伝播中は D のパラメータの requires_grad を切る。
    # 従来コードはこの段階で D の勾配も (捨てられるのに) 計算していた。結果は変わらない。
    freeze_d_in_g_phase: bool = True

    @classmethod
    def from_hps(
        cls,
        hps: Any,
        amp_enabled: bool,
        amp_dtype: torch.dtype,
        device_type: str = "cuda",
    ) -> "StepConfig":
        return cls(
            segment_size=hps.train.segment_size,
            hop_length=hps.data.hop_length,
            win_length=hps.data.win_length,
            filter_length=hps.data.filter_length,
            n_mel_channels=hps.data.n_mel_channels,
            sampling_rate=hps.data.sampling_rate,
            mel_fmin=hps.data.mel_fmin,
            mel_fmax=hps.data.mel_fmax,
            c_mel=hps.train.c_mel,
            c_kl=hps.train.c_kl,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            device_type=device_type,
        )


_SCALAR_FIELDS = (
    "loss_disc_all",
    "loss_gen_all",
    "loss_disc",
    "loss_gen",
    "loss_fm",
    "loss_mel",
    "loss_dur",
    "loss_kl",
)
_LIST_FIELDS = ("losses_gen", "losses_disc_r", "losses_disc_g")
_NORM_FIELDS = ("grad_norm_d", "grad_norm_g")


@dataclass
class StepOutputs:
    """1 ステップの出力 (すべて GPU 上の 0 次元テンソル)。

    CUDA Graph 経由の場合、これらは「静的な出力バッファ」であり、次の `run()` で上書きされる。
    使う側は `to_floats()` などで直ちに読み出すこと。
    """

    loss_disc_all: torch.Tensor
    loss_gen_all: torch.Tensor
    loss_disc: torch.Tensor
    loss_gen: torch.Tensor
    loss_fm: torch.Tensor
    loss_mel: torch.Tensor
    loss_dur: torch.Tensor
    loss_kl: torch.Tensor
    losses_gen: list
    losses_disc_r: list
    losses_disc_g: list
    grad_norm_d: Optional[torch.Tensor] = None
    grad_norm_g: Optional[torch.Tensor] = None

    def _flat(self) -> list[tuple[str, torch.Tensor]]:
        items: list[tuple[str, torch.Tensor]] = []
        for name in _SCALAR_FIELDS:
            items.append((name, getattr(self, name)))
        for name in _LIST_FIELDS:
            for i, t in enumerate(getattr(self, name)):
                items.append((f"{name}.{i}", t))
        for name in _NORM_FIELDS:
            t = getattr(self, name)
            if t is not None:
                items.append((name, t))
        return items

    def to_floats(self) -> dict[str, float]:
        """全スカラーを 1 回の同期でまとめて Python float にする (ログ時だけ呼ぶ)。"""
        items = self._flat()
        stacked = torch.stack([t.detach().float().reshape(()) for _, t in items])
        values = stacked.tolist()
        return {name: v for (name, _), v in zip(items, values)}

    def copy_from(self, other: "StepOutputs") -> None:
        """(fake バックエンド用) 同じ構造の出力を in-place で上書きする。"""
        for name in _SCALAR_FIELDS:
            getattr(self, name).copy_(getattr(other, name))
        for name in _LIST_FIELDS:
            for a, b in zip(getattr(self, name), getattr(other, name)):
                a.copy_(b)
        for name in _NORM_FIELDS:
            a, b = getattr(self, name), getattr(other, name)
            if a is not None and b is not None:
                a.copy_(b)


# --------------------------------------------------------------------------------------
# 1 ステップの本体 (eager でも CUDA Graph capture 内でも同じものを実行する)
# --------------------------------------------------------------------------------------
def train_step_body(
    net_g: torch.nn.Module,
    net_d: torch.nn.Module,
    optim_g: torch.optim.Optimizer,
    optim_d: torch.optim.Optimizer,
    batch: Sequence[torch.Tensor],
    mas_noise_scale: Any,
    cfg: StepConfig,
    want_grad_norms: bool = True,
) -> StepOutputs:
    """train_ms_jp_extra.py::train_and_evaluate の 1 ステップ分 (fp32 / bf16、DDP なし) と同じ計算。

    GPU->CPU 同期 (.item() / bool() / nonzero など) を一切含まない。
    `net_g` / `net_d` は DDP ラッパではなく素のモジュールを渡すこと。
    `mas_noise_scale` は float でも 0 次元テンソルでもよい (グラフ化する場合はテンソル)。
    """
    (
        x,
        x_lengths,
        spec,
        spec_lengths,
        y,
        _y_lengths,
        speakers,
        tone,
        language,
        bert,
        style_vec,
    ) = batch

    if getattr(net_g, "use_noise_scaled_mas", False):
        net_g.current_mas_noise_scale = mas_noise_scale

    def ac():
        return torch.autocast(
            device_type=cfg.device_type, dtype=cfg.amp_dtype, enabled=cfg.amp_enabled
        )

    # ---------------- Generator 順伝播 + Discriminator 更新 ----------------
    with ac():
        (
            y_hat,
            l_length,
            _attn,
            ids_slice,
            _x_mask,
            z_mask,
            (_z, z_p, m_p, logs_p, _m_q, logs_q),
            (_hidden_x, _logw, _logw_),
            _g,
        ) = net_g(
            x, x_lengths, spec, spec_lengths, speakers, tone, language, bert, style_vec
        )
        mel = spec_to_mel_torch(
            spec,
            cfg.filter_length,
            cfg.n_mel_channels,
            cfg.sampling_rate,
            cfg.mel_fmin,
            cfg.mel_fmax,
        )
        y_mel = commons.slice_segments(
            mel, ids_slice, cfg.segment_size // cfg.hop_length
        )
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1).float(),
            cfg.filter_length,
            cfg.n_mel_channels,
            cfg.sampling_rate,
            cfg.hop_length,
            cfg.win_length,
            cfg.mel_fmin,
            cfg.mel_fmax,
        )
        y = commons.slice_segments(y, ids_slice * cfg.hop_length, cfg.segment_size)

        y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
        with ac():
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            loss_disc_all = loss_disc

    optim_d.zero_grad(set_to_none=False)
    loss_disc_all.backward()
    if cfg.amp_enabled:
        torch.nn.utils.clip_grad_norm_(
            parameters=net_d.parameters(), max_norm=cfg.clip_norm_d
        )
    grad_norm_d = commons.grad_total_norm(net_d.parameters()) if want_grad_norms else None
    optim_d.step()

    # ---------------- Generator 更新 ----------------
    d_params = list(net_d.parameters())
    d_flags = [p.requires_grad for p in d_params]
    if cfg.freeze_d_in_g_phase:
        for p in d_params:
            p.requires_grad_(False)
    try:
        with ac():
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            with ac():
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * cfg.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * cfg.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
    finally:
        if cfg.freeze_d_in_g_phase:
            for p, flag in zip(d_params, d_flags):
                p.requires_grad_(flag)

    optim_g.zero_grad(set_to_none=False)
    loss_gen_all.backward()
    if cfg.amp_enabled:
        torch.nn.utils.clip_grad_norm_(
            parameters=net_g.parameters(), max_norm=cfg.clip_norm_g
        )
    grad_norm_g = commons.grad_total_norm(net_g.parameters()) if want_grad_norms else None
    optim_g.step()

    def d(t: torch.Tensor) -> torch.Tensor:
        return t.detach()

    return StepOutputs(
        loss_disc_all=d(loss_disc_all),
        loss_gen_all=d(loss_gen_all),
        loss_disc=d(loss_disc),
        loss_gen=d(loss_gen),
        loss_fm=d(loss_fm),
        loss_mel=d(loss_mel),
        loss_dur=d(loss_dur),
        loss_kl=d(loss_kl),
        losses_gen=[d(t) for t in losses_gen],
        losses_disc_r=[d(t) for t in losses_disc_r],
        losses_disc_g=[d(t) for t in losses_disc_g],
        grad_norm_d=d(grad_norm_d) if grad_norm_d is not None else None,
        grad_norm_g=d(grad_norm_g) if grad_norm_g is not None else None,
    )


# --------------------------------------------------------------------------------------
# Optimizer / LR まわり
# --------------------------------------------------------------------------------------
def probe_capturable_adamw(device: torch.device) -> Optional[str]:
    """capturable=True + テンソル lr の AdamW が使えるか確認する。使えれば None、駄目なら理由。"""
    try:
        p = torch.nn.Parameter(torch.zeros(4, device=device))
        lr = torch.tensor(1e-3, dtype=torch.float32, device=device)
        opt = torch.optim.AdamW([p], lr=lr, betas=(0.8, 0.99), eps=1e-9, capturable=True)
        p.grad = torch.ones_like(p)
        opt.step()
        opt.step()
        lr.fill_(0.0)
        opt.step()
        if not bool(torch.isfinite(p).all()):
            return "non-finite result with capturable AdamW"
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def prepare_optimizer_for_capture(
    optimizer: torch.optim.Optimizer, lr_tensor: torch.Tensor
) -> None:
    """AdamW を CUDA Graph 対応にする (capturable=True / lr をテンソル / step を GPU 上の float32 に)。

    * lr が Python float だと capture 時の値がグラフに焼き込まれ、LR スケジュールが効かなくなる。
    * チェックポイントから load_state_dict すると param_groups が上書きされて capturable=False に
      戻るため、この関数は「読み込み後」に呼ぶ。
    """
    for group in optimizer.param_groups:
        group["capturable"] = True
        group["lr"] = lr_tensor
    for p, st in optimizer.state.items():
        step = st.get("step")
        if torch.is_tensor(step):
            if step.device != p.device or step.dtype != torch.float32:
                st["step"] = step.to(device=p.device, dtype=torch.float32)
        elif step is not None:
            st["step"] = torch.tensor(float(step), dtype=torch.float32, device=p.device)


def rebind_lr(optimizer: torch.optim.Optimizer, lr_tensor: torch.Tensor) -> None:
    """scheduler.step() の後に呼ぶ。scheduler が lr を float で上書きした場合でも
    テンソル lr に値を移して param_groups に戻す (新しい PyTorch は in-place fill_ なので何もしない)。"""
    for group in optimizer.param_groups:
        cur = group["lr"]
        if cur is not lr_tensor:
            lr_tensor.fill_(float(cur))
            group["lr"] = lr_tensor


# --------------------------------------------------------------------------------------
# バックエンド (本番: CUDA / テスト: CPU 上の偽物)
# --------------------------------------------------------------------------------------
class CudaBackend:
    """torch.cuda のストリーム / グラフ API の薄いラッパ。"""

    device_type = "cuda"
    capturable = True

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)
        self._side: Optional[torch.cuda.Stream] = None

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)

    @contextlib.contextmanager
    def side_stream(self):
        """CUDA Graph 公式手順のウォームアップ用サイドストリーム。"""
        if self._side is None:
            self._side = torch.cuda.Stream(device=self.device)
        main = torch.cuda.current_stream(self.device)
        self._side.wait_stream(main)
        with torch.cuda.stream(self._side):
            yield
        main.wait_stream(self._side)

    def make_pool(self):
        return torch.cuda.graph_pool_handle()

    def capture(self, fn: Callable[[], Any], pool, guard=None):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool, capture_error_mode="thread_local"):
            result = fn()
        return g, result

    def replay(self, graph) -> None:
        graph.replay()

    def seed_rng(self, seed: int) -> None:
        torch.cuda.manual_seed(seed)

    def get_rng(self):
        return torch.cuda.get_rng_state(self.device)

    def set_rng(self, state) -> None:
        torch.cuda.set_rng_state(state, self.device)

    def to_device(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(self.device, non_blocking=True)

    def make_scalar(self, value: float) -> torch.Tensor:
        return torch.tensor(float(value), dtype=torch.float32, device=self.device)

    def record_event(self):
        """現在のストリームの「ここまで」を表すイベント。synchronize() はスレッドを眠らせて待つ
        (blocking=True。既定のスピン待ちだと、CPU の少ないコンテナで DataLoader と奪い合う)。"""
        ev = torch.cuda.Event(blocking=True)
        ev.record()
        return ev


class _FakeGraph:
    def __init__(self, fn: Callable[[], StepOutputs], outputs: StepOutputs):
        self.fn, self.outputs = fn, outputs


class FakeBackend:
    """CPU 上で GraphedTrainStep のオーケストレーションを検証するためのバックエンド。
    capture = 1 回実行して出力の構造を得る (状態は guard が巻き戻す)、replay = 再実行して同じ出力テンソルへ書き込む。
    """

    device_type = "cpu"
    capturable = False

    def __init__(self):
        self.device = torch.device("cpu")
        self.captures = 0
        self.replays = 0
        self.event_syncs = 0

    def synchronize(self) -> None:
        pass

    @contextlib.contextmanager
    def side_stream(self):
        yield

    def make_pool(self):
        return object()

    def capture(self, fn, pool, guard=None):
        self.captures += 1
        rng = torch.get_rng_state()  # 本物の capture は何も実行しないので、乱数も進めない
        with guard() if guard is not None else contextlib.nullcontext():
            out = fn()
        torch.set_rng_state(rng)
        return _FakeGraph(fn, out), out

    def replay(self, graph) -> None:
        self.replays += 1
        graph.outputs.copy_from(graph.fn())

    def seed_rng(self, seed: int) -> None:
        torch.manual_seed(seed)

    def get_rng(self):
        return torch.get_rng_state()

    def set_rng(self, state) -> None:
        torch.set_rng_state(state)

    def to_device(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def make_scalar(self, value: float) -> torch.Tensor:
        return torch.tensor(float(value), dtype=torch.float32)

    def record_event(self):
        backend = self

        class _Event:
            def synchronize(self) -> None:
                backend.event_syncs += 1

        return _Event()


# --------------------------------------------------------------------------------------
# GraphedTrainStep
# --------------------------------------------------------------------------------------
class _Entry:
    __slots__ = ("key", "state", "visits", "static", "graph", "outputs")

    def __init__(self, key):
        self.key = key
        self.state = "new"  # new -> warm -> ready | failed
        self.visits = 0
        self.static: Optional[tuple[torch.Tensor, ...]] = None
        self.graph = None
        self.outputs: Optional[StepOutputs] = None


class _Snapshot:
    def __init__(self, tensors: list[torch.Tensor], n_g: int, n_d: int):
        self.tensors = tensors
        self.saved = [t.detach().clone() for t in tensors]
        self.n_g, self.n_d = n_g, n_d

    def restore(self) -> None:
        with torch.no_grad():
            for t, s in zip(self.tensors, self.saved):
                t.copy_(s)


_CHECK_KEYS = (
    "loss_disc_all",
    "loss_gen_all",
    "loss_fm",
    "loss_mel",
    "loss_dur",
    "loss_kl",
    "grad_norm_d",
    "grad_norm_g",
)


def _unwrap(m: torch.nn.Module) -> torch.nn.Module:
    return m.module if hasattr(m, "module") else m


class GraphedTrainStep:
    """形状ごとに CUDA Graph を作って学習 step を実行する。失敗時は eager (同じ本体) に切り替える。"""

    def __init__(
        self,
        net_g: torch.nn.Module,
        net_d: torch.nn.Module,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        cfg: StepConfig,
        backend: Optional[Any] = None,
        device: torch.device | str | None = None,
        max_shapes: Optional[int] = None,
        validate: Optional[str] = None,
        rtol: Optional[float] = None,
        max_ahead: Optional[int] = None,
    ):
        self.net_g, self.net_d = _unwrap(net_g), _unwrap(net_d)
        self.optim_g, self.optim_d = optim_g, optim_d
        self.cfg = cfg
        self.backend = backend if backend is not None else CudaBackend(device or "cuda")
        self.max_shapes = int(
            max_shapes
            if max_shapes is not None
            else os.environ.get("SBV2_CUDA_GRAPH_MAX_SHAPES", "24")
        )
        self.validate_mode = (
            validate
            if validate is not None
            else os.environ.get("SBV2_CUDA_GRAPH_VALIDATE", "all")
        )
        self.rtol = float(
            rtol if rtol is not None else os.environ.get("SBV2_CUDA_GRAPH_RTOL", "0.1")
        )
        self.param_rtol = 0.35
        self.max_ahead = int(
            max_ahead
            if max_ahead is not None
            else os.environ.get("SBV2_CUDA_GRAPH_MAX_AHEAD", "3")
        )
        self._events: deque = deque()

        self.entries: dict[Any, _Entry] = {}
        self.pool = self.backend.make_pool()
        self.disabled = False
        self.validated = False
        self.n_graphs_ok = 0
        self._warned: set[str] = set()
        self.n_eager = 0
        self.n_replay = 0

        # テンソル化するスカラ (これらは in-place で更新する)
        self.mas_t = self.backend.make_scalar(0.0)
        self.use_tensor_lr = bool(self.backend.capturable)
        if self.use_tensor_lr:
            self.lr_g = self.backend.make_scalar(float(optim_g.param_groups[0]["lr"]))
            self.lr_d = self.backend.make_scalar(float(optim_d.param_groups[0]["lr"]))
            prepare_optimizer_for_capture(optim_g, self.lr_g)
            prepare_optimizer_for_capture(optim_d, self.lr_d)

    # ---------------------------------------------------------------- 公開 API
    def after_lr_schedule(self) -> None:
        """scheduler.step() の直後に呼ぶ。"""
        if self.use_tensor_lr:
            rebind_lr(self.optim_g, self.lr_g)
            rebind_lr(self.optim_d, self.lr_d)

    def current_lr(self) -> float:
        lr = self.optim_g.param_groups[0]["lr"]
        return float(lr)

    def run(
        self, batch: Sequence[torch.Tensor], mas_noise_scale: float, want_norms: bool
    ) -> StepOutputs:
        """1 step 実行。戻り値は次の run() まで有効 (グラフ経由の場合は静的バッファ)。"""
        outs = self._run_impl(batch, mas_noise_scale, want_norms)
        self._throttle()
        return outs

    def _throttle(self) -> None:
        """CPU が GPU より max_ahead step 以上先に進まないようにする。

        graph の replay は CPU 側では一瞬で終わるので、何もしないと CPU は DataLoader のバッチを
        どんどん取り込んで GPU に投入し続け (CUDA のキュー深さが上限)、その間ピン留めメモリが
        解放されずに溜まる。N step 前に記録したイベントの完了を待つだけなので、GPU は常に
        数 step 分の仕事を抱えたままで、スループットは変わらない。"""
        if self.max_ahead <= 0:
            return
        self._events.append(self.backend.record_event())
        while len(self._events) > self.max_ahead:
            self._events.popleft().synchronize()

    def _run_impl(
        self, batch: Sequence[torch.Tensor], mas_noise_scale: float, want_norms: bool
    ) -> StepOutputs:
        if self.disabled:
            return self._eager(batch, mas_noise_scale, want_norms)
        key = self._key(batch)
        entry = self.entries.get(key)
        if entry is None:
            if len(self.entries) >= self.max_shapes:
                self._warn_once(
                    "max-shapes",
                    f"グラフ形状の上限 ({self.max_shapes}) に達したため、新しい形状は eager で実行します。",
                )
                return self._eager(batch, mas_noise_scale, want_norms)
            entry = self.entries[key] = _Entry(key)
        entry.visits += 1

        if entry.state == "ready":
            return self._replay(entry, batch, mas_noise_scale)
        if entry.state == "failed":
            return self._eager(batch, mas_noise_scale, want_norms)
        if entry.state == "new":
            t0 = time.time()
            with self.backend.side_stream():
                outs = self._eager_core(batch, mas_noise_scale, True)
            entry.state = "warm"
            self.n_eager += 1
            if os.environ.get("SBV2_CUDA_GRAPH_VERBOSE", "0") == "1":
                _log(f"shape {key}: warm-up(eager) {time.time() - t0:.1f}s")
            return outs
        return self._capture_and_replay(entry, batch, mas_noise_scale, want_norms)

    def summary(self) -> str:
        ready = sum(1 for e in self.entries.values() if e.state == "ready")
        failed = sum(1 for e in self.entries.values() if e.state == "failed")
        return (
            f"shapes={len(self.entries)} (ready={ready}, failed={failed}), "
            f"replay_steps={self.n_replay}, eager_steps={self.n_eager}, disabled={self.disabled}"
        )

    # ---------------------------------------------------------------- eager
    def _eager_core(
        self, batch: Sequence[torch.Tensor], mas_value: float, want_norms: bool
    ) -> StepOutputs:
        inputs = [self.backend.to_device(t) for t in batch]
        self.mas_t.fill_(float(mas_value))
        return train_step_body(
            self.net_g,
            self.net_d,
            self.optim_g,
            self.optim_d,
            inputs,
            self.mas_t,
            self.cfg,
            want_grad_norms=want_norms,
        )

    def _eager(self, batch, mas_value, want_norms) -> StepOutputs:
        self.n_eager += 1
        return self._eager_core(batch, mas_value, want_norms)

    # ---------------------------------------------------------------- capture / replay
    @staticmethod
    def _key(batch: Sequence[torch.Tensor]):
        return (tuple(batch[0].shape), tuple(batch[2].shape), tuple(batch[4].shape))

    def _bind_static(self, entry: _Entry, batch: Sequence[torch.Tensor]) -> None:
        if entry.static is None:
            entry.static = tuple(
                torch.empty_like(t, device=self.backend.device) for t in batch
            )
        for buf, t in zip(entry.static, batch):
            buf.copy_(t, non_blocking=True)

    def _replay(self, entry: _Entry, batch, mas_value) -> StepOutputs:
        assert entry.static is not None and entry.graph is not None
        for buf, t in zip(entry.static, batch):
            buf.copy_(t, non_blocking=True)
        self.mas_t.fill_(float(mas_value))
        self.backend.replay(entry.graph)
        self.n_replay += 1
        assert entry.outputs is not None
        return entry.outputs

    def _state_snapshot(self) -> _Snapshot:
        pg = list(self.net_g.parameters())
        pd = list(self.net_d.parameters())
        others = [b for m in (self.net_g, self.net_d) for b in m.buffers()]
        for opt in (self.optim_g, self.optim_d):
            for st in opt.state.values():
                others.extend(v for v in st.values() if torch.is_tensor(v))
        return _Snapshot(pg + pd + others, len(pg), len(pd))

    @contextlib.contextmanager
    def _guard(self):
        snap = self._state_snapshot()
        try:
            yield
        finally:
            snap.restore()

    def _do_capture(self, entry: _Entry, mas_value: float) -> None:
        assert entry.static is not None
        self.mas_t.fill_(float(mas_value))
        static = entry.static

        def fn() -> StepOutputs:
            return train_step_body(
                self.net_g,
                self.net_d,
                self.optim_g,
                self.optim_d,
                static,
                self.mas_t,
                self.cfg,
                want_grad_norms=True,
            )

        self.backend.synchronize()
        entry.graph, entry.outputs = self.backend.capture(
            fn, self.pool, guard=self._guard
        )

    def _capture_and_replay(
        self, entry: _Entry, batch, mas_value: float, want_norms: bool
    ) -> StepOutputs:
        t0 = time.time()
        do_validate = self.validate_mode == "all" or (
            self.validate_mode == "first" and not self.validated
        )
        snap: Optional[_Snapshot] = None
        rng_state = None
        try:
            self._bind_static(entry, batch)
            ref = None
            if do_validate:
                rng_state = self.backend.get_rng()
                snap = self._state_snapshot()
                self.backend.seed_rng(20240601)
                with self.backend.side_stream():
                    self.mas_t.fill_(float(mas_value))
                    e_outs = train_step_body(
                        self.net_g,
                        self.net_d,
                        self.optim_g,
                        self.optim_d,
                        entry.static,
                        self.mas_t,
                        self.cfg,
                        want_grad_norms=True,
                    )
                ref = {
                    "vals": e_outs.to_floats(),
                    "pg": [p.detach().clone() for p in self.net_g.parameters()],
                    "pd": [p.detach().clone() for p in self.net_d.parameters()],
                }
                snap.restore()
            self._do_capture(entry, mas_value)
            t1 = time.time()
            if do_validate:
                self.backend.seed_rng(20240601)
            self.mas_t.fill_(float(mas_value))
            self.backend.replay(entry.graph)
            self.n_replay += 1
            if do_validate:
                ok, info = self._compare(ref, entry, snap)
                if not ok:
                    raise RuntimeError(f"graph/eager mismatch: {info}")
                self.validated = True
                _log(f"自己検証 OK (eager と一致): {info}")
                self.backend.set_rng(rng_state)
            entry.state = "ready"
            self.n_graphs_ok += 1
            _log(
                f"shape {entry.key}: capture {t1 - t0:.1f}s, 以降は replay "
                f"({sum(1 for e in self.entries.values() if e.state == 'ready')}/{len(self.entries)} shapes ready)"
            )
            assert entry.outputs is not None
            return entry.outputs
        except Exception as e:  # noqa: BLE001
            first_failure = self.n_graphs_ok == 0
            _warn(
                f"shape {entry.key} の CUDA Graph 化に失敗しました ({type(e).__name__}: {e})。"
                " このステップは eager で実行し、以降" +
                ("すべて eager で続行します。" if first_failure else "この形状は eager で続行します。")
            )
            if snap is not None:
                snap.restore()  # 失敗前の状態へ巻き戻してから、本物の 1 step を eager でやり直す
            if rng_state is not None:
                self.backend.set_rng(rng_state)
            entry.state = "failed"
            entry.graph = None
            entry.outputs = None
            entry.static = None
            if first_failure:
                self.disabled = True
            self.n_eager += 1
            return self._eager_core(batch, mas_value, want_norms)

    # ---------------------------------------------------------------- 自己検証
    def _compare(self, ref, entry: _Entry, snap: _Snapshot) -> tuple[bool, str]:
        assert entry.outputs is not None
        cur = entry.outputs.to_floats()
        problems: list[str] = []
        parts: list[str] = []
        for k in _CHECK_KEYS:
            a, b = ref["vals"].get(k), cur.get(k)
            if a is None or b is None:
                continue
            if not (a == a and b == b) or abs(a) == float("inf") or abs(b) == float("inf"):
                problems.append(f"{k}: non-finite (eager={a}, graph={b})")
                continue
            rel = abs(a - b) / max(abs(a), abs(b), 1e-6)
            parts.append(f"{k}={b:.4g}(eager {a:.4g})")
            if rel > self.rtol and abs(a - b) > 1e-3:
                problems.append(f"{k}: eager={a:.6g} graph={b:.6g} rel={rel:.3g}")
        with torch.no_grad():
            for name, params, saved0, pe in (
                ("G", list(self.net_g.parameters()), snap.saved[: snap.n_g], ref["pg"]),
                (
                    "D",
                    list(self.net_d.parameters()),
                    snap.saved[snap.n_g : snap.n_g + snap.n_d],
                    ref["pd"],
                ),
            ):
                num = torch.linalg.vector_norm(
                    torch.stack(torch._foreach_norm(torch._foreach_sub(params, pe)))
                )
                den = torch.linalg.vector_norm(
                    torch.stack(torch._foreach_norm(torch._foreach_sub(pe, saved0)))
                )
                num_f, den_f = float(num), float(den)
                if den_f > 0.0:
                    ratio = num_f / den_f
                    parts.append(f"Δparam_{name}={ratio:.3g}")
                    if not (ratio == ratio) or ratio > self.param_rtol:
                        problems.append(
                            f"parameter update differs for {name}: |graph-eager|/|eager-before|={ratio:.3g}"
                        )
                else:
                    parts.append(f"Δparam_{name}=skipped(lr=0)")
        return (not problems), (", ".join(problems) if problems else ", ".join(parts))

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            _warn(msg)


# --------------------------------------------------------------------------------------
# DDP の代わりに使う薄いラッパ / 有効化判定
# --------------------------------------------------------------------------------------
class SingleProcessWrapper(torch.nn.Module):
    """world_size == 1 のとき DDP の代わりに使う。`.module` で中身を参照できる点
    (checkpoints / safetensors / evaluate が使っている) だけ DDP と互換。
    DDP は 1 プロセスでは何も同期しないのに、フックや requires_grad の固定化で CUDA Graph の邪魔になる。"""

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def parse_mode(value: Optional[str]) -> str:
    v = (value or "auto").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return "on"
    if v in ("0", "false", "no", "off"):
        return "off"
    return "auto"


def decide_cuda_graph(
    mode: str,
    *,
    world_size: int,
    fp16_run: bool,
    has_dur_disc: bool,
    has_wavlm: bool,
    custom_batch_sampler: bool,
    device: Optional[torch.device | str] = None,
) -> tuple[bool, str]:
    """CUDA Graph 経路を使うか。(使う?, 理由) を返す。ここで False なら従来の学習ループがそのまま動く。"""
    mode = parse_mode(mode)
    if mode == "off":
        return False, "disabled by --cuda_graph off / SBV2_CUDA_GRAPH=off"
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    if world_size != 1:
        return False, f"multi-GPU (world_size={world_size}) は対象外"
    if fp16_run:
        return False, "fp16_run (GradScaler) は対象外 (bf16 または fp32 のみ)"
    if has_dur_disc:
        return False, "duration discriminator 使用時は対象外"
    if has_wavlm:
        return False, "WavLM discriminator 使用時は対象外"
    if not custom_batch_sampler:
        return False, "--not_use_custom_batch_sampler では形状を固定できないため対象外"
    dev = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(dev)
    name = torch.cuda.get_device_name(dev)
    if mode == "auto" and major < 9:
        return (
            False,
            f"{name} (sm_{major}{minor}) は GPU 律速なので eager のまま "
            "(強制するには --cuda_graph on)",
        )
    if not hasattr(torch.cuda, "CUDAGraph") or not hasattr(torch.cuda, "graph_pool_handle"):
        return False, f"torch {torch.__version__} は CUDA Graph API が不足"
    reason = probe_capturable_adamw(dev)
    if reason is not None:
        return False, f"capturable AdamW を使えません: {reason}"
    return True, f"{name} (sm_{major}{minor}), torch {torch.__version__}"
