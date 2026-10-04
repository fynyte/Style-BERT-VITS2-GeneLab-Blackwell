"""
ディスパッチ層で「GPU 上では CPU-GPU 同期 (または H2D / D2H 転送) になる操作」を検出するプローブ。

GPU が無くても、`meta` デバイス (形状だけを持ち、計算しないテンソル) でモデルを動かしながら
TorchDispatchMode で全 aten 演算を見張ることで、次の 4 種類を数えられる。

  scalar_read          テンソルの値を Python に読み出す (`.item()` / `float(t)` / `if t:` など)
                       → CUDA では GPU の完了待ち (同期) になる
  data_dependent_shape 出力の形がデータで決まる演算 (nonzero / masked_select / unique / bincount ...)
                       → CUDA では結果の個数を読み戻すため同期する
  mask_indexing        bool マスクによる添字 (`x[mask]`, `x[mask] = v`)。内部で nonzero を呼ぶ
  transfer             CPU <-> デバイス間のコピー (`torch.zeros(n).to(device)` / `.cpu()` など)

CUDA Graph はこれらを 1 つでも含むと capture できない (または同期で台無しになる) ので、
train_step_body がこれらを含まないことをテストで保証する (tests/test_no_sync_meta.py)。

注意: これは「aten 演算レベル」の検査で、CUDA ランタイム (cudaMalloc の初回呼び出し、cuDNN の
benchmark など) が内部で行う同期までは見えない。それらは warm-up 実行で済ませる設計になっている。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode

aten = torch.ops.aten


def _ops(*names: str) -> set:
    """aten.<name>.<overload> を (存在するものだけ) 全オーバーロード集める (torch のバージョン差を吸収)。"""
    found = set()
    for name in names:
        packet = getattr(aten, name, None)
        if packet is None:
            continue
        for ov in packet.overloads():
            if ov.endswith("out"):
                continue
            found.add(getattr(packet, ov))
    return found


SCALAR_READ_OPS = _ops("_local_scalar_dense")
DATA_DEPENDENT_SHAPE_OPS = _ops(
    "nonzero",
    "masked_select",
    "_unique2",
    "unique_dim",
    "unique_consecutive",
    "bincount",
    "equal",
)
# repeat_interleave(Tensor) は output_size を渡さないとサイズを読み戻す。self_int 版は固定サイズなので除外
DATA_DEPENDENT_SHAPE_OPS |= {aten.repeat_interleave.Tensor}
INDEXING_OPS = _ops("index", "_unsafe_index", "index_put_", "index_put", "_index_put_impl_")
_MASK_DTYPES = (torch.bool, torch.uint8, torch.int8)


@dataclass(frozen=True)
class Finding:
    kind: str  # scalar_read / data_dependent_shape / mask_indexing / transfer
    op: str
    detail: str

    def __str__(self) -> str:  # pragma: no cover - 表示用
        return f"{self.kind}: {self.op} {self.detail}"


def _placeholder_scalar(t: torch.Tensor) -> Any:
    """meta テンソルは値を持たないので、scalar_read を検出した後に処理を続けるための仮の値を返す。"""
    if t.dtype == torch.bool:
        return False
    if t.dtype.is_floating_point:
        return 0.0
    if t.dtype.is_complex:
        return 0j
    return 0


class SyncProbe(TorchDispatchMode):
    """with SyncProbe() as probe: ... の間に発行された aten 演算を見張る。

    scalar_placeholder=True のときは scalar_read を記録した上で仮の値 (0 / 0.0 / False) を返し、
    処理を最後まで続ける (旧コードに何箇所あるかを数える対照実験用)。False のときは meta テンソルからの
    読み出しがそのままエラーになる。
    """

    def __init__(self, scalar_placeholder: bool = False):
        super().__init__()
        self.scalar_placeholder = scalar_placeholder
        self.findings: list[Finding] = []
        self.n_ops = 0

    # ---- 集計 ----
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.kind] = out.get(f.kind, 0) + 1
        return out

    def summary(self) -> str:
        return f"ops={self.n_ops} findings={self.counts() or 0}"

    # ---- 本体 ----
    def _add(self, kind: str, func: Any, detail: str = "") -> None:
        self.findings.append(Finding(kind, str(func), detail))

    @staticmethod
    def _has_mask_index(func: Any, args: tuple) -> bool:
        indices = args[1] if len(args) > 1 else None
        if not isinstance(indices, (list, tuple)):
            return False
        return any(isinstance(i, torch.Tensor) and i.dtype in _MASK_DTYPES for i in indices)

    def __torch_dispatch__(self, func: Any, types: Any, args: tuple = (), kwargs: Optional[dict] = None):
        kwargs = kwargs or {}
        self.n_ops += 1
        if func in SCALAR_READ_OPS:
            t = args[0]
            self._add("scalar_read", func, f"{tuple(t.shape)} {t.dtype} on {t.device}")
            if self.scalar_placeholder:
                return _placeholder_scalar(t)
        elif func in DATA_DEPENDENT_SHAPE_OPS:
            self._add("data_dependent_shape", func)
        elif func in INDEXING_OPS:
            if self._has_mask_index(func, args):
                self._add("mask_indexing", func)
        elif func is aten._to_copy.default:
            src = args[0]
            dst = kwargs.get("device", None)
            if dst is not None and torch.device(dst).type != src.device.type:
                self._add("transfer", func, f"{src.device} -> {torch.device(dst)}")
        elif func is aten.copy_.default:
            dst_t, src_t = args[0], args[1]
            if isinstance(src_t, torch.Tensor) and dst_t.device.type != src_t.device.type:
                self._add("transfer", func, f"{src_t.device} -> {dst_t.device}")
        return func(*args, **kwargs)
