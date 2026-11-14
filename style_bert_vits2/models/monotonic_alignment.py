"""
以下に記述されている関数のコメントはリファクタリング時に GPT-4 に生成させたもので、
コードと完全に一致している保証はない。あくまで参考程度とすること。
"""

from typing import Any, Optional

import numba
import torch
from numpy import float32, int32, zeros

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False

try:
    from super_monotonic_align import maximum_path as _sma_maximum_path

    _SMA_AVAILABLE = True
except ImportError:
    _SMA_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _maximum_path_triton_kernel(
        path_ptr,
        neg_cent_ptr,
        mask_ptr,
        val_ptr,
        stride_pb,
        stride_py,
        stride_px,
        stride_nb,
        stride_ny,
        stride_nx,
        stride_mb,
        stride_my,
        stride_mx,
        stride_vb,
        stride_vy,
        T_y: int,
        T_x: int,
        BLOCK_X: tl.constexpr,
    ):
        b = tl.program_id(0)
        offs_x = tl.arange(0, BLOCK_X)

        m_x = tl.load(
            mask_ptr + b * stride_mb + offs_x * stride_mx,
            mask=offs_x < T_x,
            other=0,
        )
        t_x = tl.sum((m_x != 0).to(tl.int32), axis=0)

        t_y = 0
        for y0 in range(0, T_y, BLOCK_X):
            oy = y0 + offs_x
            m_y = tl.load(
                mask_ptr + b * stride_mb + oy * stride_my,
                mask=oy < T_y,
                other=0,
            )
            t_y += tl.sum((m_y != 0).to(tl.int32), axis=0)

        max_neg_val = -1e9
        n_base = neg_cent_ptr + b * stride_nb
        v_base = val_ptr + b * stride_vb

        for y in range(t_y):
            x_lo = tl.maximum(0, t_x + y - t_y)
            x_hi = tl.minimum(t_x, y + 1)
            active = (offs_x >= x_lo) & (offs_x < x_hi)

            nc = tl.load(
                n_base + y * stride_ny + offs_x * stride_nx,
                mask=active,
                other=0.0,
            ).to(tl.float32)

            v_cur = tl.load(
                v_base + (y - 1) * stride_vy + offs_x,
                mask=active & (y > 0) & (offs_x != y),
                other=max_neg_val,
            )
            v_prev_loaded = tl.load(
                v_base + (y - 1) * stride_vy + (offs_x - 1),
                mask=active & (y > 0) & (offs_x > 0),
                other=max_neg_val,
            )
            v_prev = tl.where((y == 0) & (offs_x == 0), 0.0, v_prev_loaded)

            val = nc + tl.maximum(v_prev, v_cur)
            tl.store(
                v_base + y * stride_vy + offs_x,
                val,
                mask=active,
            )
            tl.debug_barrier()

        p_base = path_ptr + b * stride_pb
        index = t_x - 1
        for y_rev in range(t_y):
            y = t_y - 1 - y_rev
            tl.store(p_base + y * stride_py + index * stride_px, 1, mask=t_x > 0)
            can_step = (y > 0) & (index != 0)
            v_c = tl.load(
                v_base + (y - 1) * stride_vy + index,
                mask=can_step,
                other=0.0,
            )
            v_p = tl.load(
                v_base + (y - 1) * stride_vy + (index - 1),
                mask=can_step,
                other=0.0,
            )
            step = can_step & ((index == y) | (v_c < v_p))
            index = tl.where(step, index - 1, index)


def _maximum_path_triton(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    B, T_y, T_x = neg_cent.shape
    path = torch.zeros((B, T_y, T_x), dtype=neg_cent.dtype, device=neg_cent.device)
    if B == 0 or T_y == 0 or T_x == 0:
        return path
    val_buf = torch.empty((B, T_y, T_x), dtype=torch.float32, device=neg_cent.device)
    block_x = max(128, triton.next_power_of_2(T_x))
    _maximum_path_triton_kernel[(B,)](
        path,
        neg_cent,
        mask,
        val_buf,
        path.stride(0),
        path.stride(1),
        path.stride(2),
        neg_cent.stride(0),
        neg_cent.stride(1),
        neg_cent.stride(2),
        mask.stride(0),
        mask.stride(1),
        mask.stride(2),
        val_buf.stride(0),
        val_buf.stride(1),
        T_y=T_y,
        T_x=T_x,
        BLOCK_X=block_x,
        num_warps=1,
    )
    return path


def _maximum_path_numba(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent_np = neg_cent.data.cpu().numpy().astype(float32)
    path = zeros(neg_cent_np.shape, dtype=int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(int32)
    __maximum_path_jit(path, neg_cent_np, t_t_max, t_s_max)

    return torch.from_numpy(path).to(device=device, dtype=dtype)


def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    sbv2/VITS 側: neg_cent, mask は [B, T_mel, T_text]
    GPU 上では転置コピー不要の単一 Triton カーネルを優先使用し、
    利用不可の場合は super-monotonic-align または Numba JIT にフォールバックする。
    """
    if neg_cent.is_cuda:
        if _TRITON_AVAILABLE and neg_cent.size(2) <= 2048 and neg_cent.size(1) <= 8192:
            return _maximum_path_triton(neg_cent, mask)
        if _SMA_AVAILABLE:
            dtype = neg_cent.dtype
            neg_cent_t = neg_cent.transpose(1, 2).float()
            mask_t = mask.transpose(1, 2).to(torch.int32)
            path_t = _sma_maximum_path(neg_cent_t, mask_t, dtype=torch.float32)
            return path_t.transpose(1, 2).to(dtype)
    return _maximum_path_numba(neg_cent, mask)


@numba.jit(
    numba.void(
        numba.int32[:, :, ::1],
        numba.float32[:, :, ::1],
        numba.int32[::1],
        numba.int32[::1],
    ),
    nopython=True,
    nogil=True,
)  # type: ignore
def __maximum_path_jit(paths: Any, values: Any, t_ys: Any, t_xs: Any) -> None:
    """
    与えられたパス、値、およびターゲットの y と x 座標を使用して JIT で最大パスを計算する
    """
    b = paths.shape[0]
    max_neg_val = -1e9
    for i in range(int(b)):
        path = paths[i]
        value = values[i]
        t_y = t_ys[i]
        t_x = t_xs[i]

        v_prev = v_cur = 0.0
        index = t_x - 1

        for y in range(t_y):
            for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
                if x == y:
                    v_cur = max_neg_val
                else:
                    v_cur = value[y - 1, x]
                if x == 0:
                    if y == 0:
                        v_prev = 0.0
                    else:
                        v_prev = max_neg_val
                else:
                    v_prev = value[y - 1, x - 1]
                value[y, x] += max(v_prev, v_cur)

        for y in range(t_y - 1, -1, -1):
            path[y, index] = 1
            if index != 0 and (
                index == y or value[y - 1, index] < value[y - 1, index - 1]
            ):
                index = index - 1
