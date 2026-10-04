"""
GPU 同期フリー版のスプライン (style_bert_vits2/models/transforms.py) が、変更前の実装
(tests/reference_transforms.py) と数値的に一致することの確認: 順方向・逆方向・入力/パラメータへの勾配。

学習では SDP の ConvFlow 8 回ぶんがここを通り、推論 (TTS / ONNX) でも同じ関数を使うので、
学習・推論どちらの品質にも影響しないことをここで保証する。
"""

import importlib.util
import os

import pytest
import torch

from style_bert_vits2.models import transforms as new

_REF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_transforms.py")
_spec = importlib.util.spec_from_file_location("reference_transforms", _REF)
old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old)  # type: ignore[union-attr]

NAMES = ("out", "logabsdet", "g_x", "g_w", "g_h", "g_d")


def _run(mod, x, w, h, d, inverse, bound):
    x, w, h, d = (t.clone().requires_grad_(True) for t in (x, w, h, d))
    out, lad = mod.piecewise_rational_quadratic_transform(
        x, w, h, d, inverse=inverse, tails="linear", tail_bound=bound
    )
    g = torch.Generator().manual_seed(7)
    a = torch.randn(out.shape, generator=g, dtype=out.dtype)
    b = torch.randn(lad.shape, generator=g, dtype=lad.dtype)
    ((out * a).sum() + (lad * b).sum()).backward()
    return out.detach(), lad.detach(), x.grad, w.grad, h.grad, d.grad


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-12), (torch.float32, 5e-5)])
@pytest.mark.parametrize("inverse", [False, True])
def test_new_spline_matches_the_original(dtype, tol, inverse):
    bound = 5.0
    worst = dict.fromkeys(NAMES, 0.0)
    for trial in range(20):
        torch.manual_seed(trial)
        B, C, T, K = 3, 1, 23, 10
        scale = bound * (1.3 if inverse else 1.6)  # 区間外 (線形部分) の要素も混ぜる
        x = (torch.rand(B, C, T, dtype=dtype) * 2 - 1) * scale
        w = torch.randn(B, C, T, K, dtype=dtype)
        h = torch.randn(B, C, T, K, dtype=dtype)
        d = torch.randn(B, C, T, K - 1, dtype=dtype)
        ref = _run(old, x, w, h, d, inverse, bound)
        got = _run(new, x, w, h, d, inverse, bound)
        for name, a, b in zip(NAMES, ref, got):
            assert torch.isfinite(b).all(), (name, "non-finite in the new implementation")
            worst[name] = max(worst[name], (a - b).abs().max().item())
    assert all(v <= tol for v in worst.values()), worst


def _edge_inputs(scale):
    x = torch.full((2, 1, 5), scale)
    return x, torch.randn(2, 1, 5, 10), torch.randn(2, 1, 5, 10), torch.randn(2, 1, 5, 9)


def test_all_inside_edge_case_matches_the_original():
    x, w, h, d = _edge_inputs(0.5)
    ref = _run(old, x, w, h, d, False, 5.0)
    got = _run(new, x, w, h, d, False, 5.0)
    for name, a, b in zip(NAMES, ref, got):
        assert torch.isfinite(b).all(), name
        assert (a - b).abs().max().item() <= 5e-5, name


def test_all_outside_edge_case_is_identity():
    """全要素が区間外のとき、変更前の実装は空テンソルへの min() で RuntimeError になる (潜在バグ)。
    新しい実装は恒等写像 (出力=入力, logabsdet=0) を返し、勾配も有限。"""
    x, w, h, d = _edge_inputs(10.0)
    with pytest.raises(RuntimeError):
        _run(old, x, w, h, d, False, 5.0)
    out, lad, g_x, g_w, g_h, g_d = _run(new, x, w, h, d, False, 5.0)
    assert torch.equal(out, x)
    assert torch.count_nonzero(lad) == 0
    assert all(torch.isfinite(t).all() for t in (g_x, g_w, g_h, g_d))
