#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sbv2_gpu_diag.py  --  Style-Bert-VITS2 の「H200/B200/B300 が遅い」問題の確定診断ツール

Style-Bert-VITS2 リポジトリ不要。torch だけで動く。
各 GPU (t4/l4/l40s/rtx6000pro/h200/b200/b300) で実行して出力を比較すること。

    python sbv2_gpu_diag.py

測るもの:
  [1] CUDAコア(non-tensor) FP32 と Tensor Core (TF32/BF16) の実効スループット比
  [2] SBV2 JP-Extra の HiFi-GAN Generator 相当の Conv1d スタックを
      fp32 / fp32+cudnn.benchmark / bf16 / bf16+benchmark / compile で計測
  [3] MPD (Conv2d) 同上
  [4] gc.collect() + torch.cuda.empty_cache() の実コスト
  [5] カーネルローンチ律速かどうか (小カーネル連打)
"""

import gc
import math
import time

import torch
import torch.nn as nn
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import weight_norm

DEV = "cuda"
LRELU = 0.1


# --------------------------------------------------------------------------
# 計測ユーティリティ
# --------------------------------------------------------------------------
def bench(fn, warmup=10, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def hdr(s):
    print("\n" + "=" * 74)
    print(s)
    print("=" * 74)


# --------------------------------------------------------------------------
# SBV2 JP-Extra の実コンフィグ
# --------------------------------------------------------------------------
UPSAMPLE_RATES = [8, 8, 2, 2, 2]          # 合計 512x
UPSAMPLE_KERNELS = [16, 16, 8, 2, 2]
UPSAMPLE_INIT_CH = 512
RESBLOCK_KERNELS = [3, 7, 11]
RESBLOCK_DILATIONS = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
INTER_CH = 192
SEGMENT_SIZE = 16384
HOP = 512
MPD_PERIODS = [2, 3, 5, 7, 11]


class ResBlock1(nn.Module):
    def __init__(self, ch, k, d):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [weight_norm(Conv1d(ch, ch, k, 1, dilation=dd, padding=(k * dd - dd) // 2)) for dd in d]
        )
        self.convs2 = nn.ModuleList(
            [weight_norm(Conv1d(ch, ch, k, 1, dilation=1, padding=(k - 1) // 2)) for _ in d]
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU)), LRELU))
            x = xt + x
        return x


class Generator(nn.Module):
    """SBV2 JP-Extra の dec (HiFi-GAN) 相当"""

    def __init__(self):
        super().__init__()
        self.num_kernels = len(RESBLOCK_KERNELS)
        self.conv_pre = Conv1d(INTER_CH, UPSAMPLE_INIT_CH, 7, 1, padding=3)
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(UPSAMPLE_RATES, UPSAMPLE_KERNELS)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        UPSAMPLE_INIT_CH // (2**i),
                        UPSAMPLE_INIT_CH // (2 ** (i + 1)),
                        k, u, padding=(k - u) // 2,
                    )
                )
            )
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = UPSAMPLE_INIT_CH // (2 ** (i + 1))
            for k, d in zip(RESBLOCK_KERNELS, RESBLOCK_DILATIONS):
                self.resblocks.append(ResBlock1(ch, k, d))
        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)

    def forward(self, x):
        x = self.conv_pre(x)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU))
            xs = None
            for j in range(self.num_kernels):
                r = self.resblocks[i * self.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / self.num_kernels
        return torch.tanh(self.conv_post(F.leaky_relu(x)))


class PeriodDisc(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        chs = [(1, 32), (32, 128), (128, 512), (512, 1024), (1024, 1024)]
        self.convs = nn.ModuleList(
            [weight_norm(Conv2d(i, o, (5, 1), (3, 1), padding=(2, 0))) for i, o in chs]
        )
        self.conv_post = weight_norm(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        b, c, t = x.shape
        if t % self.period:
            x = F.pad(x, (0, self.period - (t % self.period)), "reflect")
            t = x.shape[2]
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = F.leaky_relu(l(x), LRELU)
        return self.conv_post(x)


class MPD(nn.Module):
    def __init__(self):
        super().__init__()
        self.ds = nn.ModuleList([PeriodDisc(p) for p in MPD_PERIODS])

    def forward(self, x):
        return [d(x) for d in self.ds]


# --------------------------------------------------------------------------
def gpu_info():
    p = torch.cuda.get_device_properties(0)
    hdr(f"GPU: {p.name}")
    print(f"  torch           : {torch.__version__}  (CUDA {torch.version.cuda})")
    print(f"  compute cap     : sm_{p.major}{p.minor}")
    print(f"  SM count        : {p.multi_processor_count}")
    print(f"  VRAM            : {p.total_memory/1024**3:.1f} GiB")
    try:
        print(f"  clock (max SM)  : {torch.cuda.clock_rate()} MHz")
    except Exception:
        pass
    return p


def bench_raw_flops():
    """FP32(CUDAコア) vs TF32/BF16(Tensor Core) の実効スループット"""
    hdr("[1] 生スループット: CUDAコア FP32 vs Tensor Core")
    n = 8192
    a = torch.randn(n, n, device=DEV)
    b = torch.randn(n, n, device=DEV)
    flop = 2 * n**3

    old_mm, old_cudnn = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    t_fp32 = bench(lambda: torch.mm(a, b), 5, 20)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    t_tf32 = bench(lambda: torch.mm(a, b), 5, 20)

    ab, bb = a.bfloat16(), b.bfloat16()
    t_bf16 = bench(lambda: torch.mm(ab, bb), 5, 20)

    torch.backends.cuda.matmul.allow_tf32 = old_mm
    torch.backends.cudnn.allow_tf32 = old_cudnn

    f32, t32, b16 = (flop / (t * 1e-3) / 1e12 for t in (t_fp32, t_tf32, t_bf16))
    print(f"  FP32 (CUDAコア, TF32無効) : {f32:8.1f} TFLOPS")
    print(f"  TF32 (Tensor Core)        : {t32:8.1f} TFLOPS   ({t32/f32:.1f}x)")
    print(f"  BF16 (Tensor Core)        : {b16:8.1f} TFLOPS   ({b16/f32:.1f}x)")
    print(f"\n  >> この GPU は Tensor Core を使うと FP32 比 {b16/f32:.1f} 倍の余力がある。")
    print("     学習が fp32 のままだと、この倍率ぶんが丸ごと捨てられている。")
    return f32, b16


def bench_model(name, model, make_input, batch):
    print(f"\n  --- {name} (batch={batch}) ---")
    results = {}

    def run(dtype, benchmark, label, compile_=False):
        torch.backends.cudnn.benchmark = benchmark
        torch.cuda.empty_cache()
        m = model
        if compile_:
            m = torch.compile(model, mode="max-autotune-no-cudagraphs", dynamic=False)
        x = make_input(batch)

        def step():
            with torch.autocast("cuda", dtype=dtype, enabled=(dtype != torch.float32)):
                y = m(x)
            s = y.float().square().mean() if torch.is_tensor(y) else sum(
                o.float().square().mean() for o in y
            )
            s.backward()
            model.zero_grad(set_to_none=True)

        try:
            ms = bench(step, warmup=15 if not compile_ else 25, iters=25)
        except Exception as e:  # noqa
            print(f"    {label:<34}:  FAILED ({type(e).__name__})")
            return
        results[label] = ms
        base = results.get("fp32  / cudnn.benchmark=False")
        sp = f"   ({base/ms:.2f}x)" if base and label != "fp32  / cudnn.benchmark=False" else ""
        print(f"    {label:<34}: {ms:8.2f} ms/step{sp}")

    run(torch.float32, False, "fp32  / cudnn.benchmark=False")
    run(torch.float32, True, "fp32  / cudnn.benchmark=True")
    run(torch.bfloat16, True, "bf16  / cudnn.benchmark=True")
    try:
        run(torch.bfloat16, True, "bf16  / benchmark + torch.compile", compile_=True)
    except Exception as e:
        print(f"    compile skipped: {e}")
    return results


def bench_nets(batch):
    hdr(f"[2][3] SBV2 実モデル相当 (segment_size={SEGMENT_SIZE})")
    gen = Generator().to(DEV)
    frames = SEGMENT_SIZE // HOP
    r1 = bench_model(
        "HiFi-GAN Generator (dec)",
        gen,
        lambda b: torch.randn(b, INTER_CH, frames, device=DEV, requires_grad=True),
        batch,
    )
    del gen
    torch.cuda.empty_cache()

    mpd = MPD().to(DEV)
    r2 = bench_model(
        "MultiPeriodDiscriminator",
        mpd,
        lambda b: torch.randn(b, 1, SEGMENT_SIZE, device=DEV, requires_grad=True),
        batch,
    )
    del mpd
    torch.cuda.empty_cache()
    return r1, r2


def bench_empty_cache():
    hdr("[4] gc.collect() + torch.cuda.empty_cache() の実コスト")
    print("    (train_and_evaluate() の末尾で毎エポック呼ばれている)")
    blocks = [torch.empty(int(8e6), device=DEV) for _ in range(120)]
    del blocks
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    gc.collect()
    t_gc = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    t_ec = (time.perf_counter() - t0) * 1e3

    # 解放後に確保し直すコスト (cudaMalloc の再取得)
    t0 = time.perf_counter()
    blocks = [torch.empty(int(8e6), device=DEV) for _ in range(120)]
    torch.cuda.synchronize()
    t_re = (time.perf_counter() - t0) * 1e3
    del blocks

    print(f"    gc.collect()            : {t_gc:8.2f} ms")
    print(f"    empty_cache()           : {t_ec:8.2f} ms")
    print(f"    直後の再確保 (cudaMalloc): {t_re:8.2f} ms")
    total = t_gc + t_ec + t_re
    print(f"    合計                    : {total:8.2f} ms / epoch")
    print(f"    >> 1エポック7ステップなら 1イテレーションあたり +{total/7:.1f} ms")
    return total


def bench_launch_overhead():
    hdr("[5] カーネルローンチ律速かどうか")
    x = torch.randn(64, 64, device=DEV)
    n = 2000
    def many():
        y = x
        for _ in range(n):
            y = y * 1.000001
    ms = bench(many, 3, 10)
    per = ms * 1e3 / n
    print(f"    極小カーネル {n} 本: {ms:.2f} ms  ->  1本あたり {per:.2f} us")
    print("    >> 5us/本 を超えるならホストCPUのローンチが律速。")
    print("       VITS は1ステップで数千〜1万カーネルを発行するため影響が大きい。")
    return per


def main():
    assert torch.cuda.is_available(), "CUDA が見えていない"
    p = gpu_info()
    f32, b16 = bench_raw_flops()

    batch = 8
    r_gen, r_mpd = bench_nets(batch)
    ec = bench_empty_cache()
    launch = bench_launch_overhead()

    hdr("判定")
    base = "fp32  / cudnn.benchmark=False"
    best = "bf16  / benchmark + torch.compile"
    for nm, r in (("Generator", r_gen), ("MPD", r_mpd)):
        if base in r:
            cand = [k for k in (best, "bf16  / cudnn.benchmark=True",
                                "fp32  / cudnn.benchmark=True") if k in r]
            if cand:
                k = min(cand, key=lambda k: r[k])
                print(f"  {nm:<10}: 現状(fp32) {r[base]:.1f} ms -> 最速 {r[k]:.1f} ms "
                      f"({r[base]/r[k]:.2f}x 短縮)  [{k.strip()}]")
    print(f"\n  CUDAコアFP32 {f32:.0f} TFLOPS / Tensor Core BF16 {b16:.0f} TFLOPS "
          f"= 未使用の伸びしろ {b16/f32:.1f}x")
    print(f"  毎エポックの empty_cache コスト: {ec:.0f} ms")
    print(f"  カーネルローンチ: {launch:.2f} us/本")
    print(f"  SM {p.multi_processor_count} 基 / batch={batch} -> "
          f"占有率が低いほど大きい GPU ほど不利")


if __name__ == "__main__":
    main()
