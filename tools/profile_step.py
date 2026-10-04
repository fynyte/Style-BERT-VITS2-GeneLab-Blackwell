#!/usr/bin/env python
"""
学習 1 step の診断・計測スクリプト (GPU 1 枚・データセット不要・数分)。

「H200 / B200 / B300 で step が遅い」原因 (= GPU ではなく CPU が 1 step ≒ 12 万個の演算を発行する速度で
律速されている) を、実機で直接確かめるためのもの。次を順に出力する:

  [1] 環境              GPU / CPU / torch / コンテナの CPU クォータ
  [2] 発行コスト        演算 1 個を GPU に発行する CPU 側の時間 (µs)。これが大きいホストほど eager の step が遅い
  [3] eager 計測        修正後の train_step_body を普通に (CUDA Graph 無しで) 回したときの s/step
  [4] 同期チェック      --sync-check: torch.cuda.set_sync_debug_mode("warn") で GPU 同期が残っていないか
  [5] プロファイル      --profile:    1 step の CUDA カーネル数 / GPU が実際に計算している時間 (= GPU 稼働率)
  [6] CUDA Graph 計測   capture → 自己検証 → replay の s/step
  [7] 判定              eager と graph の差 = ホスト (CPU) オーバーヘッド

使い方 (リポジトリ直下で):
    python tools/profile_step.py --config configs/config_jp_extra.json --batch-size 8
    python tools/profile_step.py --batch-size 8 --spec-len 500 --text-len 150 --sync-check --profile

--batch-size は学習時の batch_size に合わせる。--spec-len はスペクトログラムのフレーム数
(sampling_rate / hop_length = 約 86 フレーム/秒)、--text-len は音素列の長さ (add_blank 後)。
結果をそのまま貼り付けてもらえれば原因の切り分けができる。--json で機械可読な結果も保存できる。

学習スクリプトと同じモデル構成 (config の model / train / data) を使うが、duration / WavLM discriminator は
CUDA Graph 経路の対象外なので無効にして測る。GPU が無い環境では --device cpu --tiny で動作確認だけできる
(その場合 CUDA Graph は偽物のバックエンドで流れ、時間の数値に意味は無い)。
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import platform
import statistics
import sys
import time
import types
import warnings
from collections import Counter
from typing import Any, Callable

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import importlib.util  # noqa: E402


def _stub_if_missing(name: str, **attrs: Any) -> None:
    """losses.py が import するだけで使わない重い依存 (WavLM 用) が無い環境でも動かすための保険。"""
    if name in sys.modules:
        return
    try:
        if importlib.util.find_spec(name) is not None:
            return
    except (ValueError, ImportError):
        pass
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod


_stub_if_missing("torchaudio")
_stub_if_missing("transformers", AutoModel=object)
_stub_if_missing("super_monotonic_align", maximum_path=lambda *a, **k: None)

import torch  # noqa: E402

import cuda_graph_step as CG  # noqa: E402
from style_bert_vits2.models.hyper_parameters import HyperParameters  # noqa: E402
from style_bert_vits2.models.models_jp_extra import (  # noqa: E402
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from style_bert_vits2.nlp.symbols import (  # noqa: E402
    NUM_LANGUAGES,
    NUM_TONES,
    SYMBOLS,
)

RESULT: dict[str, Any] = {}


def _install_cpu_mas() -> None:
    """CUDA が無い環境 (動作確認用) では、Triton 版 MAS の代わりに numba 版を使う。"""
    import numpy as np

    from style_bert_vits2.models import monotonic_alignment as ma

    def maximum_path(neg_cent: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        device, dtype = neg_cent.device, neg_cent.dtype
        nc = neg_cent.data.cpu().numpy().astype(np.float32)
        path = np.zeros(nc.shape, dtype=np.int32)
        t_t = mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32)
        t_s = mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32)
        getattr(ma, "__maximum_path_jit")(path, nc, t_t, t_s)
        return torch.from_numpy(path).to(device=device, dtype=dtype)

    ma.maximum_path = maximum_path


def say(msg: str = "") -> None:
    print(msg, flush=True)


def head(title: str) -> None:
    say()
    say(f"=== {title} " + "=" * max(3, 76 - len(title)))


# ------------------------------------------------------------------------------------------------
# [1] 環境
# ------------------------------------------------------------------------------------------------
def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def cpu_quota() -> str:
    """コンテナの CPU 上限 (cgroup)。Modal / Lightning などでは vCPU 数が絞られていることがある。"""
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:  # cgroup v2
            quota, period = f.read().split()[:2]
        if quota == "max":
            return "無制限"
        return f"{int(quota) / int(period):.2f} CPU"
    except (OSError, ValueError):
        pass
    try:  # cgroup v1
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as f:
            quota = int(f.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as f:
            period = int(f.read())
        return "無制限" if quota < 0 else f"{quota / period:.2f} CPU"
    except (OSError, ValueError):
        return "不明"


def report_environment(device: torch.device) -> None:
    head("[1] 環境")
    say(f"python {platform.python_version()} | torch {torch.__version__} | cuda {torch.version.cuda}")
    try:
        import triton  # type: ignore

        say(f"triton {triton.__version__}")
    except Exception:  # noqa: BLE001
        say("triton: (未インストール)")
    env: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cpu_model": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "cpu_quota": cpu_quota(),
    }
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        env.update(gpu=p.name, sm=f"{p.major}.{p.minor}", mem_gb=round(p.total_memory / 2**30, 1), sms=p.multi_processor_count)
        say(f"GPU: {p.name} (sm_{p.major}{p.minor}), {p.total_memory / 2**30:.0f} GiB, {p.multi_processor_count} SM")
    else:
        say(f"device: {device} (CUDA なし。動作確認のみ)")
    say(f"CPU: {env['cpu_model']} | 論理CPU {env['logical_cpus']} | cgroup 上限: {env['cpu_quota']}")
    interesting = (
        "CUDA_LAUNCH_BLOCKING",
        "PYTORCH_CUDA_ALLOC_CONF",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "CUDA_MODULE_LOADING",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "SBV2_CUDA_GRAPH",
        "SBV2_NUM_WORKERS",
    )
    set_vars = {k: os.environ[k] for k in interesting if k in os.environ}
    if set_vars:
        say("環境変数: " + ", ".join(f"{k}={v}" for k, v in set_vars.items()))
        env["env_vars"] = set_vars
    if os.environ.get("CUDA_LAUNCH_BLOCKING", "0") not in ("", "0"):
        say("★ CUDA_LAUNCH_BLOCKING が有効です。すべてのカーネル起動が同期実行になり、極端に遅くなります (unset してください)。")
    if "expandable_segments:True" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        say("※ expandable_segments:True は CUDA Graph の capture と相性が悪い場合があります (失敗したら自動で eager に戻ります)。")
    RESULT["environment"] = env


# ------------------------------------------------------------------------------------------------
# [2] 発行コスト
# ------------------------------------------------------------------------------------------------
def measure_issue_cost(device: torch.device, n: int = 20000) -> dict[str, float]:
    """演算 1 個の発行にかかる CPU 時間。GPU の速さには依存しない (カーネルは極小)。"""
    head("[2] 演算 1 個を発行する CPU 側コスト (小さいほど良い)")
    sync = _sync_fn(device)
    out: dict[str, float] = {}
    x = torch.zeros(256, device=device)
    for _ in range(500):
        x.add_(1.0)
    sync()
    t0 = time.perf_counter()
    for _ in range(n):
        x.add_(1.0)
    t1 = time.perf_counter()
    sync()
    out["plain_op_us"] = (t1 - t0) / n * 1e6

    w = torch.ones(256, device=device, requires_grad=True)
    for _ in range(200):
        (w * x + 1.0).sum()
    sync()
    m = n // 4
    t0 = time.perf_counter()
    for _ in range(m):
        w * x + 1.0  # 順伝播 2 演算 + autograd のグラフ記録
    t1 = time.perf_counter()
    sync()
    out["autograd_op_us"] = (t1 - t0) / (2 * m) * 1e6
    say(f"素の演算 (x.add_)            : {out['plain_op_us']:.2f} µs/個")
    say(f"autograd 付きの演算 (w*x+1)   : {out['autograd_op_us']:.2f} µs/個")
    say("→ 1 step は約 12 万演算・約 3 万カーネル。eager の step 時間の下限 ≒ 演算数 × この値。")
    say("  (H100 世代の標準的な CPU で 5〜8 µs/個。これが 10 µs/個を超えるホストは eager が遅くなる)")
    RESULT["issue_cost"] = out
    return out


# ------------------------------------------------------------------------------------------------
# モデル / 合成バッチ
# ------------------------------------------------------------------------------------------------
def _sync_fn(device: torch.device) -> Callable[[], None]:
    if device.type == "cuda":
        return lambda: torch.cuda.synchronize(device)
    return lambda: None


def build_models(hps: Any, device: torch.device, tiny: bool):
    m = hps.model
    inter, hidden, filt, up0, gin = (
        (32, 32, 64, 64, 32) if tiny else (m.inter_channels, m.hidden_channels, m.filter_channels, m.upsample_initial_channel, m.gin_channels)
    )
    torch.manual_seed(hps.train.seed)
    use_noise = bool(m.use_noise_scaled_mas)
    net_g = SynthesizerTrn(
        len(SYMBOLS),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=max(1, hps.data.n_speakers),
        mas_noise_scale_initial=0.01 if use_noise else 0.0,
        noise_scale_delta=2e-6 if use_noise else 0.0,
        use_spk_conditioned_encoder=m.use_spk_conditioned_encoder,
        use_noise_scaled_mas=m.use_noise_scaled_mas,
        use_mel_posterior_encoder=m.use_mel_posterior_encoder,
        use_duration_discriminator=False,
        use_wavlm_discriminator=False,
        inter_channels=inter,
        hidden_channels=hidden,
        filter_channels=filt,
        n_heads=m.n_heads,
        n_layers=m.n_layers,
        kernel_size=m.kernel_size,
        p_dropout=m.p_dropout,
        resblock=m.resblock,
        resblock_kernel_sizes=m.resblock_kernel_sizes,
        resblock_dilation_sizes=m.resblock_dilation_sizes,
        upsample_rates=m.upsample_rates,
        upsample_initial_channel=up0,
        upsample_kernel_sizes=m.upsample_kernel_sizes,
        n_layers_q=m.n_layers_q,
        use_spectral_norm=m.use_spectral_norm,
        gin_channels=gin,
        slm=m.slm,
    ).to(device)
    net_d = MultiPeriodDiscriminator(m.use_spectral_norm).to(device)
    kw = dict(betas=hps.train.betas, eps=hps.train.eps)
    optim_g = torch.optim.AdamW(filter(lambda p: p.requires_grad, net_g.parameters()), hps.train.learning_rate, **kw)
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, **kw)
    net_g.train()
    net_d.train()
    return net_g, net_d, optim_g, optim_d


def make_batch(hps: Any, batch_size: int, spec_len: int, text_len: int, seed: int = 0) -> list[torch.Tensor]:
    """data_utils.TextAudioSpeakerCollate が返すものと同じ形・同じ dtype の合成バッチ (CPU, pinned)。"""
    g = torch.Generator().manual_seed(seed)
    B, T, L = batch_size, spec_len, text_len
    hop = hps.data.hop_length
    spec_ch = hps.data.filter_length // 2 + 1
    x = torch.randint(1, len(SYMBOLS), (B, L), generator=g)
    tone = torch.randint(0, NUM_TONES, (B, L), generator=g)
    lang = torch.randint(0, NUM_LANGUAGES, (B, L), generator=g)
    bert = torch.randn(B, 1024, L, generator=g)
    spec = torch.rand(B, spec_ch, T, generator=g) * 0.1 + 1e-3
    y = (torch.rand(B, 1, T * hop, generator=g) - 0.5) * 0.2
    batch = [
        x,
        torch.full((B,), L, dtype=torch.long),
        spec,
        torch.full((B,), T, dtype=torch.long),
        y,
        torch.full((B,), T * hop, dtype=torch.long),
        torch.zeros(B, dtype=torch.long),
        tone,
        lang,
        bert,
        torch.randn(B, 256, generator=g),
    ]
    if torch.cuda.is_available():
        batch = [t.pin_memory() for t in batch]
    return batch


# ------------------------------------------------------------------------------------------------
# 計測
# ------------------------------------------------------------------------------------------------
def time_chunks(fn: Callable[[], Any], sync: Callable[[], None], warmup: int, steps: int, chunks: int = 4) -> list[float]:
    """steps 回を chunks 個に分けて (チャンクごとに 1 回だけ同期) 1 step あたりの秒数を返す。
    学習ループと同じく step ごとには同期しない。"""
    for _ in range(warmup):
        fn()
    sync()
    per = max(1, steps // chunks)
    out = []
    for _ in range(chunks):
        t0 = time.perf_counter()
        for _ in range(per):
            fn()
        sync()
        out.append((time.perf_counter() - t0) / per)
    return out


def fmt_stats(sec: list[float]) -> str:
    return f"{statistics.mean(sec):.3f} s/step (min {min(sec):.3f}, max {max(sec):.3f})"


def finite_report(outs: CG.StepOutputs) -> str:
    v = outs.to_floats()
    keys = ("loss_disc_all", "loss_gen_all", "loss_mel", "loss_kl", "loss_dur", "loss_fm")
    ok = all(v[k] == v[k] and abs(v[k]) != float("inf") for k in keys if k in v)
    return ("有限" if ok else "★非有限の値あり★") + " | " + ", ".join(f"{k}={v[k]:.3f}" for k in keys if k in v)


def run_sync_check(step_fn: Callable[[], Any], device: torch.device) -> dict[str, Any]:
    head("[4] 同期チェック (torch.cuda.set_sync_debug_mode)")
    res: dict[str, Any] = {}
    if device.type != "cuda":
        say("CUDA が無いのでスキップ")
        return res
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(2):
                step_fn()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    syncs = [w for w in caught if "synchroniz" in str(w.message).lower()]
    res["n_sync_warnings_in_2_steps"] = len(syncs)
    if not syncs:
        say("2 step の間に GPU 同期は検出されませんでした (期待どおり)")
    else:
        say(f"2 step の間に GPU 同期が {len(syncs)} 回検出されました。発生箇所 (上位):")
        where = Counter(f"{w.filename}:{w.lineno}" for w in syncs)
        for loc, n in where.most_common(8):
            say(f"   {n:4d} 回  {loc}")
    RESULT["sync_check"] = res
    return res


def _dev_time_us(e: Any) -> float:
    for attr in ("self_device_time_total", "self_cuda_time_total", "device_time_total", "cuda_time_total"):
        v = getattr(e, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def run_profile(step_fn: Callable[[], Any], device: torch.device, eager_sec: float, n_steps: int = 2) -> dict[str, Any]:
    head("[5] プロファイル (1 step の CUDA カーネル数と GPU が実際に計算している時間)")
    res: dict[str, Any] = {}
    if device.type != "cuda":
        say("CUDA が無いのでスキップ")
        return res
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(n_steps):
                step_fn()
            torch.cuda.synchronize(device)
        dev_type = torch.autograd.DeviceType.CUDA
        kernels = [e for e in prof.events() if e.device_type == dev_type]
        n_k = len(kernels) / n_steps
        busy_us = sum(_dev_time_us(e) for e in kernels) / n_steps
        res["kernels_per_step"] = round(n_k)
        res["gpu_busy_ms_per_step"] = round(busy_us / 1e3, 1)
        say(f"CUDA カーネル/memcpy 数: {n_k:,.0f} 個/step")
        say(f"GPU の実計算時間: {busy_us / 1e3:,.0f} ms/step")
        if eager_sec > 0:
            pct = busy_us / 1e6 / eager_sec * 100
            res["gpu_busy_pct_of_eager"] = round(pct, 1)
            say(f"→ eager の 1 step ({eager_sec * 1e3:,.0f} ms) のうち GPU が計算しているのは約 {pct:.0f}%。残りは GPU が次の発行を待っている時間")
        if n_k > 0 and eager_sec > 0:
            say(f"→ 発行コストの実効値 ≒ {eager_sec / n_k * 1e6:.1f} µs/カーネル")
        top = Counter()
        for e in kernels:
            top[e.name[:70]] += _dev_time_us(e) / n_steps
        say("GPU 時間の上位カーネル:")
        for name, us in top.most_common(8):
            say(f"   {us / 1e3:8.1f} ms  {name}")
    except Exception as e:  # noqa: BLE001
        say(f"プロファイラが使えませんでした ({type(e).__name__}: {e})")
        res["error"] = f"{type(e).__name__}: {e}"
    RESULT["profile"] = res
    return res


# ------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="SBV2 学習 1 step の eager / CUDA Graph 計測")
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "config_jp_extra.json"))
    ap.add_argument("--batch-size", type=int, default=None, help="学習時の batch_size (既定: config の値)")
    ap.add_argument("--spec-len", type=int, default=400, help="スペクトログラムのフレーム数 (400 ≒ 4.6 秒)")
    ap.add_argument("--text-len", type=int, default=120, help="音素列の長さ (add_blank 後)")
    ap.add_argument("--dtype", choices=["config", "bf16", "fp32"], default="config")
    ap.add_argument("--steps", type=int, default=16, help="計測する step 数 (4 チャンクに分けて平均)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default=None, help="cuda / cpu (既定: cuda があれば cuda)")
    ap.add_argument("--tiny", action="store_true", help="モデル幅を縮小する (CPU での動作確認用)")
    ap.add_argument("--sync-check", action="store_true", help="GPU 同期が残っていないか確認する")
    ap.add_argument("--profile", action="store_true", help="torch.profiler でカーネル数と GPU 稼働時間を測る")
    ap.add_argument("--skip-eager", action="store_true")
    ap.add_argument("--skip-graph", action="store_true")
    ap.add_argument("--json", default=None, help="結果を JSON で保存するパス")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    else:
        _install_cpu_mas()
    sync = _sync_fn(device)

    hps = HyperParameters.load_from_json(args.config)
    if args.tiny:
        hps.train.segment_size = 2048
    B = args.batch_size or hps.train.batch_size
    if args.dtype == "config":
        amp = bool(hps.train.bf16_run) and not hps.train.fp16_run
    else:
        amp = args.dtype == "bf16"
    amp_dtype = torch.bfloat16 if amp else torch.float32
    if args.spec_len * hps.data.hop_length < hps.train.segment_size:
        say(f"--spec-len が短すぎます (segment_size={hps.train.segment_size} サンプル = {hps.train.segment_size // hps.data.hop_length} フレーム以上が必要)")
        return 2

    report_environment(device)
    RESULT["settings"] = dict(batch_size=B, spec_len=args.spec_len, text_len=args.text_len, amp=("bf16" if amp else "fp32"), steps=args.steps)
    say(f"設定: batch_size={B}, spec_len={args.spec_len} フレーム, text_len={args.text_len}, 精度={'bf16' if amp else 'fp32'}")
    if device.type == "cuda":
        measure_issue_cost(device)

    net_g, net_d, optim_g, optim_d = build_models(hps, device, args.tiny)
    batch_cpu = make_batch(hps, B, args.spec_len, args.text_len)
    cfg = dataclasses.replace(
        CG.StepConfig.from_hps(hps, amp_enabled=amp, amp_dtype=amp_dtype, device_type=device.type),
    )
    mas = torch.tensor(0.01, device=device)

    def eager_step() -> CG.StepOutputs:
        batch = [t.to(device, non_blocking=True) for t in batch_cpu]
        return CG.train_step_body(net_g, net_d, optim_g, optim_d, batch, mas, cfg, want_grad_norms=False)

    eager_mean = 0.0
    if not args.skip_eager:
        head("[3] eager 計測 (CUDA Graph 無し。修正後の train_step_body を普通に実行)")
        try:
            t0 = time.perf_counter()
            outs = eager_step()
            sync()
            say(f"初回 step (cuDNN/Triton 初期化込み): {time.perf_counter() - t0:.1f} s")
            sec = time_chunks(eager_step, sync, args.warmup, args.steps)
            eager_mean = statistics.mean(sec)
            say(f"eager: {fmt_stats(sec)}")
            say(f"損失 ({finite_report(outs)})")
            RESULT["eager_s_per_step"] = round(eager_mean, 4)
            if device.type == "cuda":
                RESULT["max_mem_gib_after_eager"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
                say(f"GPU メモリ最大: {RESULT['max_mem_gib_after_eager']:.1f} GiB")
        except torch.cuda.OutOfMemoryError:
            say("★ GPU メモリ不足です。--batch-size か --spec-len を小さくしてください。")
            return 1
        if args.sync_check:
            run_sync_check(eager_step, device)
        if args.profile:
            run_profile(eager_step, device, eager_mean)

    if not args.skip_graph:
        head("[6] CUDA Graph 計測 (capture → 自己検証 → replay)")
        graph_mean = 0.0
        backend = None if device.type == "cuda" else CG.FakeBackend()
        try:
            runner = CG.GraphedTrainStep(net_g, net_d, optim_g, optim_d, cfg, backend=backend, device=device)
            t0 = time.perf_counter()
            runner.run(batch_cpu, 0.01, True)  # 1 回目: eager (ウォームアップ)
            sync()
            say(f"1 回目 (eager ウォームアップ): {time.perf_counter() - t0:.1f} s")
            t0 = time.perf_counter()
            runner.run(batch_cpu, 0.01, True)  # 2 回目: capture (+ 初回の自己検証)
            sync()
            t_cap = time.perf_counter() - t0
            ready = [e for e in runner.entries.values() if e.state == "ready"]
            say(f"2 回目 (capture + 自己検証): {t_cap:.1f} s | {runner.summary()}")
            RESULT["graph_capture_s"] = round(t_cap, 2)
            if not ready or runner.disabled:
                say("★ CUDA Graph は有効になりませんでした (上のログの [cuda-graph] 警告に理由があります)。")
                RESULT["graph"] = "unavailable"
            else:
                sec = time_chunks(lambda: runner.run(batch_cpu, 0.01, False), sync, args.warmup, args.steps)
                graph_mean = statistics.mean(sec)
                outs = runner.run(batch_cpu, 0.01, True)
                sync()
                say(f"graph: {fmt_stats(sec)}")
                say(f"損失 ({finite_report(outs)})")
                RESULT["graph_s_per_step"] = round(graph_mean, 4)
                if device.type == "cuda":
                    RESULT["max_mem_gib_after_graph"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
                    say(f"GPU メモリ最大: {RESULT['max_mem_gib_after_graph']:.1f} GiB (graph 用プール込み)")
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            say(f"★ CUDA Graph の計測中に例外: {type(e).__name__}: {e}")
            RESULT["graph"] = f"error: {type(e).__name__}: {e}"

        head("[7] 判定")
        if eager_mean and graph_mean:
            ratio = eager_mean / graph_mean
            over = (eager_mean - graph_mean) * 1e3
            say(f"eager {eager_mean * 1e3:,.0f} ms/step  →  CUDA Graph {graph_mean * 1e3:,.0f} ms/step  ({ratio:.2f} 倍)")
            RESULT["speedup"] = round(ratio, 2)
            if ratio >= 1.25:
                say(f"■ ホスト (CPU) 律速です。1 step あたり約 {over:,.0f} ms は GPU の計算ではなく、CPU が演算を発行する時間でした。")
                say("  CUDA Graph でその分が消えます。学習スクリプトは既定 (--cuda_graph auto) で sm_90 以上なら自動で有効になります。")
            else:
                say("■ GPU 律速に近い状態です (eager と graph の差が小さい)。CUDA Graph の効果は限定的です。")
            say(f"  データ供給の目安: graph 時は 1 バッチを {graph_mean * 1e3:,.0f} ms 以内に用意できないと DataLoader が律速になります")
            say("  (足りなければ環境変数 SBV2_NUM_WORKERS=2 や 3 を試す)")
        else:
            say("eager / graph のどちらかが計測できなかったため判定できません。")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(RESULT, f, ensure_ascii=False, indent=2)
        say(f"結果を {args.json} に保存しました")
    return 0


if __name__ == "__main__":
    sys.exit(main())
