# CUDA Graph 学習 (H200 / B200 / B300 で step が遅い問題への対処)

## 症状と原因

- 症状: T4 / L4 / L40S / RTX PRO 6000 は性能どおりなのに、H200 / B200 / B300 だけ想定より遅い
  (例: H200 `1.25 s/it`、RTX 6000 Pro `0.74 s/it`。GPU の性能と逆転している)。
- 原因 (確度: 高。ただし開発環境に GPU が無く、実機では未検証): **GPU ではなく CPU が律速**。
  - 学習 1 step は約 **12 万個の aten 演算 (CUDA カーネル起動は約 3 万回)** を、Python の 1 スレッドが
    1 個ずつ GPU に発行する「ホスト (CPU) 律速」の負荷になっている (eager 実行)。
  - GPU が十分に速いと、GPU は次の演算の到着待ちになり、step 時間は「GPU の速さ」ではなく
    「**ホスト CPU の 1 スレッド性能 × 演算数**」で決まる。この下限は 1 step ≒ 1 秒前後
    (GPU 無し・極小モデルでも同程度の時間がかかることを確認済み)。
  - H200 / B200 / B300 のホストが RTX 6000 Pro のホストより 1 スレッド性能の低い CPU だと、
    GPU は速いのに step は遅い、という逆転が起きる。
  - 「CPU 使用率が低い・コア数は足りている」は反証にならない。律速しているのは **1 スレッドが
    100% 張り付いている状態**で、全体の CPU 使用率には出ない。
- 副次要因 (これも修正済み): 1 step に約 175 回の GPU→CPU 同期 (`.item()` / ブール添字 / `torch.zeros(...).to(device)`
  など)。同期のたびに GPU のキューが空になり、CPU の発行遅れがそのまま見える。
  エポックごとの `torch.cuda.empty_cache()` (1 エポック 7〜8 step では数 step ごとに cudaFree/cudaMalloc) も除去。

## 変更点

| ファイル | 内容 |
|---|---|
| `cuda_graph_step.py` (新規) | 学習 1 step 全体 (G 順伝播 → D 更新 → G 更新 + 勾配ノルム) を CUDA Graph にして `replay()` 1 回で実行。形状ごとに 1 グラフ。eager との自己検証つき、失敗時は自動で eager |
| `train_ms_jp_extra.py` | `--cuda_graph {auto,on,off}`、graph 経路のループ、`[perf]` ログ、エポック末の `gc/empty_cache` を 100 step に 1 回へ間引き |
| `data_utils.py` | `StaticShapeTable`: バッチ形状をバケットごとに固定 (スペクトログラム長=バケット上限、テキスト長=バケット内最大) |
| `style_bert_vits2/models/transforms.py` | スプラインのブール添字・定義域チェックを `clamp` + `torch.where` に置換 (同期ゼロ。元実装と数値一致) |
| `style_bert_vits2/models/modules.py`, `commons.py` | `Flip` を device 上で生成、TorchScript の活性化を通常の関数へ、勾配ノルムを同期なしで計算 |
| `style_bert_vits2/models/models.py`, `models_jp_extra.py` | SDP のノイズを device 上で生成、静的形状でも MAS ノイズの std を従来と同じ範囲で計算 |
| `style_bert_vits2/models/utils/checkpoints.py` | 保存する optimizer state を従来形式へ戻す (graph 学習の checkpoint を従来コードで再開できる) |
| `mel_processing.py` | CUDA テンソルの範囲チェック (`.item()`) を既定で無効化 (`SBV2_MEL_RANGE_CHECK=1` で復活) |
| `tools/profile_step.py` (新規) | GPU 1 枚・データ不要で、原因の確認と効果の測定を 1 回で行う診断スクリプト |
| `tests/` (新規) | CPU / meta デバイスで動くテスト一式 (`python -m pytest tests -q`) |

推論 (TTS / ONNX) で使う共通コードの出力は、変更前と**ビット単位で一致**することを確認済み (ノイズ 0 の設定)。

## 使い方

学習コマンドは変更不要。既定の `auto` は **Hopper 以降 (sm_90 以上: H100 / H200 / B200 / B300 / RTX PRO 6000 など)** で自動的に有効になる。

```
python train_ms_jp_extra.py --cuda_graph auto   # 既定。sm_90 以上のみ有効
python train_ms_jp_extra.py --cuda_graph on     # 強制 (sm_89 以下でも)
python train_ms_jp_extra.py --cuda_graph off    # 従来どおり
```
環境変数でも指定できる: `SBV2_CUDA_GRAPH=auto|on|off`。

対象外 (従来の学習ループがそのまま動く): `fp16_run` (GradScaler)、複数 GPU (DDP)、duration / WavLM discriminator 使用時、
`--not_use_custom_batch_sampler`。bf16 / fp32 の単一 GPU が対象。

| 環境変数 | 既定 | 意味 |
|---|---|---|
| `SBV2_CUDA_GRAPH_VALIDATE` | `all` | 自己検証 (`all`=形状ごとの初回 capture / `first`=最初の 1 形状だけ / `off`) |
| `SBV2_CUDA_GRAPH_RTOL` | `0.1` | 自己検証の許容相対誤差 |
| `SBV2_CUDA_GRAPH_MAX_SHAPES` | `24` | 保持するグラフ形状の上限 |
| `SBV2_CUDA_GRAPH_MAX_AHEAD` | `3` | CPU が GPU より先行してよい step 数 (ピン留めメモリの増加防止) |
| `SBV2_NUM_WORKERS` | `1` | DataLoader のワーカー数 (graph で step が速くなりデータ供給が間に合わないとき 2, 3 に) |

## ログの読み方

```
CUDA Graph training: ON (NVIDIA H200 (sm_90), torch 2.x)          ← 有効。OFF なら理由が括弧内に出る
CUDA Graph: static batch shapes (spec_len, text_len) = [(300, 96), (400, 120), ...]
[cuda-graph] 自己検証 OK (eager と一致): loss_disc_all=..., Δparam_G=0.0xx ...   ← 形状ごとの最初の capture 時
[cuda-graph] shape (...): capture 3.1s, 以降は replay (2/8 shapes ready)
[perf] 直近 200 step: 0.35 s/step (2.86 it/s), DataLoader 待ち 4%    ← ログ間隔ごとの実効速度
CUDA Graph summary: shapes=8 (ready=8, failed=0), replay_steps=..., eager_steps=..., disabled=False
```

- 形状ごとに「1 回目は eager (ウォームアップ)、2 回目で capture、3 回目以降は replay」。7〜8 step / エポックでも 2 エポックで全形状が replay になる。
- `[cuda-graph] ... CUDA Graph 化に失敗しました` が出たら、その形状 (最初の失敗なら全体) は eager で続行している。学習は止まらない。**その警告文を共有してもらえれば原因を特定できる**。
- tqdm の `s/it` は graph 経路では実際の速度とずれて見える (CPU が先行して投入するため)。`[perf]` 行の値を見ること。
- `DataLoader 待ち` が大きい (30% 以上) なら GPU ではなくデータ供給が律速。`SBV2_NUM_WORKERS=2` 以上を試す。

## 実機での確認 (最初の 3 分)

```
python tools/profile_step.py --config <学習に使う config.json> --batch-size <学習時の batch_size> --sync-check --profile
```
出力の見どころ:
- `[2] 演算 1 個を発行する CPU 側コスト` … 大きい (10 µs/個超) ホストほど eager が遅い。
- `[5] プロファイル` … 「eager の 1 step のうち GPU が計算しているのは約 N%」。低い (30〜50% など) ならホスト律速で確定。
- `[7] 判定` … eager と CUDA Graph の step 時間の比。差が小さければ GPU 律速で、今回の対処の効果は限定的。

(`--device cpu --tiny` で GPU 無しの動作確認もできる。その場合の数値に意味は無い。)

## 設計メモ

- **静的形状**: DistributedBucketSampler のバケット境界までゼロ埋め (padding)。ゼロ埋め部分は `x_lengths` / `spec_lengths` のマスクで無効化されるので学習の意味は変わらない
  (MAS ノイズの `std` だけは「バッチ内の有効範囲」で計算していたため、`_std_over_batch_max` で同じ範囲に揃えた)。
  代償として、バケット内の短いサンプルは上限まで計算するぶん GPU の仕事が少し増える (エンコーダ側のみ。デコーダはセグメント固定)。
- **optimizer**: AdamW を `capturable=True`、学習率を GPU 上の 0 次元テンソルにして graph 内で更新。`scheduler.step()` 後に `after_lr_schedule()` で束縛し直す。
  checkpoint 保存時は従来形式 (float の lr / `capturable=False` / CPU の step) に戻す。
- **自己検証**: 形状ごとの最初の capture 時に「eager で 1 step → 状態を巻き戻す → capture → replay」を行い、損失・勾配ノルム・パラメータ更新量が一致することを確認する (1 形状あたり数秒)。
  一致しなければ警告して、その形状 (最初の形状なら全体) を eager へ。
- **既知の制約**: `expandable_segments:True` と graph capture は相性が悪い場合がある。`requirements.txt` の `torch<2.4` は Blackwell (torch 2.7 以上が必要) と矛盾している。
