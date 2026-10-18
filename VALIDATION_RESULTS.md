# 改修版の検証結果

検証日: 2026年10月19日（日本時間）。テストはすべてModalで実行しました。

## 確認できたこと

| 項目 | 結果 |
|---|---|
| GPU | NVIDIA A100-SXM4-40GB |
| PyTorch | 2.6.0+cu124 |
| 回帰テスト | 18件すべて成功 |
| 精度 | FP32・BF16・FP16の順伝播／逆伝播で有限な勾配を確認 |
| 互換性 | 既存モデルの重みを維持して追加部を初期化。公式モデルでも初期推論を確認 |
| 実音声データ | einiite/af_alt_20file の20音声 |
| 音声の照合 | テストに使用した20音声すべてが指定HFアーカイブとバイト単位で一致 |
| 前処理 | 正規の日本語DeBERTa・WeSpeakerで特徴量を再生成 |
| 学習／検証の分割 | 18音声／2音声 |
| 実学習 | train_ms_jp_extra.py の学習関数で9更新、約28.8秒 |
| 実学習の精度 | BF16 |
| WavLM | 実音声テストでは無効 |
| 追加経路の学習 | Prior残差・音素長スタイル接続・音素長補正ゲートの更新を確認 |
| 学習後の推論 | NaN/Infのない音声出力を確認 |
| 配布用パッチ | 元コミットのクリーンなクローンへ適用でき、変更Pythonファイルの構文確認も成功 |

回帰テストでは、F0抽出、有声区間だけのF0集計、キャッシュ更新、複数候補を平均せず選択する処理、スタイルへの勾配、旧Optimizerの移行と再開、データ整列、ゼロ埋め特徴量の拒否、WavLM損失がDuration Discriminatorなしでも実学習ループで逆伝播することを検証しました。

公式事前学習モデルは `litagin/Style-Bert-VITS2-2.0-base-JP-Extra` です。G/DのSHA-256を公式配布のメタデータと照合しました。公式ベースに含まれない話者・スタイル固有パラメーターは、比較する両モデルで同じ初期値を使用しました。実音声テストでの初期出力の最大絶対差は0でした。テストの許容値は `atol=2e-6, rtol=1e-5` です。

## 評価の限界

これはコードの動作と学習経路の検証です。9更新によって演技の品質が向上したことや、未学習文の感情再現が大幅に向上することを実証した結果ではありません。採用判断には、十分な追加学習と、元モデルを対照にした未学習文の試聴比較が必要です。

## ログと再現用情報

- 回帰テスト: https://modal.com/apps/y-viewtiful/main/ap-gOP8tpwLevZVzsNiLtmUDe
- 実音声での学習・推論: https://modal.com/apps/y-viewtiful/main/ap-3y9Sjxl3qsybtrlyEb07Mp
- 最終パッチの適用確認: https://modal.com/apps/y-viewtiful/main/ap-dp6N4LkIOeCxcADmHyGWCs
- 音声ファイルの一致確認: https://modal.com/apps/y-viewtiful/main/ap-SSV5GXtgO8dwyTzUFFu8e8
- 元コミット: `84eebb27d893982ddc6e1e196c1f6dda0b1ad399`
- 専用Modal Volume: `sbv2-prosody-20261019`
- 実音声テストの要約: 同Volumeの `/smoke/summary.json`
- 動作確認用の短時間学習重み: `/smoke/smoke_adapter.safetensors`。対応設定は `/smoke/smoke_config.json`。完成した音声モデルとして評価したものではありません。

同梱の `modal_prosody_test.py` で回帰テストを再実行できます。音声や認証トークンは配布ZIPへ含めていません。
