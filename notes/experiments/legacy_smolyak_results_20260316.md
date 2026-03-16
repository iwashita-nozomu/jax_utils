# Legacy Smolyak Results 2026-03-16

このメモは、旧版 Smolyak 実装の大規模実験結果を `main` から辿れるように残すためのものです。

## 関連 branch

- 結果 branch:
  - `results/functional-smolyak-scaling`
- 改訂版の継続実験 branch:
  - `results/functional-smolyak-scaling-tuned`

## 主要結果

対象として残している主な結果は次の 2 つです。

- 完了済み GPU run
  - `smolyak_scaling_gpu_20260315T140215Z.json`
- 長時間 run の partial 集計
  - `smolyak_scaling_gpu_20260316T061620Z_partial.json`

## 定性的な考察

1. 旧版でも `float64` は安定で、level を上げても平均誤差は小さく保たれました。
2. `float16` と `bfloat16` は高 level・高次元で誤差悪化が顕著で、差分則の相殺誤差に強く影響されました。
3. `float32` は中間的ですが、大規模化するとやはり誤差悪化が見えました。
4. 大規模 run の主要ボトルネックは GPU 演算本体より host 側の格子構築とメモリ保持でした。
5. partial run に `float16` しか残っていないのは、旧 runner が `dtype -> level -> dimension` 順にケースを並べていたためです。途中停止時の比較性は低く、この点は改訂版で改善対象になりました。

## メモ

旧版結果の可視化は `results/functional-smolyak-scaling` branch 上の report ディレクトリにあります。`main` には結果ファイル本体は置かず、ここでは branch と考察だけを残します。
