# Legacy Smolyak Results Notes

このメモは `results/functional-smolyak-scaling` ブランチ上の旧版 Smolyak 積分実験について、結果の読み方を残すためのものです。

## 対象結果

- 完了済み GPU run:
  - `results/smolyak_scaling_gpu_20260315T140215Z.json`
  - `results/smolyak_scaling_gpu_20260315T140215Z_report/`
- 途中停止した長時間 GPU run:
  - `results/smolyak_scaling_gpu_20260316T061620Z.jsonl`
  - `results/run_smolyak_scaling_20260316T061619Z.log`
  - `results/smolyak_scaling_gpu_20260316T061620Z_partial.json`
  - `results/smolyak_scaling_gpu_20260316T061620Z_partial_report/`

## 定性的な考察

1. `float64` は旧版でもっとも安定しており、完了済み run では level を上げても平均誤差は十分小さいままでした。
2. `float16` と `bfloat16` は高 level・高次元で誤差が急速に悪化しました。差分則の正負重みを含む和の相殺誤差が、低精度では前面に出ています。
3. `float32` は低精度 2 種よりは安定ですが、大規模化するとやはり誤差悪化が見えました。
4. 長時間 run の partial 結果では `worker_terminated` と `host_oom` が目立ち、GPU 計算本体より host 側の格子構築と保持が支配的でした。
5. partial run が `float16` しか含まないのは renderer の問題ではなく、旧 runner のケース順が `dtype -> level -> dimension` だったためです。
6. partial run から読む限り、旧版の `float16` frontier はおおむね
   - level 1..5: dimension 32
   - level 6: dimension 28
   - level 7: dimension 18
   - level 8: dimension 13
   付近でした。

## 位置づけ

この branch の結果は「旧版 Smolyak 実装の挙動把握」としては有用です。一方で、メモリ配置・ケース順・reporting を含めた運用上の問題も大きく見えたため、その後の改訂版は別 branch `results/functional-smolyak-scaling-tuned` で継続しています。
