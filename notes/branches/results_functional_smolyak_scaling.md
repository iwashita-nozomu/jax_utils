# Legacy Smolyak Results Branch Summary

## Branch

- `results/functional-smolyak-scaling`

## Role

- explicit grid / weight 保持を前提にした旧版 Smolyak 実装の結果を保持する branch
- 完了済み GPU run と、途中停止した partial run の両方を記録している

## Primary Notes

- 実験要約:
  - [legacy_smolyak_results_20260316.md](/workspace/notes/experiments/legacy_smolyak_results_20260316.md)

## Key Results

完了済み GPU run では `float16`, `bfloat16`, `float32`, `float64` の 4 dtype 全部を含む比較が取れていた。これにより、低精度では level を上げるほど誤差が悪化し、`float64` では level とともにきれいに改善するという、旧版実装で最初に見えた数値傾向を確認できた。一方、途中停止 partial run では当時の case 順が `dtype -> level -> dimension` だったため、`float16` が先に大量に進み、他 dtype 比較には使いにくかった。この branch は、その「数値的な傾向が読めた run」と「runner / ordering の弱さが露出した run」の両方を持っている。

実装面での知見も大きかった。旧版は host 側で grid を明示構築し、全点連結と `np.unique(axis=0)` を含む大域集約を行っていたため、CPU RSS と pinned host memory が大きく膨らんだ。GPU 利用率が低く見えた根本理由も、積分 kernel ではなく host 側の格子構築と転送準備が支配的だった点にある。旧版 branch は、数値の比較結果だけでなく、「explicit grid 戦略がどこで厳しくなるか」を示すベースラインとして価値がある。

## Interpretation

- この branch は、旧版 Smolyak の「数値傾向」と「grid 明示構築の限界」を知る基準点として重要
- 一方で、今後の runner や tuned 実装の評価基準として使うときは、case 順や runner 構造の差も一緒に読む必要がある

## Status

- archived
- Retention: `persistent`
- `main` には結果本体ではなく、要約 note と branch note を残す
