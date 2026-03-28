# Smolyak Benchmark Levels Analysis

日付: 2026-03-19\
関連 worktree: `work/smolyak-improvement-20260318`

## Source

- worktree 上で作られた一時 benchmark suite による Smolyak 初期化時間と積分時間の観測。
- benchmark 実装自体はそのまま main に持ち込まず、方針だけを [20_benchmark_policy.md](/workspace/documents/conventions/python/20_benchmark_policy.md) へ吸収した。

## Three Levels

- `Light`: 秒から数十秒で終わる、日常の前後比較向け。
- `Heavy`: 数分かけて傾向を見る、改善前後の比較向け。
- `Extreme`: 長時間で限界を探る、設計判断向け。

## Observations

- 低次元では初期化時間はほぼ線形に見えるが、高次元側では term plan の増加が効き始める。
- level を上げると評価点数は急に増えるが、初期化時間の増え方は別の形を取ることがある。
- `float32` と `float64` の差は、ケースサイズや JAX 側の振る舞いで見え方が変わる。

## Interpretation

- 局所的な実装変更を確かめるなら、まず Light か Heavy で十分なことが多い。
- 高次元や failure frontier を見たい場合は、benchmark ではなく experiment に切り替えるべき。

## Related

- [Benchmark vs Experiment](/workspace/notes/knowledge/benchmark_vs_experiment.md)
- [Smolyak スケーリング実験](/workspace/notes/experiments/smolyak_scaling_experiment.md)
