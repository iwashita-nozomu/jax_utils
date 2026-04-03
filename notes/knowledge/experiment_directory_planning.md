# Experiment Directory Planning

このメモは、`results/smolyak-experiment-20260321` と `work/smolyak-improvement-20260318` で詰めた「実験コードをどこへ置くか」の教訓を、現在の main 向けに整理したものです。

## 先に結論

- reusable runtime は `python/experiment_runner/` に置く
- experiment 固有コードは `experiments/<topic>/` に置く
- 長時間 run の raw 結果は `results/*` branch に置く
- `main` には code、最小 final JSON、note を持ち帰る

## なぜ `python/experiment/` にしなかったか

途中では `python/experiment/` のような共通実験モジュールも検討しました。

ただしこの案は、runner 共通部と topic 固有の実験ロジックを同じ層へ混ぜやすく、実験ごとの入口も遠くなります。

最終的には、共通基盤だけを `python/experiment_runner/` に残し、Smolyak のような topic 固有コードは `experiments/` 側へ寄せる方が分かりやすいと判断しました。

## 今の標準配置

- `experiments/<topic>/README.md`
- `experiments/<topic>/cases.py`
- `experiments/<topic>/experimentcode.py`
- `experiments/<topic>/result/<run_name>/`
- `experiments/report/<run_name>.md`

この形にすると、topic ごとの入口、条件定義、実装本体、結果、1 回分 report が同じ場所から辿れます。

## 運用上の分離

短い smoke run や code/test 更新は `main` に近い worktree で扱います。

一方、週単位の長時間 run や raw JSONL 保持は `results/*` branch の worktree に逃がします。

この分離を入れると、code branch で結果生成物に引きずられにくくなります。

## Benchmark との関係

benchmark は implementation の前後比較に向きます。

experiment は condition sweep、failure analysis、partial salvage に向きます。

同じ topic でも、benchmark と long experiment は置き場と責務を分けたほうが整理しやすいです。

## Surviving Lessons

- experiment 実装は topic の近くに置いたほうが保守しやすいです
- reusable process lifecycle は runtime モジュールへ寄せるべきです
- 結果 branch と code branch を分けると衝突が大きく減ります
- directory 構造は「再現に必要な入口を 1 か所で辿れるか」を基準に決めるべきです

## References

- [results_smolyak_experiment_20260321.md](/workspace/notes/branches/results_smolyak_experiment_20260321.md)
- [work_smolyak_improvement_20260318.md](/workspace/notes/branches/work_smolyak_improvement_20260318.md)
- [benchmark_vs_experiment.md](/workspace/notes/knowledge/benchmark_vs_experiment.md)
- [30_experiment_directory_structure.md](/workspace/documents/conventions/python/30_experiment_directory_structure.md)
