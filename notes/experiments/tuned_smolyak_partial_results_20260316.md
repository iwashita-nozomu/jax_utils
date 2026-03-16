# Tuned Smolyak Partial Results

## Scope

このメモは、`results/functional-smolyak-scaling-tuned` で進行中だった GPU 実験をいったん停止し、途中までの JSONL から読めたことを整理するための report である。対象の partial 集計は [smolyak_scaling_gpu_20260316T132125Z_partial.json](/workspace/.worktrees/results-functional-smolyak-scaling-tuned/experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260316T132125Z_partial.json) と [partial report](/workspace/.worktrees/results-functional-smolyak-scaling-tuned/experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260316T132125Z_partial_report/index.html) である。生データは [smolyak_scaling_gpu_20260316T132125Z.jsonl](/workspace/.worktrees/results-functional-smolyak-scaling-tuned/experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260316T132125Z.jsonl) にある。

## Observations

partial の完了件数は `547` で、内訳は `ok=99`, `failed=448` だった。dtype の進み方はほぼ均等で、`float16=137`, `bfloat16=137`, `float32=137`, `float64=136` である。これは case 順を `level -> dimension -> dtype` にしていた時点の partial であり、少なくとも「一つの dtype だけが先に進み過ぎる」という旧 runner の問題はかなり緩和されていた。

観測できた level は事実上 `0` と `1` に限られる。`level=0` は仕様上の失敗であり、ここから読み取るべきなのは実装の品質ではなく failure accounting が正しく記録されているかである。意味があるのは `level=1` 側で、ここでは全 dtype で `num_points=1` のケースしか通っていない。それでも次元を上げると `integrator_init_seconds` と `process_rss_mb` が大きく伸びる。

`level=1` の成功末尾を見ると、`float16` は `d=25` で `integrator_init_seconds=16.32s`, `process_rss_mb=3659.8`、`bfloat16` は `d=25` で `14.49s`, `3612.6 MB`、`float32` は `d=27` で `54.91s`, `12646.4 MB`、`float64` は `d=24` で `6.99s`, `2080.5 MB` だった。一方、`avg_integral_seconds` はどの dtype でもほぼ `10^-4` から `10^-3` 秒台で、初期化時間に比べて非常に小さい。したがって、

$$
\text{integration kernel} \ll \text{initializer / lowering / memory}
$$

という不均衡が、少なくともこの partial の範囲ではかなり明瞭である。

失敗の種類も、純粋な数値精度限界より実行基盤の制約を示している。全体の failure kind は `error=412`, `oom=8`, `worker_terminated=28` で、`level=1` の frontier 付近では `oom` と `worker_terminated` が混ざっている。次元ごとに見ると、`d=20..23` は全 dtype 成功、`d=24..25` から dtype ごとに OOM が混じり始め、`d=26` では全 dtype OOM、`d>=28` では `worker_terminated` が増える。これは「低精度ほど一様に早く崩れる」というより、runner・初期化・メモリの相互作用で実行が不安定になっている形に近い。

## Interpretation

この partial が示しているのは、Smolyak の数学的な点数爆発より前に、現在の tuned 実装の初期化経路がすでに厳しいということである。`level=1` で `num_points=1` なのに初期化時間と RSS が強く次元依存する以上、主因は quadrature 点そのものではなく、積分器初期化、lowering、compile、あるいはそれに付随する metadata 構築にあると考えるのが自然である。

もう一つ重要なのは、dtype 差より構造差の影響が大きいことである。`float16` と `bfloat16` は `d=25` まで成功し、`float64` は `d=24` まで、`float32` は `d=27` まで一応成功しているが、これはそのまま「float32 が最良」という話にはならない。実際には `float32` で `d=24` に OOM が出ているので、frontier が単調ではない。したがってこの partial は、数値誤差比較の根拠というより「実装がどこで不安定になるか」を見る資料として読むべきである。

## Practical Consequences

現時点では、`level>=2` での精度改善や dtype ごとの高 level 比較を議論するのは早い。まず必要なのは、`level=1` でも次元 25 前後から重くなる理由を切り分けることである。これには HLO 解析、初期化時間の分解、JIT まわりの tracing / lowering / execution の分離計測が有効である。また、case 順は frontier を早く読むという観点で `dimension -> level -> dtype` に見直したので、今後の partial はこのメモより実用的な形になるはずである。

## Related Notes

- branch summary:
  - [results_functional_smolyak_scaling_tuned.md](/workspace/notes/branches/results_functional_smolyak_scaling_tuned.md)
- tuning worktree extraction:
  - [worktree_smolyak_tuning_2026-03-16.md](/workspace/notes/worktrees/worktree_smolyak_tuning_2026-03-16.md)
- runner modularization extraction:
  - [worktree_experiment_runner_module_2026-03-16.md](/workspace/notes/worktrees/worktree_experiment_runner_module_2026-03-16.md)
