# Smolyak Research Loop 011 Full Frontier (Partial Salvage)

Date: 2026-04-01

## Goal

`shifted_laplace_product` を `d=1..50`, `level=1..4` まで 1 次元ずつ押し上げ、`auto / points / indexed / batched` の全 mode を GPU 上で failure-inclusive に観測する。狙いは、非平滑 family に対して current integrator がどこまで到達できるかと、そのとき Monte Carlo に対してどの程度競争力を失うかを定量化することだった。

## Executed Calculations

- Family: `shifted_laplace_product`
- Domain: `[-0.5, 0.5]^d`
- Integrand:
  `f(x) = exp(-sum_j beta_j |x_j - shift_j|)`
- Parameters:
  `beta_j` は `1.0 -> 6.0` の線形列、`shift_j` は `-0.25 -> 0.25` の線形列
- DType: `float64`
- Dimensions: `1,2,...,50`
- Levels: `1,2,3,4`
- Requested modes: `auto,points,indexed,batched`
- Chunk size: `16384`
- Batch size: `32`
- Timeout per case: `120 s`
- Matrix command:
  `python3 -m experiments.smolyak_experiment.run_smolyak_mode_matrix --platform gpu --dimensions 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50 --levels 1,2,3,4 --dtypes float64 --families shifted_laplace_product --requested-modes auto,points,indexed,batched --chunk-sizes 16384 --batch-size 32 --warm-repeats 1 --timeout-seconds 120 --workers-per-gpu 1 --quiet --output-dir /workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_011_shifted_laplace_product_frontier`

## Primary Outputs

- Partial matrix run dir:
  `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_011_shifted_laplace_product_frontier/report_20260401T094812Z`
- Partial JSONL:
  `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_011_shifted_laplace_product_frontier/report_20260401T094812Z/results.jsonl`
- Partial Markdown report:
  `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_011_shifted_laplace_product_frontier/report_20260401T094812Z/report.md`
- Frontier CSV:
  `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_011_shifted_laplace_product_frontier/report_20260401T094812Z/mode_matrix_frontier.csv`
- Raw CSV:
  `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_011_shifted_laplace_product_frontier/report_20260401T094812Z/mode_matrix_raw_cases.csv`
- Full-frontier compare JSON:
  `/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/results/research_loops/loop_011_shifted_laplace_product_frontier/compare_full/compare_smolyak_vs_mc_1775038348.json`

## Result Summary

- Matrix status: `partial salvage`
- Parent return code: `-15`
- Cases recorded before interruption: `761 / 800`
- Cases succeeded: `694`
- Cases failed: `67`
- Realized mode counts: `{'points': 372, 'indexed': 192, 'batched': 130}`
- Failure counts: `{'oom': 51, 'error': 16}`

### Frontier snapshot

- `auto`:
  `level=1,2,3` は `d=50` まで全成功、`level=4` も `d=50` 成功。最初の失敗は `d=38` で、以後も成功が混ざる。
- `points`:
  `level=4` では最初の OOM が `d=17`。それでも後続の成功が混ざるので、単調な frontier ではない。
- `indexed`:
  `level=4` では最初の failure が `d=21`、最初の OOM が `d=25`。一方で成功は `d=50` まで続く。
- `batched`:
  `level=4` では `d=1,5,10` で OOM が出ており、成功は確認できた範囲で `d=11` まで。execution path として最も不安定だった。

### High-dimension auto cells

`auto, level=4` の高次元帯域では、実現 mode は `indexed` に切り替わった。

- `d=41`: `indexed`, `num_points=320498`, `abs_error=5.74e-07`, `avg_gpu_util=53.17%`
- `d=45`: `indexed`, `num_points=422506`, `abs_error=1.42e-07`, `avg_gpu_util=51.0%`
- `d=50`: `indexed`, `num_points=577826`, `num_terms=23426`, `storage=66.5 MB`, `abs_error=2.34e-08`, `avg_gpu_util=58.5%`

この部分だけ見ると execution frontier 自体はかなり伸びており、storage bug 修正後の実装改善は確かに効いている。

## Monte Carlo Compare

最も高い成功セルとして `d=50, level=4, auto(indexed)` を追加比較した。

- Analytic value: `4.0252e-17`
- Smolyak value: `-2.3427e-08`
- Smolyak absolute error: `2.3427e-08`
- Smolyak warm runtime: `2.61 ms`
- Monte Carlo same-budget samples: `577826`
- Monte Carlo same-budget absolute error: `2.99e-18`
- Monte Carlo same-budget warm runtime: `3.57 ms`
- Monte Carlo matched-error absolute error: `4.02e-17`
- Monte Carlo matched-error chosen samples: `1`
- Monte Carlo matched-error warm runtime: `0.253 ms`

結論として、この non-smooth family では `d=50, level=4` に「実行する」ことはできたが、「正確に積分する」ことには失敗している。Smolyak の出力が正の integrand に対して負になっており、difference-rule の打ち消しが精度を壊していることが分かる。same-budget では MC が 9 桁以上良い。

## Critical Review

- この family は文献レビューで想定した通り、plain Smolyak に不利な non-smooth cusp family だった。到達性の評価には有用だが、精度保証の代表例にはならない。
- `auto -> indexed` の切替そのものは正しかった。少なくとも `points` より大きい帯域まで到達できている。
- ただし mode 選択は execution frontier を広げるだけで、accuracy を保証しない。`d=50, level=4` の catastrophic sign error はその典型例である。
- `batched` 強制 mode の OOM が `level=4` で `d=1` から散発するのは明らかに不自然で、prefix/suffix batching の shape 設計か JAX compile path が不安定だと見るべきである。
- この run は parent が `SIGTERM` で終了したが、partial JSONL を salvage して report 化できた。failure-inclusive 運用としては、今後はこちらを標準化すべきである。

## Measurement Improvements

- `num_terms` と `num_evaluation_points` に加えて、理論上の組合せ数 `C(d+l-1, d)` を毎セル残す。`50D level15` の議論では execution limit と数学的 growth limit を分ける必要がある。
- 正値 integrand に対して負の積分値が出たときは `sign_violation` を別途 flag 化する。non-smooth family の catastrophic cancellation を早く見つけられる。
- `batched` mode の OOM は compile-time と runtime を切り分けて記録する。現状は `oom` に潰れていて原因の層が見えない。

## Next Step

1. 同じ full-frontier partial salvage を `shifted_anisotropic_gaussian` と `balanced_exponential` にも広げ、smooth / non-smooth / cancellation の 3 種で実装限界を比較する。
2. `smolyak.py` に theoretical term-count logging を入れ、`50D level15` が storage bug の問題なのか、term explosion の問題なのかを可視化する。
3. `batched` mode の shape 設計を見直し、少なくとも `level=4` での散発 OOM を解消する。
