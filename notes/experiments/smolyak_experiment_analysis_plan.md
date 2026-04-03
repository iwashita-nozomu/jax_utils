# Smolyak Experiment Analysis Plan

日付: 2026-03-31  
関連 worktree: `work/smolyak-integrator-opt-20260328`

## Goal

Smolyak 積分器の評価を、`精度` と `実装コスト` を混ぜずに読める形へ整理する。

この計画では CPU 実行を前提にせず、既存 runner はそのまま使う。主対象は
`experiments/smolyak_experiment/` 配下の既存スクリプト群と、その既存 report
資産である。

## Current Coverage

現状の実験コードは、比較軸としてはかなり揃っている。

| Question | Script | Main output |
| --- | --- | --- |
| 実装全体の成功率・初期化時間・failure kind はどうなっているか | `experiments/smolyak_experiment/run_smolyak_experiment_simple.py` | `results/<size>/results_*.jsonl`, `final_results_*.json` |
| 同点数なら Smolyak と Monte Carlo のどちらが高精度か | `experiments/smolyak_experiment/report_smolyak_same_budget_accuracy.py` | `summary.json`, `report.md`, SVG |
| Smolyak と同精度を Monte Carlo が得るには何サンプル必要か | `experiments/smolyak_experiment/report_smolyak_vs_mc.py` | `summary.json`, `report.md`, SVG |
| Gaussian で same-budget の観測は理論的な MC 誤差とどう整合するか | `experiments/smolyak_experiment/report_smolyak_theory_comparison.py` | `literature_theory.md`, SVG |
| GPU で single / batched throughput はどう伸びるか | `experiments/smolyak_experiment/report_smolyak_gpu_sweep.py` | `summary.json`, `report.md`, SVG |
| GPU の batch size はどこで頭打ちになるか | `experiments/smolyak_experiment/report_smolyak_gpu_batch_scaling.py` | `summary.json`, `report.md`, SVG |
| 長時間 full sweep で frontier と failure map はどう見えるか | `experiments/smolyak_experiment/run_smolyak_large_full_report.py` | raw JSONL, final JSON, compact report |

## Existing Evidence

すでに読める report から、分析の優先順も見えている。

- same-budget Gaussian sweep の代表: `../../experiments/smolyak_experiment/results/smolyak_same_budget_accuracy/report_20260330T060742Z/report.md`
- matched-accuracy 比較の代表: `../../experiments/smolyak_experiment/results/smolyak_vs_mc_reports/report_20260328T125507Z/report.md`
- GPU dimension sweep の代表: `../../experiments/smolyak_experiment/results/smolyak_gpu_sweep/report_20260330T054425Z/report.md`
- GPU batch scaling の代表: `../../experiments/smolyak_experiment/results/smolyak_gpu_batch_scaling/report_20260330T054723Z/report.md`

ここから読める現時点の仮説は次のとおり。

- Gaussian same-budget では、低 level から中 level の広い領域で Monte Carlo が精度優位になる。
- 一方で matched-accuracy では、少数の低次元ケースで Smolyak が warm runtime 優位になる余地がある。
- GPU では batching による throughput 改善は見えるが、軽いケースでは utilization 自体はまだ低い。
- 大規模 frontier の読みでは、数値積分そのものより初期化や OOM / timeout の実装側制約が先に効いている。

## Comparison Targets

まず主 comparison を 3 本柱に固定する。

1. Same-budget accuracy  
   点数を揃えて純粋に数値手法としてどちらが良いかを見る。主指標は `absolute_error`,
   `relative_error`, `MC/Smolyak error ratio`。

2. Matched-accuracy runtime  
   Smolyak の誤差に Monte Carlo が追いつくまで sample を増やし、runtime と sample
   budget を比べる。主指標は `chosen_num_samples`, `warm_runtime_ms`,
   `MC/Smolyak runtime ratio`。

3. GPU throughput / utilization  
   `single` と `vmap(integrate(f))` を分けて、GPU が律速か CPU 側初期化が律速かを切り分ける。
   主指標は `throughput_integrals_per_second`, `gpu_util`, `pstate`,
   `peak_mem_used_mb`。

## Analysis Order

1. 既存 report の再読と gap 埋め  
   新しい長時間 run より先に、same-budget / matched-accuracy / GPU reports で
   読める事実を一枚にそろえる。

2. Same-budget を主系列にする  
   Gaussian `dimension=1..20`, `level=2..4`, `float32/float64` を baseline にして、
   どの領域で Smolyak が精度優位か、どこから Monte Carlo 優位へ反転するかを読む。

3. 反転点だけ matched-accuracy で掘る  
   same-budget で勝敗が入れ替わる近傍だけを対象に、Monte Carlo がどれだけ sample を
   積むと Smolyak に追いつくかを調べる。

4. GPU は代表ケースで切る  
   全点を GPU sweep するのではなく、`same-budget` で特徴的だった次元と level を代表点に
   選び、single / batch / batch-size scaling を測る。

5. 大規模 full sweep は最後に回す  
   実装改善後の frontier と failure kind を確認する用途に限る。探索目的の最初の一手にはしない。

## Metrics To Track

- 数値精度: `absolute_error`, `relative_error`
- 予算: `num_evaluation_points`, `chosen_num_samples`
- 実行時間: `warm_runtime_ms`, `first_call_ms`, `compile_ms`
- 実装 overhead: `integrator_init_seconds`, `integrate_seconds`, `elapsed_seconds`
- 実行可能性: success rate, OOM, timeout, worker termination
- GPU 利用: throughput, `gpu_util`, `pstate`, `peak_mem_used_mb`

## Immediate Code Directions

- `report_smolyak_vs_mc.py` は matched-accuracy に加えて same-budget の summary も前面に出す。
- 新しい実験を足す前に、既存 report の schema を崩さず summary を読みやすくする。
- 実験コードの改善は runner 改造ではなく、比較の読みやすさと GPU 実行の解像度向上に寄せる。

## Non-Goals For This Round

- runner の責務変更
- CPU 向けの実行フロー整備
- いきなり full sweep を回してから考える運用
