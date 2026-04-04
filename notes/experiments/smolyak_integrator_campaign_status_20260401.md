# Smolyak Integrator Campaign Status

Date: 2026-04-01
Worktree: `work/smolyak-integrator-opt-20260328`

## Scope

このレポートは、今回の implementation-focused campaign の現時点の到達点をまとめる。対象は Monte Carlo との比較を残したまま、`SmolyakIntegrator` を高次元・高 level へ押し広げることだった。文献上の比較軸は [smolyak_literature_review_20260401.md](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/notes/experiments/smolyak_literature_review_20260401.md) に整理してあり、本レポートではその観点に沿って「何を直したか」「何が分かったか」「まだ何ができていないか」を実測ベースで書く。

## What Changed

### 1. 積分器本体

- [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py) の重大な rule-storage bug を修正した。以前は multi-index の `1`-norm 上限 `d + l - 1` を、そのまま最大 1D difference-rule level として前計算していたため、`level=1` でも storage が指数的に膨らんでいた。これは [smolyak_integrator_loop01_rule_storage_fix_20260401.md](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/notes/experiments/smolyak_integrator_loop01_rule_storage_fix_20260401.md) に詳細を残している。
- 同ファイルでは unused helper を削除し、term plan 初期化を 1 本化した。これで materialization 経路の読み筋がかなり単純になった。

### 2. 実験ハーネス

- [families.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/families.py) を新設し、`gaussian`、`anisotropic_gaussian`、`shifted_anisotropic_gaussian`、`quadratic`、`absolute_sum`、`exponential`、`balanced_exponential`、`shifted_laplace_product` を shared module 化した。
- [compare_smolyak_vs_mc.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/compare_smolyak_vs_mc.py) と [run_smolyak_mode_matrix.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/run_smolyak_mode_matrix.py) は、この shared family 定義を使うように整理した。これにより「matrix では回るが compare は family を知らない」という不整合を消した。
- [run_smolyak_research_loops.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/run_smolyak_research_loops.py) には partial-salvage を入れた。matrix parent が落ちても、残った JSONL から CSV / SVG / Markdown を再生成し、loop 単位の考察を残せる。

### 3. 回帰確認

- [test_smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/tests/functional/test_smolyak.py)
- [test_smolyak_experiment_families.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/tests/functional/test_smolyak_experiment_families.py)

いまの relevant test は `25 passed` で、`git diff --check` も通っている。

## Quantitative Findings

### Smooth baseline: Gaussian, loop 001

小さい frontier として [smolyak_research_loop_001.md](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/notes/experiments/loops/smolyak_research_loop_001.md) を完成させた。`d=1..15`, `level=1..4`, `auto/points/indexed/batched`, `float64`, `gaussian` の 240 ケースで failure は 0 だった。

- Cases succeeded: `240 / 240`
- Auto frontier at highest recorded level: `d=15, level=4`
- Realized mode counts: `points=120`, `indexed=60`, `batched=60`

代表図:

![Gaussian frontier](../../experiments/smolyak_experiment/results/research_loops/loop_001_gaussian_frontier/report_20260401T093120Z/frontier_gaussian_float64_c16384.svg)

読み方: 横軸は level、縦軸は成功した最大次元。smooth かつ低 level では全 mode がまだ十分広く通っている。

この loop の critical review は、「failure が出ないので cap が保守的すぎる」「低 level では MC 優位のセルがまだ多い」というものだった。つまり、execution は安定したが、この帯域だけでは sparse-grid の優位性を主張するには弱い。

### Non-smooth stress: shifted Laplace, loop 011

非平滑 family については [smolyak_research_loop_011.md](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/notes/experiments/loops/smolyak_research_loop_011.md) の小さい loop と、より広い partial salvage 版 [smolyak_research_loop_011_full_partial_20260401.md](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/notes/experiments/loops/smolyak_research_loop_011_full_partial_20260401.md) の両方を残した。

full partial frontier の主要数値:

- Cases recorded: `761 / 800`
- Cases succeeded: `694`
- Cases failed: `67`
- Failure counts: `oom=51`, `error=16`
- `auto, level=4`: `d=50` まで成功
- `batched, level=4`: 成功は `d=11` まで、しかも低次元で OOM が散発

代表図:

![Shifted Laplace frontier](../../experiments/smolyak_experiment/results/research_loops/loop_011_shifted_laplace_product_frontier/report_20260401T094812Z/frontier_shifted_laplace_product_float64_c16384.svg)

読み方: 横軸は level、縦軸は成功した最大次元。`auto/indexed/points` は 50 次元まで食い込む一方、`batched` だけが早く崩れている。

![Shifted Laplace fastest mode](../../experiments/smolyak_experiment/results/research_loops/loop_011_shifted_laplace_product_frontier/report_20260401T094812Z/fastest_success_mode_shifted_laplace_product_float64_c16384.svg)

読み方: 各セルはその `d, l` で batched warm runtime が最も良かった requested mode。`auto` が勝つ帯域もあるが、`indexed` や `points` がより良い場所もはっきり残っている。

さらに hardest successful cell として `d=50, level=4` を Monte Carlo と比較すると、execution と accuracy がきれいに分離した。

- Smolyak: `indexed`, `577826` points, `66.5 MB`, warm `2.61 ms`, absolute error `2.34e-08`
- Monte Carlo same-budget: warm `3.57 ms`, absolute error `2.99e-18`

このセルでは Smolyak の値が負になっており、正の integrand に対して catastrophic cancellation を起こしている。つまり、「50 次元で実行できた」こと自体は進歩だが、「50 次元で信用できる積分ができた」ことを意味しない。

## Literature Comparison

文献レビューでは、Bungartz-Griebel 系の classical sparse grid は mixed regularity のある smooth problem で強く、Jakeman-Roberts 系の adaptive literature は non-smooth / anisotropic / localized difficulty に対して plain Smolyak が弱いことを示していた。[smolyak_literature_review_20260401.md](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/notes/experiments/smolyak_literature_review_20260401.md)

今回の実測はその整理と整合的である。

- Gaussian では、少なくとも low-level frontier では execution が安定する。
- shifted Laplace では、execution frontier は押し上がっても accuracy が壊れる。
- したがって、現在の実装改善は「plain isotropic Smolyak を GPU 上でより大きな帯域へ運べるようにした」であって、「adaptive sparse grid literature に並んだ」ではない。

この差は重要である。文献との違いを正直に書くなら、現在のコードはまだ

- dimension adaptive でない
- local adaptive でない
- singularity / cusp を観測して refinement しない
- mode 選択が accuracy-aware でない

という点で、先行研究の強い版には届いていない。

## What Currently Limits 50D Level 15

ここが最も重要な批判的結論である。`50D level15` に向けた障害は、もはや単なる storage bug ではない。

1. storage bug は loop 01 で潰せた。
2. mode materialization も `indexed` 追加で大きく改善した。
3. それでも high-level では multi-index の組合せ数そのものが爆発する。

つまり次のボトルネックは「実装の無駄」だけではなく、「plain isotropic Smolyak の term growth」をどう扱うかである。ここを超えるには、

- 理論 term count の明示的ログ
- term table 全 materialization を避ける streaming 的な扱い
- もしくは adaptive / weighted Smolyak への踏み込み

のいずれかが必要になる可能性が高い。

## Next Work

次に優先するべき実装課題は 4 つある。

1. `smolyak.py` に theoretical term-count logging を追加し、failure が implementation limit なのか combinatorial limit なのかを各セルで判定できるようにする。
2. `batched` mode の OOM 不安定性を解消する。特に non-smooth family の `level=4` で低次元 OOM が出るのは許容できない。
3. `shifted_anisotropic_gaussian` と `balanced_exponential` の full frontier / partial salvage を追加し、smooth / anisotropic / cancellation の 3 軸で mode の癖を比較する。
4. 最終的な研究 claim は「50D level15 に到達したか」だけでなく、「どの integrand class で、どの mode が、Monte Carlo に対して accuracy または time-to-accuracy で競争力を持つか」に置き直す。
