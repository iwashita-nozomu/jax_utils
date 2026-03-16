# Smolyak 積分器の設計と観測結果

## 1. Scope

この report は、Smolyak 積分器について、何を試し、何を観測し、何が分かったかを 1 ファイルで読める形にまとめたものです。

ここでは、内部の branch 名や作業順ではなく、方法そのものを主語にします。

履歴は reference に回し、本体は外から読んでも分かる説明を優先します。

## 2. Problem Setting

対象の積分は、主として

$$
\Omega_d = [-1/2, 1/2]^d
$$

上でのベクトル値関数積分である。解析解を持つ基準問題としては、係数ベクトル $a \in \mathbb{R}^d$ に対する指数関数型

$$
f_a(x) = \exp(a^\top x)
$$

を多く使った。このとき積分は分離し、

$$
\int_{\Omega_d} \exp(a^\top x)\,dx
=
\prod_{k=1}^d
\frac{2\sinh(a_k/2)}{a_k},
$$

ただし $a_k = 0$ では極限値 $1$ を取る。非線形性を見るために、ガウス型

$$
g_\alpha(x) = \exp(-\alpha \|x\|_2^2)
$$

も用いた。これも 1 次元積分に分離でき、

$$
\int_{\Omega_d} g_\alpha(x)\,dx
=
\left(
\sqrt{\frac{\pi}{\alpha}}
\operatorname{erf}\!\left(\frac{\sqrt{\alpha}}{2}\right)
\right)^d.
$$

このため、誤差比較と scaling 比較を同じ実験系で扱いやすい。

## 3. 点群を全部作る方法

まず試した方法は、1D rule を組み合わせて multi-index を列挙し、各 term の tensor-product points を host 上で明示的に構築し、重複点を global に統合してから積分する方法です。数式上は

$$
A(q,d)=\sum_{|i|_1 \le q+d-1}\Delta_{i_1}\otimes\cdots\otimes\Delta_{i_d}
$$

という Smolyak の和を実装しているが、実装では「全点をまとめて持つ」ことが強く前提になっていた。

問題は数式ではなく保持戦略にありました。point 配列、weight 配列、term ごとの一時 tensor、重複統合のための補助配列が host 側に集中するため、CPU RSS と pinned host memory が先に限界へ達しやすくなりました。実際、GPU 利用率が低く見える場面でも、支配的だったのは host 側の grid 構築でした。

Source:
Smolyak の和そのものは [Holtz 2010, p. 58, Eq. (4.5)](/workspace/references/978-3-642-16004-2.pdf) に対応します。

Source:
storage が sparse grid 実装の主要論点になること、hash table や tree が典型であることは [Murarasu 2013, p. 20](/workspace/references/sparse_grid/Murarasu_2013_PhD_Advanced_Optimization_Techniques_for_Sparse_Grids_on_Modern_Heterogeneous_Systems.pdf) [Murarasu 2013, p. 43](/workspace/references/sparse_grid/Murarasu_2013_PhD_Advanced_Optimization_Techniques_for_Sparse_Grids_on_Modern_Heterogeneous_Systems.pdf) に対応します。

## 4. なぜ別の方法が必要だったか

この方法からは二つの問題が見えました。第一に、低精度 dtype では level を上げるほど誤差が悪化し、`float64` では level とともに改善するという数値差がありました。第二に、それとは別に、host 側の明示構築コストが大きく、GPU を積分 kernel に使う前に時間とメモリを消費していました。

そこで狙ったのは、単に精度を上げることではありません。点群の全保持をやめ、積分器の評価構造そのものを軽くすることでした。特に、「公開 API として全点を返す必要があるか」「grid を本当に大域的に materialize する必要があるか」が中心論点でした。

Consideration:
この節の `全点保持をやめるべきだ` という判断は project 内の設計判断です。文献の直接引用ではありません。

## 5. plan を先に作って評価する方法

次に試したのは、公開経路として explicit grid を返す設計をやめ、`initialize_smolyak_integrator(...)` で plan を作り、`integrate(f, integrator)` で評価する方法です。これにより、評価 API は「関数を積分する」という責務に集中し、内部表現は point cloud より rule table と term plan に寄せられました。

また、`prepared_level` を持つ積分器として、最大 level までの 1D rule を先に準備し、その範囲内では `refine()` で level を上げられる形にしました。実行時は単一精度を基本とし、不要になった explicit-grid helper や重複する low-level path は削除しました。設計上は、mesh を全面展開するより「fixed dimension を再帰でたどり、leaf で batched evaluation する」方向へ寄せました。

ただし、これで「点数爆発」は弱まっても、JIT、lowering、control-flow 側のコストが前面に出る可能性が生じました。ここから先は、実験と HLO でその切り分けを進めました。

Observation:
この節の後半は project 内の観測です。文献ではなく、実験結果に基づいています。

## 6. HLO Analysis

HLO 解析のために、`jax_util.hlo.dump` と `scripts/hlo/summarize_hlo_jsonl.py` を拡張し、`stablehlo` と `hlo` の text size、proto size、compiled memory stats、cost analysis を JSONL と summary へ保存できるようにした。さらに `experiments/functional/smolyak_hlo/` を作り、単一ケースの lowering から粗い bottleneck を見る実験を追加した。

この解析から一貫して見えたのは、算術 kernel より

- `stablehlo.while`
- `func.call`
- `stablehlo.gather`

が目立つことである。小さい case でもこの傾向は安定しており、演算密度の高い大きな `dot` が主役というより、制御フローと index 処理が先に見える。したがって、plan を先に作る方法にしても、ただちに「GPU に大量の算術を投げる構造」にはなっていない。

Observation:
この節は HLO dump に基づく project 内の観測です。

## 7. GPU Visibility And Runner Behavior

GPU 1,2 が遊んで見える問題については、`debug_gpu_visibility.py` により `ProcessPoolExecutor` と subprocess の両方で child ごとの `CUDA_VISIBLE_DEVICES` が有効であることを確認した。child 側では `jax.devices()` が local に `[cuda:0]` のみを返しており、GPU 分離自体は成立していた。したがって、GPU が使われていないように見える主因は可視性バグではなく、CPU 側初期化が長く、短い GPU 実行が観測上見えにくいことだった。

この切り分けと並行して、実験 runner も `jax_util.experiment_runner` へ切り出した。host が free slot と timeout を見ながら child を dispatch し、child はケース実行と JSONL 追記を担当したうえで completion record を明示的に返す形にした。この設計変更により、長時間 run の途中停止でもケース単位の JSONL が残り、multi-GPU 実験の管理もかなり安定した。

Observation:
この節は project 内の実装と実験運用の結果です。

## 8. 点群を全部作る方法で見えたこと

この方法では、完了済み run と途中停止 partial run の両方が得られました。完了済み run では `float16`, `bfloat16`, `float32`, `float64` の 4 dtype を最後まで比較でき、低精度では level を上げるほど誤差が悪化し、`float64` では level とともに改善するという数値傾向が見えました。これは Smolyak 法そのものの離散化誤差というより、低精度での加算誤差や cancellation の影響を強く示しています。

Observation:
この節は complete run と partial run の project 内結果に基づきます。

一方、途中停止 partial run では case 順が `dtype -> level -> dimension` だったため、先頭の dtype が先に大量に進み、比較用データとしては偏りました。ここから得られた重要な教訓は、長時間 run では case ordering 自体が結果の読みやすさに影響するということです。

Observation:
この節は project 内の運用知見です。

## 9. plan を先に作る方法で見えたこと

この方法の partial では、`547` ケース時点で `ok=99`, `failed=448` でした。dtype の進み方はほぼ均等で、`float16=137`, `bfloat16=137`, `float32=137`, `float64=136` でした。観測できた level は事実上 `0` と `1` に限られ、`level=0` は仕様上の failure accounting を確認する層、実質的な情報は `level=1` に集中していました。

重要なのは、`level=1` の成功ケースが全 dtype で `num_points=1` に留まっているにもかかわらず、次元とともに `integrator_init_seconds` と `process_rss_mb` が急増していることである。たとえば成功末尾は

- `float16`: `d=25`, `integrator_init_seconds=16.32s`, `process_rss_mb=3659.8`
- `bfloat16`: `d=25`, `14.49s`, `3612.6 MB`
- `float32`: `d=27`, `54.91s`, `12646.4 MB`
- `float64`: `d=24`, `6.99s`, `2080.5 MB`

だった。一方 `avg_integral_seconds` はどの dtype でもほぼ `10^{-4}` から `10^{-3}` 秒台であり、

$$
\text{integration kernel} \ll \text{initializer / lowering / memory}
$$

という不均衡がきわめて明瞭である。

失敗種類も、純粋な数値精度限界より実行基盤の制約を示している。全体では `error=412`, `oom=8`, `worker_terminated=28` であり、`level=1` の frontier 付近では `oom` と `worker_terminated` が混在する。次元ごとに見ると `d=20..23` は全 dtype 成功、`d=24..25` から dtype ごとに OOM が混じり始め、`d=26` では全 dtype OOM、`d>=28` では `worker_terminated` が増える。したがって、この partial は「どの dtype が最も精度が高いか」を示すより、「今の実装がどこで不安定になるか」を示す資料として読むのが正しい。

Observation:
この節は `tuned_smolyak_partial_results_20260316.json` の集計に基づきます。

## 10. ここまでで言えること

ここまでで分かったことをまとめると、Smolyak 積分器では「点群を全部作る方法」をやめれば終わり、ではありません。点群全保持の方法では grid 明示構築と host memory が主因でしたが、plan を先に作る方法では初期化、lowering、compile、制御フロー、index 処理が前面に出ています。つまり、ボトルネックは移動したのであって、まだ解消していません。

一方で、runner 側の問題はかなり整理できた。GPU 分離は成立しており、JSONL 逐次保存、child completion record、case ordering 見直しによって、大規模実験の運用は以前より安定した。したがって、今後の本筋は runner ではなく積分器初期化 path の分析と整理である。

## 11. 次にやること

次に必要なのは、`level=1` でも次元 25 前後から重くなる理由を、さらに分解することです。具体的には、HLO 解析で見えている `while`、`call`、`gather` の比率を、実行時間と memory 使用量に結びつける必要があります。また、case ordering は `dimension -> level -> dtype` に変えたので、今後はより読みやすい frontier が得られるはずです。

この report の段階で最も重要なのは、「現時点の実装はまだ initialization-bound であり、数値比較より前に構造コストの整理が必要」という点です。

## Addendum: 定量スナップショット

`level=1` の範囲だけを切り出しても、いくつかの重要な定量事実があります。

- `float16`
  - 成功最大次元 `25`
  - 最初の失敗次元 `26`
- `bfloat16`
  - 成功最大次元 `25`
  - 最初の失敗次元 `26`
- `float32`
  - 成功最大次元 `27`
  - ただし `24` で OOM があり、frontier は単調でない
- `float64`
  - 成功最大次元 `24`
  - 最初の失敗次元 `25`

この結果は、`float32` が一様に優れるとか、`float64` が常に最後まで残るといった単純な物語を支持しません。むしろ、現在見えている frontier が、数値誤差だけでなくメモリ・初期化・child 終了の不安定性を含んでいることを示しています。

Observation:
この節は `tuned_smolyak_partial_results_20260316.json` の再集計に基づきます。

## Addendum: まだ言えないこと

この report は、Smolyak 積分の最終的な数値比較を与えるものではありません。今の段階では、少なくとも次の点は未確定です。

- 高 level でどの dtype が最も安定か
- plan ベースの方法が最終的にどこまで伸びるか
- ここで見えている OOM のどこまでが compile 起因か

これらは、初期化時間の分解、child 側 backend 初期化の安定化、より進んだ partial の取得なしには詰めきれません。

## 12. Data And References

### Final JSON Archived In Main

- [tuned_smolyak_partial_results_20260316.json](/workspace/notes/experiments/results/tuned_smolyak_partial_results_20260316.json)

### Results Branch Artifacts

- complete run for the point-materialization method:
  - [smolyak_scaling_gpu_20260315T140215Z.json](/workspace/.worktrees/results-functional-smolyak-scaling-tuned/experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260315T140215Z.json)
- partial JSON for the plan-based method:
  - [smolyak_scaling_gpu_20260316T132125Z_partial.json](/workspace/.worktrees/results-functional-smolyak-scaling-tuned/experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260316T132125Z_partial.json)
- partial report for the plan-based method:
  - [index.html](/workspace/.worktrees/results-functional-smolyak-scaling-tuned/experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260316T132125Z_partial_report/index.html)

### Supporting Notes

- [Smolyak partial results note](/workspace/notes/experiments/tuned_smolyak_partial_results_20260316.md)
- [Explicit-grid experiment note](/workspace/notes/experiments/legacy_smolyak_results_20260316.md)
- [Tuning worktree extraction](/workspace/notes/worktrees/worktree_smolyak_tuning_2026-03-16.md)
- [Runner modularization worktree extraction](/workspace/notes/worktrees/worktree_experiment_runner_module_2026-03-16.md)

### Literature

- [Markus Holtz, Sparse Grid Quadrature in High Dimensions with Applications in Finance and Insurance, 2010](/workspace/references/978-3-642-16004-2.pdf)
- [Adina-Eliza Murarasu, Advanced Optimization Techniques for Sparse Grids on Modern Heterogeneous Systems, 2013](/workspace/references/sparse_grid/Murarasu_2013_PhD_Advanced_Optimization_Techniques_for_Sparse_Grids_on_Modern_Heterogeneous_Systems.pdf)
- [Fredrik N. Lastdrager and Barry Koren, Error analysis for function representation by the sparse grid combination technique, 1998](/workspace/references/sparse_grid/Lastdrager_Koren_1998_Error_analysis_for_function_representation_by_the_sparse_grid_combination_technique.pdf)
- [J. D. Jakeman and S. G. Roberts, Local and Dimension Adaptive Sparse Grid Interpolation and Quadrature, 2011](/workspace/references/sparse_grid/Jakeman_Roberts_2011_Local_and_Dimension_Adaptive_Sparse_Grid_Interpolation_and_Quadrature.pdf)
- [Mark Hegland, Adaptive sparse grids, 2003](/workspace/references/sparse_grid/Hegland_2003_Adaptive_sparse_grids.pdf)

## Addendum: 主張と根拠の対応

- `Smolyak の和そのもの`
  - Source: Holtz 2010, p. 58, Eq. (4.5)
- `sparse grid では storage が主要論点になりうる`
  - Source: Murarasu 2013, p. 20, p. 43, p. 49
- `combination technique は独立の誤差解析対象になる`
  - Source: Lastdrager-Koren 1998, p. 3-4
- `adaptive quadrature と dimension adaptivity は既存の主題である`
  - Source: Jakeman-Roberts 2011, p. 2-4
  - Source: Hegland 2003, p. 1-4
- `今の実装は initialization-bound である`
  - Observation: `tuned_smolyak_partial_results_20260316.json`
  - Observation: HLO dump summary
- `GPU 分離は成立していた`
  - Observation: `debug_gpu_visibility.py` の確認結果
  - Observation: multi-GPU 実験ログ
