# Smolyak Integrator Tuning Report

## 1. Scope

この report は、Smolyak 積分器まわりで行った設計変更、HLO 解析、experiment runner の見直し、旧版と tuned 版の実験結果の読みを、`main` から 1 ファイルで読める形にまとめたものである。対象 branch は主に `results/functional-smolyak-scaling`、`work/smolyak-tuning-20260316`、`work/experiment-runner-module-20260316`、`results/functional-smolyak-scaling-tuned` である。

この report の役割は、過去の作業ログを時系列で追うことではなく、Smolyak 積分器について「何を変え」「何を測り」「何が分かったか」を技術報告として再構成することである。詳細な断片メモは reference として末尾に置き、本体はこの file 単体で理解できることを優先する。

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

## 3. Legacy Implementation

旧版 Smolyak 実装は、1D rule を組み合わせて multi-index を列挙し、各 term の tensor-product points を host 上で明示的に構築したうえで、重複点を global に統合し、最後に weights を掛けて積分値を得る構造だった。数式上は

$$
A(q,d)=\sum_{|i|_1 \le q+d-1}\Delta_{i_1}\otimes\cdots\otimes\Delta_{i_d}
$$

という Smolyak の和を実装しているが、実装では「全点をまとめて持つ」ことが強く前提になっていた。

この構造の問題は、数式自体より保持戦略にあった。point 配列、weight 配列、term ごとの一時 tensor、重複統合のための補助配列が host 側に集中するため、CPU RSS と pinned host memory が先に限界へ達しやすい。実際、旧版 GPU run では GPU 利用率が低く見える一方で、host 側の grid 構築が支配的だった。

## 4. Why Tuning Was Needed

旧版の結果からは二つの問題が見えた。第一に、低精度 dtype では level を上げるほど誤差が悪化し、`float64` では level とともに改善するという、数値的な差が見えていた。第二に、それとは別に host 側の明示構築コストが大きく、GPU を積分 kernel に使う前段で時間とメモリを消費していた。

このため tuning の狙いは、単に数値精度を良くすることではなく、explicit grid の依存を弱めて、積分器の評価構造そのものを軽くすることにあった。特に、「公開 API として全点を expose する必要はあるか」「grid を本当に大域的に materialize する必要はあるか」という点が中心論点だった。

## 5. Structural Changes in the Tuned Integrator

tuned 版では、公開経路として explicit grid を返す設計をやめ、`initialize_smolyak_integrator(...)` で plan を作り、評価は `integrate(f, integrator)` へ寄せた。これにより、評価 API は「関数を積分する」という責務に集中し、内部表現は point cloud より rule table と term plan に寄せられた。

また、`prepared_level` を持つ積分器として、最大 level までの 1D rule を先に準備し、その範囲内では `refine()` で level を上げられる形にした。実行時は単一精度を基本とし、不要になった explicit-grid helper や重複する low-level path は削除した。設計上は、mesh を全面展開するより「fixed dimension を再帰でたどり、leaf で batched evaluation する」方向へ寄せた。

ただし、これにより「点数爆発」は抑えられても、JIT / lowering / control-flow 側のコストが前面に出る可能性が生じた。実際の tuning は、ここから先を実験と HLO で切り分ける作業だった。

## 6. HLO Analysis

HLO 解析のために、`jax_util.hlo.dump` と `scripts/hlo/summarize_hlo_jsonl.py` を拡張し、`stablehlo` と `hlo` の text size、proto size、compiled memory stats、cost analysis を JSONL と summary へ保存できるようにした。さらに `experiments/functional/smolyak_hlo/` を作り、単一ケースの lowering から粗い bottleneck を見る実験を追加した。

この解析から一貫して見えたのは、算術 kernel より

- `stablehlo.while`
- `func.call`
- `stablehlo.gather`

が目立つことである。小さい case でもこの傾向は安定しており、演算密度の高い大きな `dot` が主役というより、制御フローと index 処理が先に見える。したがって tuned 版で explicit grid を減らしても、ただちに「GPU に大量の算術を投げる構造」にはなっていない。

## 7. GPU Visibility And Runner Behavior

GPU 1,2 が遊んで見える問題については、`debug_gpu_visibility.py` により `ProcessPoolExecutor` と subprocess の両方で child ごとの `CUDA_VISIBLE_DEVICES` が有効であることを確認した。child 側では `jax.devices()` が local に `[cuda:0]` のみを返しており、GPU 分離自体は成立していた。したがって、GPU が使われていないように見える主因は可視性バグではなく、CPU 側初期化が長く、短い GPU 実行が観測上見えにくいことだった。

この切り分けと並行して、実験 runner も `jax_util.experiment_runner` へ切り出した。host が free slot と timeout を見ながら child を dispatch し、child はケース実行と JSONL 追記を担当したうえで completion record を明示的に返す形にした。この設計変更により、長時間 run の途中停止でもケース単位の JSONL が残り、multi-GPU 実験の管理もかなり安定した。

## 8. Legacy Results

旧版 results branch では、完了済み GPU run と途中停止 partial run の両方が得られた。完了済み run では `float16`, `bfloat16`, `float32`, `float64` の 4 dtype を最後まで比較でき、低精度では level を上げるほど誤差が悪化し、`float64` では level とともに改善するという数値傾向が見えた。これは Smolyak 法そのものの離散化誤差というより、低精度での加算誤差や cancellation の影響を強く示唆していた。

一方、途中停止 partial run は当時の case 順が `dtype -> level -> dimension` だったため、先頭の dtype が先に大量に進み、比較用データとしては偏っていた。ここから得られた重要な教訓は、長時間 run では case ordering 自体が結果の解釈可能性に影響するということである。この教訓は tuned 版 runner の case ordering 見直しへつながった。

## 9. Tuned Partial Results

tuned 版の partial では、`547` ケース時点で `ok=99`, `failed=448` だった。dtype の進み方はほぼ均等で、`float16=137`, `bfloat16=137`, `float32=137`, `float64=136` である。観測できた level は事実上 `0` と `1` に限られ、`level=0` は仕様上の failure accounting を確認する層、実質的な情報は `level=1` に集中していた。

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

## 10. What We Know Now

ここまでで分かったことをまとめると、Smolyak 積分器の tuning は「explicit grid をやめれば終わり」ではない。旧版では grid 明示構築と host memory が主因だったが、tuned 版では初期化、lowering、compile、制御フロー、index 処理が前面に出ている。つまり、ボトルネックは移動したのであって、まだ解消していない。

一方で、runner 側の問題はかなり整理できた。GPU 分離は成立しており、JSONL 逐次保存、child completion record、case ordering 見直しによって、大規模実験の運用は以前より安定した。したがって、今後の本筋は runner ではなく積分器初期化 path の分析と整理である。

## 11. Next Steps

次に必要なのは、`level=1` でも次元 25 前後から重くなる理由を、さらに分解することである。具体的には、HLO 解析で見えている `while` / `call` / `gather` の比率を、実行時間と memory 使用量に結びつける必要がある。また、tuned 版 partial は case ordering を `dimension -> level -> dtype` に変更した新 run が進行中なので、今後はより読みやすい frontier が得られるはずである。

この report の段階では、Smolyak 法の数値精度の最終比較を結論するのではなく、「現時点での実装はまだ initialization-bound であり、数値比較より前に構造コストの整理が必要」ということを、最も重要な結論として残す。

## 12. Data And References

### Final JSON Archived In Main

- [tuned_smolyak_partial_results_20260316.json](/workspace/notes/experiments/results/tuned_smolyak_partial_results_20260316.json)

### Results Branch Artifacts

- legacy complete GPU run:
  - [smolyak_scaling_gpu_20260315T140215Z.json](/workspace/.worktrees/results-functional-smolyak-scaling-tuned/experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260315T140215Z.json)
- tuned partial JSON:
  - [smolyak_scaling_gpu_20260316T132125Z_partial.json](/workspace/.worktrees/results-functional-smolyak-scaling-tuned/experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260316T132125Z_partial.json)
- tuned partial report:
  - [index.html](/workspace/.worktrees/results-functional-smolyak-scaling-tuned/experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260316T132125Z_partial_report/index.html)

### Supporting Notes

- [Legacy Smolyak Results](/workspace/notes/experiments/legacy_smolyak_results_20260316.md)
- [Tuned Smolyak Partial Results](/workspace/notes/experiments/tuned_smolyak_partial_results_20260316.md)
- [Smolyak Tuning Worktree Extraction](/workspace/notes/worktrees/worktree_smolyak_tuning_2026-03-16.md)
- [Experiment Runner Modularization Worktree Extraction](/workspace/notes/worktrees/worktree_experiment_runner_module_2026-03-16.md)
