# Smolyak vs Monte Carlo: 10-Loop Research Report

日付: 2026-03-31  
worktree: `work/smolyak-integrator-opt-20260328`

## 目的

目標は「高次元で高精度かつ高速に積分できるか」を、Smolyak 積分器と Monte Carlo 積分の定量比較で詰めることだった。今回は単発の比較ではなく、実験 -> 批判的レビュー -> 修正/追加測定、のループを 10 回回し、理論面では既存の Gaussian 理論比較とインターネット上の一次資料も使って、どこまで理論保証と整合するかを確認した。

比較対象は一貫して Monte Carlo である。評価軸は次の 4 本に固定した。

- same-budget accuracy: Smolyak の `num_evaluation_points` と同じサンプル数の Monte Carlo を作り、絶対誤差と相対誤差を比較する。
- matched-accuracy runtime: Smolyak の絶対誤差に Monte Carlo が到達するまでサンプルを増やし、必要サンプル数と warm runtime を比べる。
- GPU throughput: `single` と `vmap(integrate(f))` を分けて、GPU 利用率と throughput の伸びを測る。
- frontier/failure: level を上げたときに精度優位が出るか、それとも OOM / timeout が先に来るかを見る。

## 実験一覧

今回実行した実験と、その役割を先に一覧化する。

| Loop | Script | Sweep / 条件 | 目的 | 主な出力 |
| --- | --- | --- | --- | --- |
| 1 | `report_smolyak_same_budget_accuracy` | `d=2,4,...,20`, `level=2,3,4`, `float32,float64`, Gaussian | broad same-budget baseline | `/tmp/smolyak_research/loop1_same_budget_baseline/report_20260331T110737Z/summary.json` |
| 2 | `report_smolyak_same_budget_accuracy` 再集約 | Loop 1 の cached cases を `resume` | report 集約バグ修正確認 | 同上 |
| 3 | `report_smolyak_theory_comparison` | Loop 1 の Gaussian summary | CLT 理論との整合確認 | `/tmp/smolyak_research/loop1_same_budget_baseline/report_20260331T110737Z/theory_summary.json` |
| 4 | `report_smolyak_vs_mc` | `d=2,4,8,12,16,20`, `level=3,4`, `float64`, Gaussian | matched-accuracy baseline | `/tmp/smolyak_research/loop2_matched_accuracy_float64_v2/report_20260331T111506Z/summary.json` |
| 5 | `report_smolyak_gpu_sweep` | `d=4,8,12,16,20`, `level=4`, `float64`, `vmap-batch-size=32` | GPU throughput / util / Pstate | `/tmp/smolyak_research/loop3_gpu_sweep_l4_f64/report_20260331T111233Z/summary.json` |
| 6 | `report_smolyak_same_budget_accuracy` | `d=8,12,16,20`, `level=5,6`, `float64`, Gaussian | high-level frontier と failure map | `/tmp/smolyak_research/loop4_same_budget_high_levels/report_20260331T111654Z/summary.json` |
| 7 | `compare_smolyak_vs_mc` | `(d8,l6)`, `(d12,l6)`, `(d16,l5)` に対して `chunk_size=16384,65536,262144` | materialized path の chunk tuning | 実行時標準出力で比較 |
| 8 | `report_smolyak_gpu_batch_scaling` | `d=8,12,16`, `level=5`, `float64`, `batch=2,4,8,16,32` | batched throughput scaling | `/tmp/smolyak_research/loop5_gpu_batch_scaling_l5_f64/report_20260331T112421Z/summary.json` |
| 9 | `report_smolyak_same_budget_accuracy` | `d=8,12,16`, `level=5,6`, `float32,float64`, `chunk_size=65536` | dtype tradeoff | `/tmp/smolyak_research/loop6_dtype_tradeoff_high_levels/report_20260331T112454Z/summary.json` |
| 10 | `compare_smolyak_vs_mc` 再現確認 | `d12,l5/l6`, `float32/float64`, 各 2 回 | dtype anomaly の再現性確認 | 実行時標準出力で比較 |

## 図表

主要な図は次のとおり。数表だけでは見えにくい「どの領域で勝敗が反転するか」「GPU がどこから効き始めるか」を、同じレポート上で追えるようにした。

### 1. Same-Budget Accuracy の全体像

低 level では高次元ほど Monte Carlo 優位だが、level 4 以降では一部で Smolyak が逆転する。

![Same-budget error ratio](assets/smolyak_mc_loop_report_20260331/same_budget_error_ratio.svg)

![Same-budget runtime ratio](assets/smolyak_mc_loop_report_20260331/same_budget_runtime_ratio_log.svg)

### 2. Matched-Accuracy の速度比較

「Smolyak と同精度に達するまで Monte Carlo が何サンプル要るか」を見る図で、level 不足の高次元では Monte Carlo のほうがむしろ少サンプルで足りていることがわかる。

![Matched-accuracy runtime](assets/smolyak_mc_loop_report_20260331/matched_warm_runtime.svg)

![Matched-accuracy points vs samples](assets/smolyak_mc_loop_report_20260331/points_vs_samples.svg)

### 3. GPU Throughput / Utilization / Pstate

GPU monitor は `nvidia-smi` ベースで動かしており、throughput だけでなく GPU utilization と Pstate も同時に記録した。軽いケースでは Pstate が高止まりし、点数が増えると利用率と throughput が一緒に伸びる。

![GPU throughput by dimension](assets/smolyak_mc_loop_report_20260331/gpu_throughput.svg)

![GPU utilization by dimension](assets/smolyak_mc_loop_report_20260331/gpu_util.svg)

![GPU Pstate by dimension](assets/smolyak_mc_loop_report_20260331/pstate.svg)

### 4. Batch Scaling

高次元・高 level ほど batch size を上げたときの伸びが大きく、今回の測定範囲では `batch_size=32` が一貫して最良だった。

![Batch throughput scaling](assets/smolyak_mc_loop_report_20260331/batch_throughput.svg)

![Batch speedup scaling](assets/smolyak_mc_loop_report_20260331/batch_vmap_speedup.svg)

### 5. 理論比較

Monte Carlo の観測誤差は CLT 理論 MAE とよく一致しており、Smolyak 側の勝敗判定が Monte Carlo の偶然に強く左右されていないことが確認できる。

![Observed MC vs theory](assets/smolyak_mc_loop_report_20260331/mc_theory_ratio_log.svg)

![Smolyak vs MC theory](assets/smolyak_mc_loop_report_20260331/smolyak_vs_mc_theory.svg)

## 主要数表

### Summary Table

| Experiment | Success / Total | Key metric 1 | Key metric 2 | Main reading |
| --- | ---: | --- | --- | --- |
| Same-budget broad baseline | `59 / 60` | Smolyak accuracy wins `20` | median MC/Smolyak error ratio `0.209` | level 4 までは高次元で MC 優位が多い |
| Matched-accuracy float64 | `11 / 12` | Smolyak runtime wins `2` | median MC/Smolyak runtime ratio `0.807` | 高次元では MC が少サンプルで追いつく |
| Theory comparison | `59 / 59` | median observed/theory MC ratio `0.995` | median Smolyak / theory-MC ratio `4.958` | MC 基準はかなり信頼できる |
| High-level frontier | `5 / 8` | `d8-l6` error ratio `330.27` | failures: `d20-l5 OOM`, `d16-l6 timeout`, `d20-l6 OOM` | 中高次元では勝てるが frontier は急に崩れる |
| GPU sweep `level=4` | `5 / 5` | `d20` batch speedup `32.25x` | `d20` avg util `36.5%`, peak `73%`, Pstate `P8` | batched case で初めて GPU が本格稼働 |
| GPU batch scaling `level=5` | `15 / 15` | best speedup `28.61x` (`d16,b32`) | `d12,b32` avg util `52.3%`, peak `73%`, Pstate `P5` | batch size 32 が最良 |

### Frontier Table

| Case | Smolyak points | Smolyak abs err | MC abs err | MC/Smolyak error ratio | Smolyak warm ms | MC warm ms | Storage bytes | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `d8-l5-f64` | `18319` | `3.96e-05` | `7.09e-04` | `17.88` | `1.392` | `0.208` | `1499696` | success |
| `d12-l5-f64` | `83932` | `1.40e-04` | `1.94e-04` | `1.39` | `1.417` | `0.918` | `10404064` | success |
| `d16-l5-f64` | `252845` | `3.65e-04` | `1.17e-04` | `0.321` | `1.506` | `1.248` | `53374064` | success |
| `d20-l5-f64` | - | - | - | - | - | - | - | OOM |
| `d8-l6-f64` | `101575` | `5.10e-07` | `1.69e-04` | `330.27` | `1.423` | `0.341` | `7743440` | success |
| `d12-l6-f64` | `659716` | `1.94e-05` | `6.37e-05` | `3.28` | `1.930` | `1.339` | `72836800` | success |
| `d16-l6-f64` | - | - | - | - | - | - | - | timeout 240 s |
| `d20-l6-f64` | - | - | - | - | - | - | - | OOM |

### GPU Monitor Table

| Experiment | Case | Best / observed batch | Throughput / speedup | Avg GPU util | Peak GPU util | Dominant Pstate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| GPU sweep | `d4-l4-f64` | `batch=32` | `5.44x` | `2.33%` | `5%` | `P2` |
| GPU sweep | `d8-l4-f64` | `batch=32` | `8.16x` | `3.33%` | `8%` | `P2` |
| GPU sweep | `d12-l4-f64` | `batch=32` | `7.59x` | `4.50%` | `6%` | `P2` |
| GPU sweep | `d16-l4-f64` | `batch=32` | `14.73x` | `5.75%` | `12%` | `P2` |
| GPU sweep | `d20-l4-f64` | `batch=32` | `32.25x` | `36.5%` | `73%` | `P8` |
| Batch scaling | `d8-l5-f64` | `batch=32` | `23729.95 ips / 15.38x` | `6.33%` | `11%` | `P2` |
| Batch scaling | `d12-l5-f64` | `batch=32` | `23062.28 ips / 25.77x` | `52.33%` | `73%` | `P5` |
| Batch scaling | `d16-l5-f64` | `batch=32` | `11588.60 ips / 28.61x` | `25.33%` | `68%` | `P2` |

## 実施した 10 ループ

### Loop 1: broad same-budget baseline

`d=2,4,...,20`, `level=2,3,4`, `dtype=float32,float64`, `family=gaussian`, `alpha=0.8`, `mc_seeds=8` を GPU で実行した。結果は 60 ケース中 59 成功、1 失敗で、失敗は `float64-d20-l4` の OOM だった。成功 59 ケースのうち Smolyak が same-budget で高精度だったのは 20 ケース、runtime で速かったのは 13 ケースだけだった。Monte Carlo / Smolyak の誤差比中央値は `0.209`、runtime 比中央値は `0.885` で、全体としては Monte Carlo がやや速く、精度でも優位なケースが多数派だった。

ただし level 別に切ると様子が変わる。level 2 の誤差比中央値は `0.0815`、level 3 は `0.1778` と Smolyak がかなり不利だったが、level 4 は `6.42` まで跳ね上がった。つまり「高次元だから負ける」のではなく、「level が低いまま高次元へ行くと負ける」が実態だった。

### Loop 2: aggregation bug review and fix

Loop 1 の集約時に `workspace_root` 参照が 1 箇所残って落ちた。ケース JSON は保存されていたので、集約側だけ修正して `resume-report-dir` で再集約した。これは実験ループとして重要で、長時間 run を壊すのは数値計算よりレポート側の脆さである、という批判的レビューにつながった。

### Loop 3: same-budget theory comparison

Loop 1 の summary を既存の理論比較スクリプトへ通した。観測された Monte Carlo MAE と CLT 理論 MAE の比の中央値は `0.995` で、Monte Carlo の観測誤差は理論と非常によく一致した。一方で Smolyak 誤差 / 理論 MC MAE の中央値は `4.96`、最良ケースは `float64-d2-l4` の `0.00109`、最悪ケースは `float64-d20-l3` の `122.43` だった。つまり Monte Carlo 側の比較基準はかなり信頼でき、Smolyak の不利は「MC が偶然悪かっただけ」ではない。

### Loop 4: matched-accuracy baseline

`d=2,4,8,12,16,20`, `level=3,4`, `dtype=float64` で matched-accuracy を回した。最初の実行では 1 ケース失敗で report 全体が落ちた。失敗点は `d20-l3`/`d20-l4` 近傍で、1 ケースの OOM/失敗が全 sweep を止める設計は研究用途として弱いと判断した。

### Loop 5: failure-tolerant matched-accuracy rerun

`report_smolyak_vs_mc.py` を修正し、ケース失敗を report 全体の失敗にしないようにした。あわせて `chunk-size` を report レベルまで伝播し、`compare_smolyak_vs_mc.py` に `storage_bytes`, `uses_materialized_plan`, `materialized_point_count` を追加した。修正後の rerun では 12 ケース中 11 成功、1 失敗 (`d20-l4` OOM) だった。成功 11 ケースのうち、matched-accuracy で Smolyak が速かったのは 2 ケースだけ、中央値の runtime 比は `0.807` だった。

重要なのは必要サンプル数である。たとえば `d16-l4` では Smolyak は `20073` 点使って same-budget でも Monte Carlo に精度負けしていたが、matched-accuracy では Monte Carlo はたった `64` サンプルで Smolyak に追いつき、runtime 比は `0.176` だった。`d20-l3` では `1871` 点に対し Monte Carlo は `1` サンプルで既に Smolyak 誤差以下に入った。高次元で level が足りないケースでは、Smolyak のコストは完全に過剰だった。

### Loop 6: GPU throughput sweep

`d=4,8,12,16,20`, `level=4`, `dtype=float64`, `vmap-batch-size=32` で GPU sweep を回した。`d20-l4` では single warm runtime が `2.48 ms`、batch warm runtime が `2.46 ms` で、throughput speedup は `32.25x` だった。平均 GPU 利用率は `36.5%`、peak は `73%` まで上がった。低次元では利用率は数 % に留まるが、点数が十分増えると batching が効いて GPU はちゃんと仕事をし始める。

### Loop 7: high-level frontier

`d=8,12,16,20`, `level=5,6`, `dtype=float64` で same-budget frontier を測った。結果はかなりはっきりしている。

- `d8-l5`: `18319` 点、Smolyak 誤差 `3.96e-05`、MC 誤差 `7.09e-04`、誤差比 `17.88`
- `d12-l5`: `83932` 点、誤差比 `1.39`
- `d16-l5`: `252845` 点、誤差比 `0.321`
- `d8-l6`: `101575` 点、誤差比 `330.27`
- `d12-l6`: `659716` 点、誤差比 `3.28`

一方で失敗は、

- `d20-l5`: OOM
- `d16-l6`: timeout 240 s
- `d20-l6`: OOM

だった。ここから言えるのは、Smolyak が本当に強くなる領域は存在するが、それは `d=8,12` あたりで十分 high level を入れたときであり、`d>=16` ではその領域に入る前にメモリか時間が先に尽きやすいということだ。

### Loop 8: chunk-size tuning

materialized path が使われている代表ケース `d8-l6`, `d12-l6`, `d16-l5` で `chunk_size = 16384, 65536, 262144` を比較した。結果はケース依存だった。

- `d12-l6`: `1.81 ms -> 1.41 ms -> 1.36 ms`
- `d8-l6`: `1.37 ms` 前後でほぼ不変
- `d16-l5`: `1.38 ms` 前後でほぼ不変

つまり chunk-size は万能ノブではないが、中規模以上の materialized case では 20-25% 程度の改善余地がある。そこで後続の高レベル比較では `chunk-size=65536` も試した。

### Loop 9: GPU batch scaling

`d=8,12,16`, `level=5`, `dtype=float64` で batch size を `2,4,8,16,32` に振った。最良 batch size は 3 次元すべてで `32` だった。

- `d8`: best throughput `23729.95 integrals/s`, speedup `15.38x`
- `d12`: best throughput `23062.28 integrals/s`, speedup `25.77x`
- `d16`: best throughput `11588.60 integrals/s`, speedup `28.61x`

高次元・高レベルほど batching の恩恵が大きく、特に `d12` と `d16` では batch size を詰める価値が高い。

### Loop 10: dtype tradeoff and reproducibility check

`d=8,12,16`, `level=5,6`, `dtype=float32,float64`, `chunk-size=65536` で same-budget を再測定した。ここでは「float64 にすれば必ず良くなる」とは言えなかった。

- `d8-l6`: float64 誤差は float32 の `0.348x` まで改善したが、runtime は `4.63x`
- `d12-l5`: float64 誤差は float32 より `1.05x` 悪く、runtime は `3.74x`
- `d12-l6`: float64 誤差は float32 より `1.62x` 悪く、runtime は `1.19x`
- `d16-l6`: float32 は OOM、float64 は timeout

この挙動が偶然かを確かめるため、`d12-l5` と `d12-l6` を各 dtype 2 回ずつ再実行した。Smolyak の absolute error は再現性があり、`float32-d12-l6 = 1.1964965e-05`、`float64-d12-l6 = 1.9438220e-05` は 2 回とも一致した。したがって、この anomaly は MC ノイズではなく Smolyak 側の数値挙動に由来している可能性が高い。少なくとも現実装では「float64 が単調に良くなる」は仮定できない。

## 批判的レビュー

1. 現状の Smolyak は「高速化」より前に「どの次元で十分な level を入れられるか」が支配的である。`d16+` では精度優位を出す前に OOM/timeout が来やすい。
2. Monte Carlo の observed/theory 比が `0.995` なので、比較相手の基準はかなり妥当である。Smolyak の不利を Monte Carlo 側の偶然には押し付けられない。
3. level 4 以下の高次元では、same-budget でも matched-accuracy でも Monte Carlo 優位が目立つ。`d16-l4` で MC が `64` サンプルで十分なのは象徴的で、Smolyak の sparse grid 自体より「その level が足りない」ことが問題。
4. 逆に `d8-l6` のように十分高 level へ行けると、Smolyak は same-budget で `330x` の精度優位を出せる。理論的な優位は確かに存在する。
5. materialized path は成功した高レベルケースでは常に使われていた。これは前向きだが、storage bytes は `d12-l6` で既に `72,836,800` byte に達しており、frontier のボトルネックは明確にメモリへ寄っている。
6. chunk-size 調整は有効だが、構造的な精度不足は救えない。これは speed knob であって accuracy knob ではない。
7. float64 の非単調性は無視できない。difference-rule の加算順序、重みの打ち消し、JAX/XLA の縮約順序などを疑うべきで、ここは今後の重点レビュー対象である。

## 今回入れた工夫

- `report_smolyak_vs_mc.py` を failure-tolerant にして、1 ケース失敗で全 report を落とさないようにした。
- `report_smolyak_same_budget_accuracy.py` / `report_smolyak_vs_mc.py` に `chunk-size` を通し、比較実験の制御ノブを増やした。
- `compare_smolyak_vs_mc.py` に `storage_bytes`, `uses_materialized_plan`, `materialized_point_count` を追加し、実装経路とメモリ負荷を観測可能にした。
- report 集約の `workspace_root` 参照バグを修正し、長時間 run の resume を壊さないようにした。

## 理論・文献との整合

インターネット上の一次資料では、Hinrichs, Novak, Ullrich (2013) が Clenshaw-Curtis Smolyak に対して解析的関数クラスで weak tractability を示している。Bungartz and Griebel (2004) は sparse grid の自由度が `O(N (\log N)^{d-1})` に抑えられ、混合微分が十分滑らかな場合に高精度化できることを整理している。一方、Novak and Wozniakowski の tractability monograph では Monte Carlo の randomized error が本質的に `n^{-1/2}` で、variance に依存する条件下で tractable になりうることが明示されている。

今回の結果はこれと矛盾しない。Gaussian は解析的で、`d8-l6` と `d12-l6` では Smolyak が same-budget で大きな精度優位を示したので、「理論が示唆する非自明な優位」は実際に観測できた。しかしその優位は有限計算資源の下では自動的には現れず、`d16+` では asymptotic regime に入る前に OOM/timeout が先に来る。言い換えると、理論保証は長期的傾向としては支えるが、現実の frontier は memory wall と implementation overhead に強く切られている。

## 現時点の結論

現実的な sweet spot は「中高次元 (`d≈8..12`) で、かなり高い level (`5..6`) を入れられる範囲」である。この帯域では Smolyak は Monte Carlo に対して same-budget で明確な精度優位を持つ。一方で `d>=16` では、現実装と現 GPU メモリでは high-accuracy/high-speed/high-dimension を同時に満たせていない。次にやるべきことは、Smolyak 自体の理論を疑うことではなく、`d16+` の frontier を押し上げるためのメモリ削減と数値安定化、とくに difference-rule の加算安定性と materialized path のさらなる圧縮である。

## 参照した主要資料

- A. Hinrichs, E. Novak, M. Ullrich, “On Weak Tractability of the Clenshaw-Curtis Smolyak Algorithm”, arXiv:1309.0360, 2013. https://arxiv.org/abs/1309.0360
- H.-J. Bungartz, M. Griebel, “Sparse grids”, Acta Numerica 13, 2004. https://doi.org/10.1017/S0962492904000182
- E. Novak, H. Woźniakowski, *Tractability of Multivariate Problems, Volume I*, 2008. https://users.fmi.uni-jena.de/~novak/2008NW-Vol1.pdf
