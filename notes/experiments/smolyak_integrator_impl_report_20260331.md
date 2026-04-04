# Smolyak Integrator Implementation Report

日付: 2026-03-31  
worktree: `work/smolyak-integrator-opt-20260328`

## 要旨

今回の 10 ループは、実験ハーネスではなく `SmolyakIntegrator` 本体の execution frontier を押し上げることに集中した。結論を先に書くと、`full points materialization` の下に `indexed materialization` を追加したことで、高次元・高 level で batched fallback に落ちる前の帯域を明確に広げられた。`level=6` では `d=13` から、`level=7` では `d=10` から `indexed` へ切り替わり、`points` とほぼ同等か、場合によってはそれ以上の warm runtime を保ったまま storage を大きく削減できた。

一方で、Monte Carlo と比べた warm runtime はまだ不利である。same-budget では `level=5,6` の `d=1..12` 全 24 ケースで Smolyak が Monte Carlo より高精度だったが、同じ 24 ケースのうち Smolyak が warm runtime で勝ったのは 3 件しかなく、`level=6` では 0 件だった。したがって現段階の本当の到達点は「高精度 high-dimension integration をより大きな領域で実行可能にした」であって、「Monte Carlo より速くした」ではない。この点は批判的に維持しておくべきである。

## 今回の実装変更

- [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py) に `indexed materialization` を追加した。point 座標を全部保持する代わりに rule index と重みだけを持つ中間経路である。
- `full points` の閾値を 96 MiB に下げ、より大きい帯域は `indexed`、さらに大きい帯域は従来の `batched` に落とす 3 段構成にした。
- `_rule_storage` を再利用する経路でも materialization 判定が走るように整理し、`prepared_level` / `refine()` を使う場合でも実装経路が揃うようにした。
- [compare_smolyak_vs_mc.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/compare_smolyak_vs_mc.py) に `materialization_mode` と `materialized_index_dtype` を載せ、実験結果から経路が追えるようにした。
- [report_smolyak_same_budget_accuracy.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/report_smolyak_same_budget_accuracy.py) と [report_smolyak_gpu_sweep.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/report_smolyak_gpu_sweep.py) の SVG に x 軸ラベルを追加し、各図の下に読み方を 1 行付けた。
- 回帰確認として [test_smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/tests/functional/test_smolyak.py) に `indexed` 経路の一致テストを追加した。`pytest` は 13 passed。

## 何を計算したか

今回の主対象は、立方体 `[-0.5, 0.5]^d` 上の Gaussian 積分

`I_d(alpha) = \int_{[-0.5,0.5]^d} exp(-alpha * ||x||_2^2) dx`

である。実験では `alpha = 0.8` を使った。比較したい点は「次元 `d` と sparse-grid level `l` を上げたとき、Smolyak が Monte Carlo よりどれくらい高精度か、またその精度を出すためにどれだけメモリと時間を使うか」である。

この積分は解析解が既知で、

`I_d(alpha) = (sqrt(pi / alpha) * erf(0.5 * sqrt(alpha)))^d`

なので、各 run で絶対誤差と相対誤差を直接計算できる。つまり今回の比較は「数値積分どうしを相互比較している」のではなく、「両者を解析解に対して採点している」。

Smolyak 側では [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py) の `SmolyakIntegrator(dimension=d, level=l, dtype=..., chunk_size=16384)` を構築し、Clenshaw-Curtis の difference rule を組み合わせた sparse-grid quadrature を実行した。Monte Carlo 側では同じ領域 `[-0.5, 0.5]^d` に一様サンプルを生成し、同じ integrand を平均した。

## 共通の測定方法

### Same-budget 比較

same-budget では、まず Smolyak を 1 回構築して `num_evaluation_points = N` を読む。次に、

1. `integrate(f)` を JIT compile して 1 回実行し、first call 時間を測る。
2. その後 3 回繰り返して warm runtime の平均を取る。
3. Monte Carlo はサンプル数を必ず `N` に固定する。
4. 乱数 seed は 8 本使い、absolute error / relative error / runtime はその平均と標準偏差を保存する。

という手順で比較した。したがって same-budget の各表は「Smolyak が使った評価点数そのもの」を Monte Carlo にそのまま与えたときの比較である。

### `compare_smolyak_vs_mc` の追加計算

[compare_smolyak_vs_mc.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/smolyak_experiment/compare_smolyak_vs_mc.py) を使ったケースでは、same-budget だけでなく Monte Carlo の geometric search も同時に回している。サンプル数は `1, 4, 16, 64, ...` と 4 倍ずつ増やし、Smolyak の absolute error 以下に Monte Carlo が入るまで追う。今回の報告では主に same-budget の数値を使ったが、JSON にはこの探索履歴も残してある。

### GPU sweep

GPU sweep では same-budget の誤差比較ではなく、Smolyak 単体の execution 特性を測った。各次元で次の 2 つの計算を回している。

1. `single`: `integrate(f_alpha)` を 1 本だけ実行する。
2. `batch`: `alpha` を 16 個まとめた `vmap(integrate(f_alpha))` を実行する。

ここで `alpha` は `0.8 * 0.75 = 0.6` から `0.8 * 1.25 = 1.0` までを線形分割している。GPU の状態は `nvidia-smi --loop-ms=100` で監視し、最低 600 ms は monitor が動くように warm call を追加した。保存したのは throughput、speedup、平均/最大 GPU utilization、平均/最大 memory utilization、peak memory used、dominant/min Pstate である。

### Forced-mode microbenchmark

`points / indexed / batched` の 3 経路比較は、report script ではなく小さな inline benchmark で行った。ここでは

1. `_MAX_FULL_MATERIALIZED_PLAN_BYTES`
2. `_MAX_INDEXED_MATERIALIZED_PLAN_BYTES`

を一時的に変えて、同じ `(d, l, dtype)` でも強制的に特定の execution path を通す。そうしてから同じ Gaussian 積分を JIT 実行し、`init_ms`, `first_ms`, `warm_ms`, `storage_mb` を比較した。つまりこの表は「同じ数学問題を、実装経路だけ変えて比べた」結果である。

## 実施した 10 ループ

### Loop 1: high-level same-budget sweep

最初に回した主 run は、

```bash
python3 -m experiments.smolyak_experiment.report_smolyak_same_budget_accuracy \
  --platform gpu \
  --dimensions 1,2,3,4,5,6,7,8,9,10,11,12 \
  --levels 5,6 \
  --dtypes float64 \
  --family gaussian \
  --mc-seeds 8 \
  --warm-repeats 3 \
  --case-timeout-seconds 180 \
  --output-dir /tmp/smolyak_integrator_loops/loop1_same_budget_l5_l6
```

である。ここでは `d=1..12` を 1 次元ずつ連続に上げ、各次元で `level=5` と `level=6` の両方を評価した。実際に計算したのは 24 ケースで、各ケースにつき

1. Smolyak の Gaussian 積分
2. その same-budget Monte Carlo 8 本
3. error / runtime 集約

を行っている。つまり総数としては「24 個の sparse-grid integral」と「24 * 8 = 192 個の Monte Carlo baseline」を回したことになる。

### Loop 2: same-budget 図の再生成

Loop 1 の cached case JSON を再利用して、図と Markdown report を再生成した。ここで新しい計算を足したわけではないが、軸名と読み方を SVG と本文に反映するために report generator を再実行した。計算そのものより「何を計算したかが report から読めるか」を整えるループである。

### Loop 3: GPU sweep at `level=6`

次に、

```bash
python3 -m experiments.smolyak_experiment.report_smolyak_gpu_sweep \
  --platform gpu \
  --dimensions 1,2,3,4,5,6,7,8,9,10,11,12 \
  --level 6 \
  --dtype float64 \
  --family gaussian \
  --vmap-batch-size 16 \
  --monitor-min-duration-ms 600 \
  --output-dir /tmp/smolyak_integrator_loops/loop2_gpu_sweep_l6
```

を回した。この run は誤差比較ではなく throughput 計測で、各次元ごとに

1. `single`: `integrate(f_alpha)` を 1 本
2. `batch`: 16 本の `alpha` をまとめた `vmap(integrate(f_alpha))`

を測った。`alpha` は `0.6..1.0` の 16 点である。さらに `nvidia-smi` monitor を並行起動し、util / pstate を記録した。したがってこの loop で回している数学計算は「同じ Gaussian 積分を複数の `alpha` で batched に解く」ことである。

### Loop 4: materialization frontier at `level=6`

ここでは report script ではなく inline Python で、`d=11..16, level=6` の integrator 構築だけを行い、

1. `materialization_mode`
2. `num_evaluation_points`
3. `storage_mb`
4. `vectorized_ndim`

を抜いた。つまりこの loop は積分値よりも「この case はどの実装経路に落ちるか」を見る診断 run である。`d=13` で `points -> indexed` に切り替わることをここで確認した。

### Loop 5: materialization frontier at `level=7`

Loop 4 と同じ診断を `d=8..11, level=7` で行った。`level=7` では `d=10` から `indexed` に入る。ここで確認したのは「より高 level では切替点がどこまで前倒しされるか」である。

### Loop 6: forced-mode benchmark at `(d=13, l=6)`

この loop では同じ Gaussian 積分 `d=13, level=6, float64` を、`points`, `indexed`, `batched` の 3 経路に強制して比較した。比較したのは

1. integrator 構築時間
2. first call 時間
3. warm runtime
4. storage size

である。ここでは数学問題は 1 つで、違うのは execution path だけなので、「新経路は何を犠牲にして何を得たか」を直接読める。

### Loop 7: forced-mode benchmark at `(d=10, l=7)`

Loop 6 と同じ強制比較を `d=10, level=7, float64` でも行った。`level=7` は full-points がさらに重くなる帯域なので、`indexed` が speed でも勝てるかを見た。結果としてこのケースでは `indexed` のほうが `points` より速かった。

### Loop 8: Monte Carlo comparison on `indexed d=13, l=6`

次に、

```bash
python3 -m experiments.smolyak_experiment.compare_smolyak_vs_mc \
  --platform gpu \
  --dimension 13 \
  --level 6 \
  --dtype float64 \
  --family gaussian \
  --warm-repeats 3 \
  --mc-seeds 8 \
  --max-samples 8000000 \
  --output-dir /tmp/smolyak_integrator_loops/loop3_compare_d13_l6
```

を回した。ここでは `indexed` に切り替わった直後の case を使い、

1. Smolyak 1 本
2. same-budget Monte Carlo 8 本
3. Monte Carlo の geometric search `1,4,16,...,4194304`

を実行した。報告の主表は same-budget を使っているが、探索履歴も JSON に残してある。

### Loop 9: Monte Carlo comparison on `indexed d=10, l=7`

Loop 8 と同じ比較を `d=10, level=7, float64` に対して行った。ここは `indexed` に入ってからも精度優位が強く残る代表例で、Monte Carlo の same-budget 誤差 `5.20e-05` に対し Smolyak は `3.69e-07` だった。

### Loop 10: dtype tradeoff on `d=10, l=7`

最後に `float32` を足し、

```bash
python3 -m experiments.smolyak_experiment.compare_smolyak_vs_mc \
  --platform gpu \
  --dimension 10 \
  --level 7 \
  --dtype float32 \
  --family gaussian \
  --warm-repeats 3 \
  --mc-seeds 8 \
  --max-samples 12000000 \
  --output-dir /tmp/smolyak_integrator_loops/loop6_compare_d10_l7_float32
```

を回した。ここで見たかったのは「float64 で `indexed` に入る case が、float32 では `points` に残ると何が起きるか」である。つまり dtype 比較であると同時に、byte budget と execution path の coupling を見る loop でもある。

## 図表

### 1. Same-budget の評価点数

![Smolyak evaluation points](assets/smolyak_integrator_impl_report_20260331/points_l5_l6.svg)

読み方: 横軸は dimension `d`、縦軸は Smolyak の評価点数で対数軸。上に行くほど積分器コストが急増する。  
軸の選択: 点数は数桁動くので対数軸が自然である。

### 2. Same-budget の精度優位

![Same-budget error ratio](assets/smolyak_integrator_impl_report_20260331/same_budget_error_ratio_l5_l6.svg)

読み方: 横軸は dimension `d`、縦軸は `MC error / Smolyak error` の対数軸。`1` より上なら Smolyak が高精度。  
軸の選択: 精度優位が 1 桁未満から 100 倍超まで動くので対数軸を使う。

### 3. Same-budget の速度差

![Same-budget runtime ratio](assets/smolyak_integrator_impl_report_20260331/same_budget_runtime_ratio_log_l5_l6.svg)

読み方: 横軸は dimension `d`、縦軸は `MC warm runtime / Smolyak warm runtime` の対数軸。`1` より上なら Smolyak が速い。  
軸の選択: 今回は多くのケースで比が `0.3..1.0` の間にあり、対数軸にして勝ち負けの距離感を見やすくした。

### 4. GPU throughput at level 6

![GPU throughput level 6](assets/smolyak_integrator_impl_report_20260331/gpu_throughput_l6.svg)

読み方: 横軸は dimension `d`、縦軸は integrals per second の対数軸。高いほど GPU 実効性能が出ている。  
軸の選択: throughput 差が約 1 桁あるので対数軸を使う。

![GPU utilization level 6](assets/smolyak_integrator_impl_report_20260331/gpu_util_l6.svg)

読み方: 横軸は dimension `d`、縦軸は平均 GPU 利用率 [%]。高いほど workload が GPU を埋められている。  
軸の選択: 利用率は `0..100` の bounded quantity なので線形軸を使う。

![GPU pstate level 6](assets/smolyak_integrator_impl_report_20260331/pstate_l6.svg)

読み方: 横軸は dimension `d`、縦軸は最小観測 Pstate 数値。小さいほど高性能状態に入りやすい。  
軸の選択: Pstate は順位尺度なので線形軸のまま扱う。

### 5. GPU throughput at level 7

![GPU throughput level 7](assets/smolyak_integrator_impl_report_20260331/gpu_throughput_l7.svg)

読み方: 横軸は dimension `d`、縦軸は integrals per second の対数軸。`d=10,11` では `indexed` へ切り替わったあとの throughput を見ている。  
軸の選択: level 7 では throughput の落差が大きく、対数軸が妥当である。

![GPU utilization level 7](assets/smolyak_integrator_impl_report_20260331/gpu_util_l7.svg)

読み方: 横軸は dimension `d`、縦軸は平均 GPU 利用率 [%]。`indexed` 帯に入っても利用率自体は高水準を保てているかを見る。  
軸の選択: bounded quantity のため線形軸。

![GPU pstate level 7](assets/smolyak_integrator_impl_report_20260331/pstate_l7.svg)

読み方: 横軸は dimension `d`、縦軸は最小観測 Pstate 数値。`P2` 近傍なら実運用上はかなり良い。  
軸の選択: 順位尺度なので線形軸。

## 主要な定量結果

### Same-budget summary

| Level | Cases | Smolyak more accurate | Smolyak faster | Median `MC err / Smolyak err` | Median `MC ms / Smolyak ms` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `5` | `12` | `12` | `3` | `191.79` | `0.832` |
| `6` | `12` | `12` | `0` | `611.40` | `0.813` |

この sweep では `d=1..12` を 1 次元ずつ連続で上げたが、少なくともこの範囲では Smolyak は same-budget 精度で全勝した。特に `level=6` は worst case の `d=12` でも誤差比 `3.28` を維持している。つまり「精度で勝てる帯域」は確かに存在し、しかも今回の改良でそこを高次元側へ押し広げる下地が整った。

ただし speed はまだ別の話である。中央値の `MC ms / Smolyak ms` はどちらの level でも 1 未満で、warm runtime は Monte Carlo が速い。したがって今回の改善でまず勝ったのは accuracy frontier であり、speed frontier は次の課題として残っている。

### GPU sweep summary

| Sweep | Case | Mode | Batch throughput [integrals/s] | Speedup vs single | Avg GPU util [%] | Peak GPU util [%] | Dominant Pstate |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| `level=6` | `d=6` | `points` | `12089.06` | `17.03x` | `37.3` | `61` | `P8` |
| `level=6` | `d=12` | `points` | `6188.67` | `11.22x` | `38.3` | `68` | `P2` |
| `level=7` | `d=9` | `points` | `4434.66` | `8.89x` | `45.0` | `65` | `P2` |
| `level=7` | `d=10` | `indexed` | `2762.91` | `18.98x` | `50.5` | `88` | `P2` |
| `level=7` | `d=11` | `indexed` | `544.49` | `14.66x` | `59.0` | `59` | `P3` |

ここで重要なのは、`indexed` に切り替わったあとも GPU が死んでいないことである。`level=7, d=10` では `avg util=50.5%`, `peak=88%`, `P2` を維持し、single 比 speedup は `18.98x` まで出ている。つまり `indexed` は単なる safety path ではなく、GPU 上でも十分実用的な execution path になっている。

### Materialization frontier

| Case | Mode | Num points | Storage [MiB] | Comment |
| --- | --- | ---: | ---: | --- |
| `d=12, l=6` | `points` | `659716` | `69.463` | まだ full-points が収まる最後の帯域 |
| `d=13, l=6` | `indexed` | `961988` | `62.086` | ここで `points -> indexed` に切り替わる |
| `d=16, l=6` | `indexed` | `2582717` | `218.191` | `indexed` でまだ保持できる |
| `d=9, l=7` | `points` | `916351` | `71.653` | level 7 の points 最後の帯域 |
| `d=10, l=7` | `indexed` | `1621012` | `77.404` | ここで level 7 の切替が始まる |
| `d=11, l=7` | `indexed` | `2732962` | `141.261` | `indexed` でさらに先へ進める |

切替点はほぼ期待どおりで、`level=6` は `d=13`、`level=7` は `d=10` で `indexed` に入った。`points` のまま閾値を 256 MiB に置いていた以前の設計だと、これらのケースは巨大な point cloud をそのまま持つ方向へ進み、host-pinned memory 警告を強く引き起こしていた。今回の設計では storage が一度下がり、frontier を一段延命できている。

### Forced-mode microbenchmark

| Case | Mode | Storage [MiB] | Warm runtime [ms] | Reading |
| --- | --- | ---: | ---: | --- |
| `d=13, l=6` | `points` | `109.792` | `2.242` | 最速だが memory は重い |
| `d=13, l=6` | `indexed` | `62.086` | `2.567` | storage を `43%` 削減、速度低下は小さい |
| `d=13, l=6` | `batched` | `7.040` | `38327.246` | memory は軽いが実用外 |
| `d=10, l=7` | `points` | `139.241` | `26.677` | full-points はここで急に重くなる |
| `d=10, l=7` | `indexed` | `77.404` | `22.644` | storage 半減かつ runtime も改善 |
| `d=10, l=7` | `batched` | `3.200` | `43244.560` | こちらも実用外 |

この表が今回の改善の核心である。`indexed` は `batched` への落下を遅らせるための中間経路だが、`d=10, l=7` では `points` より速い。つまり「memory を削ると遅くなる」だけではなく、「巨大 point cloud の読み出しコストが支配し始める帯域では、むしろ indexed のほうが良い」ことが見えた。

### Monte Carlo comparison on indexed cases

| Case | Dtype | Mode | Smolyak abs err | MC abs err at same budget | `MC/Smolyak` err ratio | Smolyak warm [ms] | MC warm [ms] |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `d=13, l=6` | `float64` | `indexed` | `3.09e-05` | `6.66e-05` | `2.15` | `2.538` | `1.354` |
| `d=10, l=7` | `float64` | `indexed` | `3.69e-07` | `5.20e-05` | `141.18` | `2.630` | `1.361` |
| `d=10, l=7` | `float32` | `points` | `3.04e-06` | `5.20e-05` | `17.14` | `3.506` | `1.369` |

`d=13, l=6` では `indexed` に入ったあとも same-budget 精度優位は残り、誤差比は `2.15` だった。`d=10, l=7` はさらに極端で、`float64 indexed` の Smolyak 誤差は `3.69e-07`、Monte Carlo は `5.20e-05` なので `141x` の精度優位が出ている。ここはまさに「高精度 high-level sparse quadrature が効く帯域」である。

同時に dtype tradeoff も見える。`d=10, l=7` では `float32` でも Monte Carlo より十分高精度だが、`float64 indexed` にすると誤差がさらに約 `8.24x` 良くなる。しかも今回の threshold 設計では `float32` は byte 数の都合で `points` に残り、`float64` は `indexed` へ移るので、runtime は `float64` のほうがむしろ良かった。この挙動は面白いが、threshold 依存でもあるので一般化しすぎるべきではない。

## 批判的レビュー

1. 依然として Monte Carlo は速い。same-budget `d=1..12` 全勝は良いニュースだが、speed では全く別で、中央値の runtime 比は常に 1 未満だった。
2. `indexed` は frontier を押し上げたが、host-pinned memory 警告はまだ残る。`d=15,16` では warning が大量に出ており、完全解決ではない。
3. `indexed` の storage は `d=16, l=6` で `218 MiB` まで伸びる。つまり `points` をやめても、さらに先へ行くにはもう一段の圧縮が要る。
4. `batched` fallback は correctness と最低限の到達性を守るが、`4e4 ms` 級では frontier として実用外である。ここへ落ちる前に止める設計が望ましい。
5. GPU 利用率は improved したが、ケース依存が大きい。`level=7, d=11` は `avg util=59%` でも throughput が `544.49` integrals/s まで落ちており、計算量そのものが急に重くなる帯域では util だけで楽観できない。
6. 現状の report は Gaussian family 中心である。smooth integrand では Smolyak に好意的なので、今後は mixed-smooth だが局所的に厳しい関数群でも同じ傾向が出るかを見る必要がある。

## 結論

今回の改善で到達したことは 3 つある。

- `indexed materialization` により、高次元・高 level で `batched` に落ちる前の有効帯域を広げた。
- その帯域では GPU 利用率と batch speedup を保ったまま same-budget 精度優位を維持できた。
- ただし warm runtime では依然として Monte Carlo が優位であり、「高精度化の成功」と「高速化の成功」は切り分けて評価すべきだと確認できた。

根拠図 1: same-budget 精度優位

![Conclusion: same-budget error ratio](assets/smolyak_integrator_impl_report_20260331/same_budget_error_ratio_l5_l6.svg)

読み方: `1` より上は Smolyak の精度優位で、`level=5,6` は `d=1..12` で一貫して上にいる。

根拠図 2: indexed 帯での GPU 実行性

![Conclusion: GPU throughput level 7](assets/smolyak_integrator_impl_report_20260331/gpu_throughput_l7.svg)

読み方: `d=10,11` は `indexed` 帯だが、throughput がゼロに潰れず、batch speedup も維持している。

次の一手は明確である。`indexed` のさらに先として、weight accumulation の圧縮と数値安定化を同時に進め、`d>=16` の帯域で `indexed` すら重くなる箇所を詰めるべきだ。候補は difference-rule の加算順序改善、index storage のさらなる圧縮、prefix/suffix grouping の再利用である。
