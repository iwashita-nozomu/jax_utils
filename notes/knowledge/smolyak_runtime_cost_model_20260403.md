# Smolyak Runtime Cost Model

## Purpose

`python/jax_util/functional/smolyak.py` の loop 内で何が時間を食っているかを、まずは

- コード構造
- HLO / buffer assignment
- 実測 wall time
- ハードウェア上限

から切り分けるためのメモ。

ここでは「どの phase が理論上どれくらい重いか」を先に見積もり、その後に

- どこまでを JAX 側の wall time で済ませられるか
- どこから先で GPU kernel 単位の profiler が必要になるか

を整理する。

## Current Loop Anatomy

現行の主要経路は [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py) の次の関数にまとまっている。

- 1D rule 生成: [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L85)
- rule storage 構築: [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L346)
- point / weight decode: [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L471)
- term update: [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L519)
- main integral loop: [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L572)

main loop の phase は概ね次の順。

1. term の current level から current rule offsets / lengths を引く。
2. term 内の point 数を prefix / suffix に分ける。
3. prefix chunk ごとに `_decode_points_and_weights(...)` を回す。
4. 必要なら suffix 側を broadcast して全点 block を作る。
5. `jax.vmap(f)` で chunk 上の関数値を作る。
6. `tensordot` で重み付き縮約する。
7. `_next_term_extra_levels(...)` で次 term へ進む。

## What Can Be Estimated Without Fine GPU Profiling

現状の問題は、いきなり Nsight に行かなくてもかなり切り分けられる。

- `integrator_init_seconds`
  - 1D rule 構築と metadata 構築の重さを見る。
- `warm_runtime_ms`
  - 実行 phase 全体の重さを見る。
- HLO memory analysis / buffer assignment
  - どの shape の temp が支配的かを見る。
- `device_memory_stats`
  - persistent storage と peak temp を分けて見る。

実際、outer `vmap(1000)` 問題は HLO buffer assignment を見るだけで、

- plain `vmap`: `f32[1000,16384]` temp
- wrapper batching: tile 化された小さい scratch

という差まで特定できた。これは [Smolyak Vmap XLA Memory](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/notes/knowledge/smolyak_vmap_xla_memory.md) に記録済み。

したがって、一次診断としては

- wall time
- HLO memory analysis
- buffer assignment
- hardware bandwidth 上限

で十分。

GPU kernel 単位の細かい profiler が必要になるのは、「関数評価」「gather」「整数 decode」「縮約」のどれが fused kernel 内で支配的かをさらに分けたい段階から。

## Cost Terms

以下では `d = dimension`, `L = level`, `q = chunk_size`, `B = outer batch size`, `m = output width` と書く。

### 1D Rule Initialization

Clenshaw-Curtis の full rule 長は

- `m_1 = 1`
- `m_l = 2^(l-1) + 1` for `l >= 2`

で、`prepared_level = L` まで全部抱える現在の storage 長は

`M(L) = sum_{l=1}^L m_l = 2^L + L - 2`

になる。

これは [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L121) と [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L346) の実装と整合する。

重み生成は DCT-I を FFT で実装しているので、level `l` の full rule 生成コストは

`W_rule(l) = Theta(n_l log n_l), n_l = 2^(l-1)`

で見てよい。[smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L85)

さらに `_difference_rule_device(l)` は `level-1` の full rule 重みも再計算しているので、全 init work は概ね

`W_init(L) = sum_{l=1}^L Theta(n_l log n_l) + sum_{l=2}^L Theta(n_{l-1} log n_{l-1}))`

`= Theta(L * 2^L)`

と見積もれる。

重要なのは、ここが `term` の数ではなく `1D rule` の prefix 全体に比例する点。高 level 1D では、この phase だけで数十秒に到達しても不思議ではない。

### Term Enumeration And Update

term 数は isotropic Smolyak なら

`T(d, L) = C(d + L - 1, d)`

で、weighted の場合は budget 制約付きの組合せ数になる。[smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L356)

current term から next term への更新 `_next_term_extra_levels(...)` は、現行コードでは `dimension` 長の `fori_loop` なので

`W_next_term = Theta(d)`

である。[smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L519)

1D ではほぼ無視できるが、高次元で integrand が軽いとこの `O(T d)` が前景化する。

### Point Decode

`_decode_points_and_weights(...)` は local point index から mixed-radix decode をして、各 axis の

- `div`
- `rem`
- `take(rule_nodes, ...)`
- `take(rule_weights, ...)`

を `scan` で回している。[smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L471)

axis 数を `a`、decode する point 数を `p` とすると、理論 work は

`W_decode(a, p) = Theta(a p)`

でよい。実際には

- integer div / rem
- gather
- weight multiply

の混成になるので、roofline 的には単純 FLOP というより「irregular memory + integer ops」側。

### Prefix / Suffix Expansion

現行の既定値は `batched_suffix_ndim = 0` なので、通常 path では suffix 展開は起きない。[smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L705)

ただし `suffix_ndim > 0` の path では

- `prefix_points_grid`
- `suffix_points_grid`
- `points_grid`
- `weight_grid`

を作るので、prefix chunk size を `C`, suffix 点数を `S` とすると、

- point tensor size: `Theta(d C S)`
- weight tensor size: `Theta(C S)`

になる。[smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L646)

この path は現在の baseline では主犯ではないが、suffix batching を有効化したときの temp 見積もりには必須。

### Function Evaluation And Reduction

suffix なしの既定 path では、prefix chunk ごとに

- `values = vmap(f)(points)` with shape roughly `(m, q)`
- `tensordot(values, weights)` with cost `Theta(m q)`

となる。[smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L641)

したがって、関数評価コストを `F_f(q)` と書けば 1 chunk あたりのコストは

`W_chunk ~= W_decode(d, q) + F_f(q) + Theta(m q)`

で近似できる。

cheap integrand では `decode + reduction` が支配し、`exp` や `dot` を含むような integrand では `F_f(q)` が支配する。

### Outer Problem Batch

plain `vmap(lambda f_i: integrate(f_i, integrator))` では、概ね `B x q` の中間が立つ。

したがって temp の第一近似は

`Mem_temp_plain ~= Theta(B q m * sizeof(dtype))`

になる。

wrapper batching を入れて tile size を `b` に落とすと

`Mem_temp_tiled ~= Theta(b q m * sizeof(dtype))`

まで下がる。

これは current HLO と一致していて、plain `vmap` では `f32[1000,16384]` が主犯だったのに対し、wrapper batching では tile 化された scratch に変わった。

## Concrete Estimates On Current Data

### Case A: 1D Level 25 Float64 Vmap

baseline の実測値:

- case: `dimension=1`, `level=25`, `dtype=float64`, `execution_variant=vmap`
- `num_points = 33,554,455`
- `chunk_size = 16,384`
- `vmap_batch_size = 1000`
- `storage_bytes = 536,871,688`
- `warm_runtime_ms = 18,690.53`
- `integrator_init_seconds = 38.76`
- `peak_bytes_in_use = 3,196,130,560`

このとき

- chunk 数: `ceil(33,554,455 / 16,384) = 2,049`
- 1 chunk の dense value matrix 上限:
  - `1000 * 16,384 * 8 = 131,072,000 bytes`
  - 約 `125 MiB`
- 全 chunk に対する streamed values の総量:
  - `2,049 * 131,072,000 = 268,566,528,000 bytes`
  - 約 `250.12 GiB`

GPU は RTX 4000 SFF Ada で、公式の memory bandwidth は `280 GB/s`。[NVIDIA datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-rtx-4000-sff-ada-datasheet-2616456-web.pdf)

よって、この `250.12 GiB` を一度だけ綺麗に stream できたとしても下限は

`268,566,528,000 / 280e9 ~= 0.96 s`

である。

実測は `18.69 s` なので、現在の runtime は単純な「1 回の dense streaming」では説明できない。ここから次のことが言える。

- pure memory bandwidth だけが主犯ではない
- gather / div-rem / loop overhead / repeated kernel launch / cheap ではない integrand evaluation がかなり効いている
- fine-grained profiler は「その 18.69s の中でどれが支配的か」を詰める段階で必要になる

### Case B: Level 22-25 Vmap Frontier

現時点の `dimension=1`, `execution_variant=vmap` の抜粋:

| level | dtype | init sec | warm ms | num_points | storage bytes |
|---|---:|---:|---:|---:|---:|
| 22 | bfloat16 | 32.61 | 166.90 | 4,194,324 | 16,777,656 |
| 23 | bfloat16 | 35.86 | 257.75 | 8,388,629 | 33,554,892 |
| 24 | bfloat16 | 36.64 | 1331.19 | 16,777,238 | 67,109,344 |
| 25 | bfloat16 | 50.61 | 720.51 | 33,554,455 | 134,218,228 |
| 22 | float32 | 32.75 | 128.73 | 4,194,324 | 33,554,952 |
| 23 | float32 | 35.67 | 219.32 | 8,388,629 | 67,109,408 |
| 24 | float32 | 36.65 | 1578.04 | 16,777,238 | 134,218,296 |
| 25 | float32 | 37.99 | 929.43 | 33,554,455 | 268,436,048 |
| 22 | float64 | 38.64 | 821.39 | 4,194,324 | 67,109,544 |
| 23 | float64 | 34.25 | 1468.48 | 8,388,629 | 134,218,440 |
| 24 | float64 | 36.42 | 2950.04 | 16,777,238 | 268,436,200 |
| 25 | float64 | 38.76 | 18690.53 | 33,554,455 | 536,871,688 |

ここから見えること:

- `compile_ms` は cliff ではない。jump の主因ではない。
- `num_points` と `storage_bytes` は level ごとにほぼ倍化する。
- `warm_runtime_ms` の jump は、この point 増加と batch 化の積の影響を強く受けている。
- `float64` は value matrix と reduction cost が最も厳しく出る。

## When Fine GPU Runtime Measurement Becomes Necessary

### まだ不要なもの

次の判断には GPU kernel profiler はまだ不要。

- OOM が temp 由来か persistent storage 由来か
- plain `vmap` が `B x q` temp を作っているか
- wrapper batching が temp を減らしたか
- `compile cliff` ではなく `num_points` の倍化が主因か

これは

- current code structure
- HLO memory analysis
- buffer assignment
- wall time

だけで分かる。

### 必要になるもの

次を詰める段階では GPU の細かい runtime が必要になる。

- `_decode_points_and_weights` の gather / div-rem が fused kernel 内で何割か
- `vmap(f)` と `tensordot` のどちらが重いか
- tile 化後の kernel launch overhead がどれくらいか
- `while_loop` / `fori_loop` に起因する occupancy 悪化があるか

この段階での推奨スタックは次。

1. `block_until_ready` 付き wall time  
   参考: [Benchmarking JAX code](https://docs.jax.dev/en/latest/benchmarking.html)
2. `jax.profiler.TraceAnnotation` を phase ごとに埋める  
   参考: [Profiling JAX programs](https://docs.jax.dev/en/latest/profiling.html)
3. XProf / Perfetto で trace を見る  
   参考: [Profiling JAX programs](https://docs.jax.dev/en/latest/profiling.html)
4. `save_device_memory_profile()` で memory snapshot を取る  
   参考: [Device memory profiling](https://docs.jax.dev/en/latest/device_memory_profiling.html)
5. Nsight Systems で kernel / copy timeline を見る  
   参考: [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/2022.3/pdf/UserGuide.pdf)
6. Nsight Compute で kernel 単位の roofline / memory workload を見る  
   参考: [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)

## Recommended Next Instrumentation

コード改善前に、まず次の phase timing を追加するとよい。

- `term_setup_seconds`
- `suffix_prepare_seconds`
- `prefix_decode_seconds`
- `integrand_eval_seconds`
- `reduction_seconds`
- `term_update_seconds`

JAX の world では完全な source-level 分解は難しいので、最初は phase を `TraceAnnotation` で囲って XProf で見るのが一番手堅い。

特に current code では

- `_decode_points_and_weights(...)`
- `vmap(f)`
- `tensordot`
- `_next_term_extra_levels(...)`

の 4 つを別 phase にするだけで、次の改善方向はかなり決まるはず。

## Practical Takeaways

- いまの bottleneck は `OOM` より `time frontier` に移っている。
- `1D level25 float64 vmap` はメモリ的には通るが、18.7 秒かかる。
- したがって次の課題は「どこで時間を浪費しているか」の切り分け。
- 理論見積もりだけでも、`1D rule init`, `decode`, `f eval`, `reduction`, `outer batch temp` のどれが怪しいかはかなり絞れる。
- その上で kernel 内訳を詰めるときだけ、Nsight / XProf が必要になる。

## References

- See also: [Smolyak Modification Methods From Literature](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/notes/knowledge/smolyak_modification_methods_from_literature_20260403.md)
- JAX Benchmarking: <https://docs.jax.dev/en/latest/benchmarking.html>
- JAX Profiling: <https://docs.jax.dev/en/latest/profiling.html>
- JAX Device Memory Profiling: <https://docs.jax.dev/en/latest/device_memory_profiling.html>
- JAX `lax.map`: <https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html>
- JAX `custom_vmap`: <https://docs.jax.dev/en/latest/_autosummary/jax.custom_batching.custom_vmap.html>
- OpenXLA Flags Guidance: <https://openxla.org/xla/flags_guidance>
- Bungartz, Griebel, *Sparse grids*, Acta Numerica 13 (2004), DOI: <https://doi.org/10.1017/S0962492904000182>
- Gerstner, Griebel, *Dimension-Adaptive Tensor-Product Quadrature*, Computing 71(1), 65–87 (2003), DOI: <https://doi.org/10.1007/s00607-003-0015-5>
- Bungartz, Dirnstorfer, *Multivariate Quadrature on Adaptive Sparse Grids*, Computing 71(1), 89–114 (2003), DOI: <https://doi.org/10.1007/s00607-003-0016-4>
- Tasmanian Math Manual: <https://mkstoyanov.github.io/tasmanian_aux_files/docs/TasmanianMathManual.pdf>
- Tasmanian docs: <https://ornl.github.io/TASMANIAN/rolling/group__TasmanianSG.html>
- SG++ docs: <https://sgpp.sparsegrids.org/docs/>
- *An Efficient and Fast Sparse Grid Algorithm for High-Dimensional Numerical Integration* (MDI-SG): <https://www.mdpi.com/2227-7390/11/19/4191>
- NVIDIA RTX 4000 SFF Ada datasheet: <https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/proviz-rtx-4000-sff-ada-datasheet-2616456-web.pdf>
- NVIDIA Nsight Systems User Guide: <https://docs.nvidia.com/nsight-systems/2022.3/pdf/UserGuide.pdf>
- NVIDIA Nsight Compute Profiling Guide: <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
