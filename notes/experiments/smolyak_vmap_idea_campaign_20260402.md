# Smolyak Vmap Idea Campaign

Source:

- [Smolyak Vmap XLA Memory](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/notes/knowledge/smolyak_vmap_xla_memory.md)
- [smolyak_single_vs_batch_20260402.memory.json](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_hlo/results/smolyak_single_vs_batch_20260402.memory.json)
- [smolyak_single_vs_batch_20260402.summary.json](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_hlo/results/smolyak_single_vs_batch_20260402.summary.json)

Branch:

- `work/smolyak-integrator-opt-20260328`

Scope:

- `vmap(integrate(f_i))` を正本 API として扱う。
- mode 追加や mode 切替で逃げない。
- 各案を 1 件ずつ試し、だめなら理由を残す。
- 実験条件、観測値、失敗理由をこの note に蓄積する。

Current Baseline:

- `chunk_size=16384`
- `batched_suffix_ndim=0`
- `single temp_size_in_bytes = 199248`
- `batch temp_size_in_bytes = 65801040`
- 主犯バッファは `f32[1000,16384]`
- したがって、主戦場は `batch_size x chunk_size` の一時領域

Priority Policy:

- `chunk_size` tuning は補助計測として扱う。
- main はコード改善であり、特に
  - gather と reduction の間で巨大中間を作らない
  - batch 軸を保持したまま streaming / tiling する
  - term / point decode と accumulation の実装形を変える
  を優先する。
- parameter tuning は、コード改善前後の比較を成立させるためにだけ使う。

## Backlog 100

Status:

- `pending`: 未着手
- `running`: 進行中
- `done`: 試行済み
- `rejected`: 試した結果、主戦場ではなかった

### Batch x Chunk Shape

1. `done` `chunk_size` を runner から露出して一次感度を取る。
2. `pending` `chunk_size=1024/2048/4096/8192/16384` の HLO temp を系統比較する。
3. `pending` `chunk_size` を `batch_size` から自動決定する heuristic を試す。
4. `pending` `chunk_size` を `dimension` と `level` も含めて自動決定する heuristic を試す。
5. `pending` `target_temp_budget_bytes` から逆算する `chunk_size` 選択を試す。
6. `pending` outer `vmap` 前提で `chunk_size` の compile time 感度を測る。
7. `pending` outer `vmap` 前提で `chunk_size` の warm runtime 感度を測る。
8. `pending` outer `vmap` 前提で `chunk_size` の peak memory 感度を測る。
9. `pending` `chunk_size` を power-of-two 以外でも振り、XLA kernel の癖を確認する。
10. `pending` 同じ `chunk_size` で `batch_size=1,8,32,128,512,1000` の temp 成長率を測る。

### Batch Evaluation Shape

11. `pending` outer `vmap` の HLO dump を runner から自動生成する。
12. `pending` outer `vmap` の buffer-assignment dump を runner から自動生成する。
13. `pending` outer `vmap` の memory-usage-report dump を runner から自動生成する。
14. `pending` `f(point)` の出力 shape を scalar へさらに寄せられないか調べる。
15. `pending` `jnp.asarray([value])` を避けた integrand で HLO 差分を取る。
16. `pending` benchmark integrand を `vectorized` 実装にして HLO 差分を取る。
17. `pending` `tensordot` を `matmul` 相当へ寄せて HLO 差分を取る。
18. `rejected` `values` の軸順を変えて `dot_general` の layout 差分を取る。
19. `pending` `batch x chunk` を `chunk x batch` に寄せて temp 差分を取る。
20. `pending` 結果 accumulation を chunk 単位の `scan` へ寄せて temp 差分を取る。

### Gather and Decode

21. `pending` `_decode_points_and_weights` の gather を 1 回に寄せる。
22. `pending` points と weights を別 gather せず fused できないか調べる。
23. `pending` `axis_offsets` と `axis_lengths` の dtype を `int32` に固定して差を見る。
24. `pending` `local_point_indices` の dtype を `int32` に寄せて差を見る。
25. `pending` `lax.div` と `lax.rem` の組を `divmod` 相当へ寄せる案を試す。
26. `pending` `scan` の carry 形状を軽くして HLO 差分を取る。
27. `pending` 逆順 decode を順方向 decode へ変えて HLO 差分を取る。
28. `pending` prefix 部分だけの decode kernel を単独 HLO 解析する。
29. `pending` suffix なし経路の decode をさらに specialized する。
30. `pending` gather 後すぐ multiply する kernel 形へ Pallas で寄せる。

### Reduction and Accumulation

31. `pending` `jnp.tensordot` を `jnp.sum(values * weights, axis=...)` へ変えて比較する。
32. `rejected` `einsum` へ書き換えて HLO 差分を取る。
33. `pending` accumulation dtype を制御して temp 差分を取る。
34. `pending` chunk accumulation を `lax.scan` に移して buffer reuse を促す。
35. `pending` accumulation 先を `Ref` / scratch 的に扱う経路を調べる。
36. `pending` reduction 前の `values` を transposed して GEMM friendly にする。
37. `pending` reduction と chunk loop を 1 kernel に寄せる可能性を調べる。
38. `pending` weighted sum を custom primitive 化する。
39. `pending` weighted sum を Pallas kernel 化する。
40. `pending` reduction 部だけ benchmark 化して XLA tuning 効果を切り分ける。

### Term Iteration

41. `pending` `_next_term_extra_levels` を単独 HLO 解析する。
42. `pending` `_next_term_extra_levels` の `fori_loop` carry を小さくする。
43. `pending` `generation_weights` を static 埋め込みして差を見る。
44. `pending` term iteration と prefix loop のネスト順を見直す。
45. `pending` prefix loop の chunk 更新を `while_loop` へ揃えて比較する。
46. `pending` `has_next` 判定の compare/select を減らせないか試す。
47. `pending` term update を `scan` 化して unroll しやすくできるか試す。
48. `pending` isotropic case 専用の term unranking を試す。
49. `pending` term update の `uint8` carry を `int32` へ寄せて HLO 差分を取る。
50. `pending` term update 部の Python scalar 由来定数をさらに減らす。

### 1D Rule Initialization

51. `pending` 1D rule の level cache を導入する。
52. `pending` 1D rule の prefix cache を `refine()` と共有する。
53. `pending` 1D rule 生成を level ごとの incremental build にする。
54. `pending` 1D rule 初期化を単独 HLO / profiler で解析する。
55. `pending` DCT ベース重み生成の kernel 構造を分解して測る。
56. `pending` nodes と weights を別々に遅延構築する。
57. `pending` 1D rule storage を flat concat ではなく level list へ寄せる。
58. `pending` 1D rule の device-side cache reuse を強化する。
59. `pending` high level だけ重い初期化 path を benchmark 化する。
60. `pending` 1D rule の compile cache 効率をケース間で調べる。

### XLA and JAX Tuning

61. `pending` `jax_compiler_enable_remat_pass=false` を full sweep で再検証する。
62. `pending` `xla_gpu_enable_while_loop_double_buffering=false` を full sweep で再検証する。
63. `pending` allocator tuning を `chunk_size` 変更後に再検証する。
64. `pending` `layout` control の効果を single / batch で比較する。
65. `pending` `save_device_memory_profile()` を runner に埋め込む。
66. `pending` compile 後と warm 後で memory profile を 2 点取る。
67. `pending` HLO dump を current best config で再取得する。
68. `pending` unsupported な XLA flag 群を環境別に整理する。
69. `pending` CPU HLO と GPU HLO の差分を systematic に記録する。
70. `pending` Triton GEMM fusion が効いている部分を isolated benchmark にする。

### Pallas and Custom Lowering

71. `pending` Pallas で gather + weight multiply + reduce の最小 kernel を作る。
72. `pending` Pallas kernel を benchmark integrand 専用にまず当てる。
73. `pending` Pallas kernel を一般 `f` 用に使える境界を整理する。
74. `pending` `Ref` / scratch memory で partial sum を保持する案を試す。
75. `pending` `BlockSpec` を使って chunk tile を明示化する。
76. `pending` software pipelining を有効にした kernel を試す。
77. `pending` custom primitive で gather + reduce を fused lowering する。
78. `pending` custom primitive と Pallas のコスト差を比較する。
79. `pending` Pallas 化したときの autodiff 制約を確認する。
80. `pending` Pallas / custom lowering 案の fallback を設けずに維持できるか整理する。

### Benchmark Semantics

81. `pending` benchmark integrand を scalar-return に揃えた基準形を作る。
82. `pending` benchmark integrand を `exp(dot(c, x))` 以外でも試す。
83. `pending` benchmark integrand が構造化されすぎていないか検証する。
84. `pending` benchmark 専用の specialized path を別計測として切り出す。
85. `pending` specialized path を一般 path と混ぜずに比較用だけ残す。
86. `pending` `num_accuracy_problems` を複数値で振って batch 軸感度を測る。
87. `pending` `dimension=1` だけでなく `4,8,16` でも HLO を取り直す。
88. `pending` `level` を複数値で振って temp の成長則を測る。
89. `pending` dtype ごとの temp 成長則を測る。
90. `pending` batch throughput と temp size のトレードオフ曲線を作る。

### Process and Regression

91. `pending` idea ごとの標準記録テンプレートを確立する。
92. `pending` benchmark runner に `--campaign-note` 的な記録支援を入れる。
93. `pending` best-known config を自動で note に追記する仕組みを作る。
94. `pending` failed idea の理由分類を定義する。
95. `pending` memory temp, init time, batch runtime の 3 指標で優先度を決める。
96. `pending` commit message 規約を idea id 連動にする。
97. `pending` push 先 branch の checkpoint 運用を固定する。
98. `pending` regression HLO digest を保存する。
99. `pending` regression memory digest を保存する。
100. `pending` campaign の中間総括を 10 件ごとに書く。

## Attempt Log

### Idea 001

Idea:

- `chunk_size` を runner から露出し、`batch_size x chunk_size` が本当に主戦場かを継続的に検証できるようにする。

Why:

- 現在見えている最大 temp は `f32[1000,16384]` で、`chunk_size` に比例している。
- まずはこの軸を簡単に振れることが campaign 全体の土台になる。

Implementation:

- [run_smolyak_scaling.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_scaling/run_smolyak_scaling.py) に `--chunk-size` を追加した。
- `run_config` に `chunk_size` を保存し、child 側 `initialize_smolyak_integrator(...)` へ渡すようにした。

Status:

- `done`

Result:

- 次の idea から scaling runner と HLO runner の両方で `chunk_size` を同じ意味で振れる状態になった。

Interpretation:

- これは最適化そのものではなく、以後の 99 件を回すための計測基盤整備。
- main issue はまだ未解決だが、`batch_size x chunk_size` 仮説を runner レベルで直ちに検証できるようになった。

### Idea 002

Idea:

- `values = vmap(f)(points)` の一括 materialize をやめ、`_weighted_point_sum(...)` で point ごとに streaming accumulation する。

Why:

- `batch x chunk` の巨大中間を避ければ temp buffer が減るはず、という素直な仮説。
- 計算の省略ではなく、順序だけを変える案として安全に試せる。

Implementation:

- [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py) に `_weighted_point_sum(...)` を追加。
- `suffix_ndim == 0` と `suffix_ndim > 0` の両方で、`vmap + tensordot` をやめて `lax.scan` ベース accumulation に置き換えた。

Status:

- `rejected`

Result:

- partial baseline [smolyak_scaling_gpu_20260402T143327Z_baseline_split.jsonl](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260402T143327Z_baseline_split.jsonl) の小さいケースで大幅悪化。
- 比較対象は旧 run [smolyak_scaling_gpu_20260402T093336Z_baseline_full_rerun2.jsonl](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260402T093336Z_baseline_full_rerun2.jsonl)。
- 例:
  - `vmap, d=1, l=1, float32`: `0.607 ms -> 131.394 ms` で約 `216.6x` 悪化
  - `vmap, d=1, l=2, float32`: `1.523 ms -> 276.722 ms` で約 `181.6x` 悪化
  - `single, d=1, l=2, float64`: `0.727 ms -> 220.925 ms` で約 `304.1x` 悪化

Failed Because:

- `batch x chunk` の巨大中間は減っても、point ごとの `scan` が point 並列性を潰した。
- 元の `vmap + tensordot` は XLA/Triton がかなり強く最適化していたが、その構造を壊してしまった。
- 「temp 削減」だけを見た順序変更で、GPU の並列性を大きく落としたのが敗因。

Interpretation:

- 今後は `streaming = point 逐次 scan` ではなく、
  - tile 単位の block parallel
  - gather + reduce の fused kernel
  - `batch x chunk` を smaller tiles に切る
  方向で進めるべき。
- つまり、順序変更だけでも「parallel shape を保つ」ことが必須。

### Idea 003

Idea:

- benchmark runner で `single` と `vmap` を同一 child に混ぜず、scheduler が `execution_variant=single/vmap` を別 case として流す。

Why:

- compile, warm runtime, memory checkpoint が混ざると regression が読みづらい。
- `single` と `vmap` は本質的に別 workload なので、最初から別 case として扱うほうが自然。

Implementation:

- [run_smolyak_scaling.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_scaling/run_smolyak_scaling.py) に `--execution-variants` を追加。
- case 定義へ `execution_variant` と `execution_variant_index` を追加。
- child は `single` なら単一係数だけ、`vmap` なら係数行列全体だけを測る形に変更。

Status:

- `done`

Result:

- smoke run `/tmp/smolyak_scaling_split_smoke_after_revert.json` で `single` / `vmap` の 2 case が分離して記録されることを確認。
- `single, d=1, l=1, float32`: `compile_ms ≈ 165.43`, `warm_runtime_ms ≈ 0.847`
- `vmap, d=1, l=1, float32`: `compile_ms ≈ 581.57`, `warm_runtime_ms ≈ 0.607`, `throughput ≈ 1.65e6 integrals/s`

Interpretation:

- 以後の実験では `single` と `vmap` を別 frontier として評価できる。
- `vmap` だけが壊れたのか、積分器自体が壊れたのかを切り分けやすくなった。

### Idea 004

Idea:

- public API を変えずに、[integrate.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/integrate.py) の wrapper に `custom_vmap` を付け、outer batch を内部で `lax.map(batch_size=auto_tile)` へ落とす。

Why:

- user-facing には `vmap(lambda f: integrate(f, integrator))(f_s)` を保ちたい。
- 主犯 temp は `batch_size x chunk_size` なので、problem batch だけを wrapper 層でタイル化できれば効果が大きいはず。
- Smolyak 専用実装ではなく、数値積分 wrapper 層に batching policy を置く方が今後の Monte Carlo 系とも整合する。

Implementation:

- [integrate.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/integrate.py) に
  - `_integrate_impl(...)`
  - `_integrate_batched = custom_vmap(...)`
  - public `integrate(...)` dispatcher
  を追加。
- plain direct call は従来経路を維持し、array-backed な `f` だけ batched wrapper を通す。
- batching rule は `f` batched / `integrator` unbatched のみを受ける。
- tile size は `chunk_size`、出力 bytes、device memory limit から internal heuristic で自動決定する。
- [run_smolyak_scaling.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_scaling/run_smolyak_scaling.py) も `current_integrator.integrate(...)` 直呼びから `integrate(...)` 経路に切り替えた。

Status:

- `done`

Validation:

- [test_smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/tests/functional/test_smolyak.py) に `eqx.Module` batch を使う回帰 test を追加。
- `python3 -m pytest python/tests/functional/test_smolyak.py -q` は `24 passed`。

Result:

- HLO / memory: [smolyak_integrate_custom_vmap_hlo_20260403.json](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_hlo/results/smolyak_integrate_custom_vmap_hlo_20260403.json)
- batch XLA dump:
  - [/tmp/xla_dump_smolyak_integrate_batch_20260403/module_1203.jit_compiled_batch.sm_8.9_gpu_after_optimizations-memory-usage-report.txt](/tmp/xla_dump_smolyak_integrate_batch_20260403/module_1203.jit_compiled_batch.sm_8.9_gpu_after_optimizations-memory-usage-report.txt)
  - [/tmp/xla_dump_smolyak_integrate_batch_20260403/module_1203.jit_compiled_batch.sm_8.9_gpu_after_optimizations-buffer-assignment.txt](/tmp/xla_dump_smolyak_integrate_batch_20260403/module_1203.jit_compiled_batch.sm_8.9_gpu_after_optimizations-buffer-assignment.txt)
- runtime probe: [/tmp/smolyak_integrate_custom_vmap_probe.json](/tmp/smolyak_integrate_custom_vmap_probe.json)

Observed Numbers:

- case: `GPU / d=1 / level=12 / float32 / batch=1000 / chunk_size=16384`
- `single temp_size_in_bytes = 199248`
- `batch temp_size_in_bytes = 274000`
- 旧 plain `vmap` batch temp は `65801040`
- したがって temp は約 `240x` 削減

buffer assignment の最大 temp も変わった。

- 旧: `f32[1000,16384]`
- 新: `2 x s64[16384]`, `2 x f32[16384,1]`, `2 x f32[16384]`, `f32[4,239,1]`, `f32[239,1]`

つまり `batch_size x chunk_size` の巨大中間は消えて、`tile_size x chunk_size` ベースへ変わった。

However:

- 旧 plain `vmap` throughput: 約 `161708 integrals/s`
- 新 wrapper batching throughput: 約 `36749 integrals/s`
- warm runtime は `6.184 ms -> 27.211 ms`

Failed To Fully Succeed Because:

- memory は劇的に改善したが、`lax.map` による tile ループで batch 並列性をかなり失った。
- XLA dump でも [integrate.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/integrate.py) 起点の outer `while` が増えており、tile ごとの反復制御がそのまま kernel 形に現れている。
- つまり「API 保持」と「temp 削減」には成功したが、「throughput 維持」は未達。

Interpretation:

- wrapper-level batching という abstraction の置き場所は正しい。
- ただし実装の first cut としての `lax.map(auto_tile)` は memory fix としては優秀でも、performance fix としてはまだ弱い。
- 次は
  - tile 内 kernel の構造改善
  - auto tile heuristic の改善
  - `lax.map` から `custom_vmap` / lower-level fusion への深化
  をこの wrapper 境界のまま進めるのが筋。

### Idea 018

Idea:

- `values` の軸順を `out,...` ではなく `prefix/suffix,out` 側へ寄せ、縮約を左側から掛ける。

Why:

- `dot_general` の layout が変われば、outer `vmap` 時の bufferization や generated code が改善する可能性がある。
- point 並列性は保ったまま shape だけを動かす案なので、`scan` ほど危険ではない。

Implementation:

- [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py) の `suffix_ndim == 0` で
  `jax.vmap(f, in_axes=1, out_axes=0)` と `jnp.tensordot(prefix_weights, values, axes=(0, 0))`
  を試した。
- `suffix_ndim > 0` でも同じ発想で `values` を `(prefix, suffix, out)` 方向へ寄せ、`masked_weight_grid` から左縮約する形を試した。
- 代表ケースとして `/tmp/smolyak_layout_probe.json` を取得し、比較対象は `/tmp/smolyak_tensordot_probe.json` に揃えた。

Status:

- `rejected`

Result:

- `single, d=1, l=12, float32` ではやや改善:
  - `warm_runtime_ms: 5.647 -> 4.360` で約 `1.30x` 改善
- ただし本命の `vmap, d=1, l=12, float32` は悪化:
  - `compile_ms: 1229.33 -> 1437.98` で約 `1.17x` 悪化
  - `warm_runtime_ms: 6.184 -> 7.853` で約 `1.27x` 悪化
  - `throughput: 161708 -> 127334 integrals/s` で約 `21.3%` 低下
- compiled memory も改善しない:
  - `batch temp_size_in_bytes: 65801040 -> 65800961` で実質同じ
  - `batch generated_code_size_in_bytes: 35434 -> 37746` で約 `6.5%` 増加

Failed Because:

- axis order を変えても、outer `vmap` 本番では主犯の `batch x chunk` temp は残った。
- layout 変更だけでは bufferization を動かし切れず、むしろ codegen が重くなった。
- `single` 改善より `vmap` 悪化のほうが大きく、本 campaign の目標に反する。

Interpretation:

- `single` を少し速くする layout と、outer `vmap` を速くする layout は一致しない。
- 今後は axis order の微調整より、「巨大中間をどう分割・再利用するか」に寄せるべき。

### Idea 032

Idea:

- `jnp.tensordot(...)` を `jnp.einsum(...)` へ置き換え、同じ数学のまま HLO / temp を変えられないか試す。

Why:

- contraction 記法を変えるだけなら、API も計算順序もほぼ維持したまま backend の lowering を動かせる。
- `dot_general` から別の fusion 形に落ちれば、一時バッファが減る可能性がある。

Implementation:

- [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py) の
  `jnp.tensordot(values, prefix_weights, ...)` と
  `jnp.tensordot(values, masked_weight_grid, ...)`
  を、それぞれ `jnp.einsum("...i,i->...", ...)` と `jnp.einsum("...ij,ij->...", ...)` へ置き換えた。
- 代表ケースとして `/tmp/smolyak_einsum_probe.json` を取得し、比較対象は `/tmp/smolyak_tensordot_probe.json` に揃えた。
- compiled memory は既存 baseline
  [smolyak_single_vs_batch_20260402.memory.json](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_hlo/results/smolyak_single_vs_batch_20260402.memory.json)
  と突き合わせた。

Status:

- `rejected`

Result:

- compiled temp は変わらない:
  - `single temp_size_in_bytes: 199248 -> 199248`
  - `batch temp_size_in_bytes: 65801040 -> 65801040`
- generated code size は微増:
  - `single: 21550 -> 21558`
  - `batch: 35434 -> 35442`
- runtime も悪化:
  - `single warm_runtime_ms: 5.647 -> 6.178` で約 `9.4%` 悪化
  - `vmap warm_runtime_ms: 6.184 -> 7.299` で約 `18.0%` 悪化
  - `vmap throughput: 161708 -> 137004 integrals/s` で約 `15.3%` 低下

Failed Because:

- `einsum` へ書き換えても XLA の本質的な contraction 形は変わらず、主犯 temp が残った。
- lowering 差分はほぼ generated code の微差に留まり、実行時にはわずかに不利だった。

Interpretation:

- 今の kernel は contraction 記法の選択で改善する段階ではない。
- `tensordot` は少なくとも current code では十分に強く、ここをいじる価値は低い。

### Idea 033

Idea:

- `_decode_points_and_weights(...)` の axis `scan` をやめ、mixed-radix stride から各軸 index を一括で復元する。

Why:

- 現行 decode は軸方向に `scan` を回していて、各 axis ごとに `div/rem + take` を carry 付きで実行していた。
- local point index は連続区間なので、各軸 digit は
  - `stride_j = prod(lengths[j+1:])`
  - `digit_j = (index // stride_j) % length_j`
  で直接出せる。
- これなら decode の loop 骨格を薄くでき、1D だけでなく中次元でも効く可能性がある。

Implementation:

- [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py) の `_decode_points_and_weights(...)` を差し替えた。
- `axis_count == 0` は従来どおり。
- 一般 case は suffix product から `axis_strides` を作り、
  - `local_axis_indices = (local_point_indices[None, :] // axis_strides[:, None]) % axis_lengths[:, None]`
  で digit をまとめて復元する。
- `axis_count == 1` だけは退化 case として最小の fast-path を置いた。
- その後の `points` / `axis_weights` は従来どおり gather し、weight は `prod(axis=0)` で作る。

Status:

- `done`

Validation:

- `python3 -m py_compile python/jax_util/functional/smolyak.py`
- `python3 -m pytest python/tests/functional/test_smolyak.py python/tests/functional/test_integrate.py -q`
- result: `30 passed`

Result:

- current code と checkpoint `57b406a` を同じ GPU probe で比較した。

1D / 4D:

- `single_1d_l18_f64`: `13.06 ms -> 9.89 ms`, `1.32x` speedup
- `batch_1d_l18_f64`: `156.50 ms -> 134.33 ms`, throughput `1635.74 -> 1905.69/s`, `1.17x`
- `single_4d_l4_f32`: `10.57 ms -> 7.50 ms`, `1.41x`
- `batch_4d_l4_f32`: `21.75 ms -> 17.50 ms`, throughput `11769.46 -> 14625.92/s`, `1.24x`

8D / 16D:

- `single_8d_l4_f32`: `60.85 ms -> 55.24 ms`, `1.10x`
- `batch_8d_l4_f32`: `153.74 ms -> 98.15 ms`, throughput `1665.13 -> 2608.36/s`, `1.57x`
- `single_16d_l3_f32`: `111.22 ms -> 42.99 ms`, `2.59x`
- `batch_16d_l3_f32`: `237.82 ms -> 135.35 ms`, throughput `1076.44 -> 1891.35/s`, `1.76x`

Interpretation:

- 改善は 1D 専用ではなく、少なくとも `4D`, `8D`, `16D` でも一貫して効いた。
- 特に `16D` で効きが大きいので、今回の本質は 1D fast-path より「axis scan を mixed-radix stride decode へ変えた一般化」にある。
- 次は `take(..., mode=\"clip\")` の safety overhead や integer dtype を詰めるのが自然。

### Idea 034

Idea:

- 構造的に index が範囲内なので、`jnp.take(..., mode="clip")` を direct indexing へ置き換えて bounds handling を減らす。

Why:

- `mode="clip"` は安全だが、HLO 上は余分な bounds logic を持ち込みうる。
- `rule_nodes[rule_indices]` / `rule_weights[rule_indices]` と `rule_offsets[level_indices]` / `rule_lengths[level_indices]` に寄せれば、gather の周辺を少し軽くできるかもしれない。

Implementation:

- [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py) の
  - `jnp.take(rule_nodes, rule_indices, mode="clip")`
  - `jnp.take(rule_weights, rule_indices, mode="clip")`
  - `jnp.take(rule_offsets, level_indices, mode="clip")`
  - `jnp.take(rule_lengths, level_indices, mode="clip")`
  を direct indexing に置き換えて probe を取った。

Status:

- `rejected`

Result:

- 代表 probe は current `Idea 033` 版と比較。
- `1D` ではほぼ横ばいか微悪化:
  - `single_1d_l18_f64`: `9.89 -> 10.28 ms`
  - `batch_1d_l18_f64`: `134.33 -> 143.08 ms`, throughput `1905.69 -> 1789.16/s`
- `8D` は mixed:
  - `single_8d_l4_f32`: `55.24 -> 39.93 ms`
  - `batch_8d_l4_f32`: `98.15 -> 105.62 ms`, throughput `2608.36 -> 2423.85/s`
- `16D` は明確に悪化:
  - `single_16d_l3_f32`: `42.99 -> 53.22 ms`
  - `batch_16d_l3_f32`: `135.35 -> 177.95 ms`, throughput `1891.35 -> 1438.59/s`

Failed Because:

- `clip` を外しても general に速くはならず、むしろ中高次元 batch で不安定に悪化した。
- XLA の gather lowering と direct indexing lowering が常に有利になるわけではなく、このコードでは `mode="clip"` 由来の保険コストは主犯ではなかった。
- 良いケースと悪いケースが混じり、general improvement として採れない。

Interpretation:

- gather の本質的なコストは bounds mode より decode / reduction 全体の構造にある。
- current code では safety gather を維持し、次は integer dtype や term traversal のような、もっと大きい構造側を試すべき。

### Idea 035

Idea:

- term ごとに不変な `axis_strides` を `_decode_points_and_weights(...)` の外へ hoist し、prefix chunk ごとの `cumprod` を避ける。

Why:

- current decode は毎 chunk ごとに `axis_lengths` から stride を組み直している。
- これは term 内では不変なので、`prefix_strides` / `suffix_strides` を term body で 1 回だけ作れば中次元以上で効くはず、という仮説。

Implementation:

- `_axis_strides(...)` と `_decode_points_and_weights_with_strides(...)` を追加。
- [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py) の term body で
  - `prefix_strides = _axis_strides(prefix_lengths)`
  - `suffix_strides = _axis_strides(suffix_lengths)`
  を作り、decode helper に渡す形へ変えた。

Status:

- `rejected`

Result:

- `Idea 033` 版との比較で、4D はほぼ横ばいだが 8D / 16D が悪化。
- 代表値:
  - `single_4d_l4_f32`: `7.50 -> 7.74 ms`
  - `batch_4d_l4_f32`: `17.50 -> 17.61 ms`, throughput `14625.92 -> 14535.98/s`
  - `single_8d_l4_f32`: `55.24 -> 43.26 ms` と一見改善したが
  - `batch_8d_l4_f32`: `98.15 -> 130.73 ms`, throughput `2608.36 -> 1958.21/s`
  - `single_16d_l3_f32`: `42.99 -> 60.48 ms`
  - `batch_16d_l3_f32`: `135.35 -> 185.73 ms`, throughput `1891.35 -> 1378.35/s`

Failed Because:

- stride を外へ出しても、実際には HLO 形が良くならず、batched path の fusion / scheduling を崩した可能性が高い。
- 「Python レベルでの再計算削減」が、そのまま XLA 実行時の利益に変わるとは限らない典型例だった。
- general improvement としては採れないので revert した。

Interpretation:

- decode まわりは「再計算を減らす」より「XLA が好む形を保つ」ほうが重要。
- ここから先は metadata の hoist より、term traversal や integer arithmetic の質を変える案へ進むべき。

### Idea 036

Idea:

- `_difference_rule_storage_device(max_level)` で full rule を毎 level 1 回だけ作り、`previous full weights` を使い回して差分則を構成する。

Why:

- 現行 storage builder は各 level で `_difference_rule_device(level)` を呼び、その中で `level` と `level-1` の full rule を毎回作り直していた。
- storage 全体を作るときは level が昇順に並ぶので、`previous_weights` を carry すれば再計算をかなり減らせる。
- これは high-level 1D init の timeout frontier に直接効く。

Implementation:

- [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py) に `_difference_rule_from_full_rule_device(...)` を追加。
- `_difference_rule_device(level)` 自体は public behavior を変えず、その helper を使う形に整理。
- `_difference_rule_storage_device(max_level)` では
  - `full_nodes, full_weights = _clenshaw_curtis_rule_device(level)`
  - `diff = _difference_rule_from_full_rule_device(level, full_nodes, full_weights, previous_full_weights)`
  - `previous_full_weights = full_weights`
  の形で昇順に積み上げるようにした。

Status:

- `done`

Validation:

- `python3 -m py_compile python/jax_util/functional/smolyak.py`
- `CUDA_VISIBLE_DEVICES=1 python3 -m pytest python/tests/functional/test_smolyak.py python/tests/functional/test_integrate.py -q`
- result: `30 passed`

Result:

- current と checkpoint `57b406a` の init-only probe を比較。
- `init_1d_l25_f64`: `39.04 s -> 36.99 s`, 約 `1.06x` 改善
- `init_16d_l3_f32`: `0.164 s -> 0.094 s`, 約 `1.74x` 改善
- `storage_bytes` と `num_points` は一致しており、数学的な結果は変わっていない。

Interpretation:

- high-level 1D では「完全に世界が変わる」ほどではないが、確実に init を削れている。
- 軽中規模ケースでは効きが大きく、full rule の重複生成が実際に無駄だったことが確認できた。
- 次は 1D rule builder のさらに深い incremental 化や、term traversal 側の改良を試すのが自然。

### Idea 037

Idea:

- `num_terms` は integrator 初期化時に static に分かっているので、term loop の本体を `has_next` つき `while_loop` から `fori_loop` へ寄せる。

Why:

- current 実装は term 列挙を動的生成していて、outer term loop も `while` で回している。
- ただし loop 回数自体は `num_terms` として static に持っているので、実行時の `has_next` 判定を毎 iteration 持ち回る必要はない。
- `vmap` 本番では loop 骨格の差が runtime に効く可能性がある。

Implementation:

- [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py) の `_smolyak_plan_integral(...)` に `num_terms` を追加。
- main term loop を
  - 通常: `lax.fori_loop(0, num_terms, term_body, state)`
  - fallback: `num_terms` が極端に大きいときだけ remaining-term count を持つ `while_loop`
  の形へ変更した。
- public path からは `integrator.num_terms` を渡す。
- internal test も private helper の新 signature に追随させた。

Status:

- `done`

Validation:

- `python3 -m py_compile python/jax_util/functional/smolyak.py python/tests/functional/test_smolyak.py`
- `CUDA_VISIBLE_DEVICES=1 python3 -m pytest python/tests/functional/test_smolyak.py python/tests/functional/test_integrate.py -q`
- result: `30 passed`

Result:

- `Idea 033` 版との runtime 比較:
  - `single_4d_l4_f32`: `7.50 -> 5.49 ms`, `1.37x`
  - `batch_4d_l4_f32`: `17.50 -> 14.18 ms`, throughput `14625.92 -> 18047.51/s`, `1.23x`
  - `single_8d_l4_f32`: `55.24 -> 32.79 ms`, `1.68x`
  - `batch_8d_l4_f32`: `98.15 -> 73.20 ms`, throughput `2608.36 -> 3497.46/s`, `1.34x`
  - `single_16d_l3_f32`: `42.99 -> 48.52 ms`, 約 `13%` 悪化
  - `batch_16d_l3_f32`: `135.35 -> 85.82 ms`, throughput `1891.35 -> 2982.89/s`, `1.58x`

HLO / memory (16D, level3, float32, batch256):

- temp size はほぼ不変:
  - `16865344 -> 16865600 bytes`
- `while` count もほぼ不変:
  - `single: 15 -> 16`
  - `batch: 34 -> 34`
- それでも runtime は特に batched path で大きく改善した。

Interpretation:

- loop skeleton の変更は、HLO op count や temp size だけでは見えにくいが、実行時には確かに効く。
- 少なくとも `vmap` 本命ケースでは明確に前進。
- `single_16d_l3_f32` のみ軽い悪化があるので、次は term traversal の中でも `current_extra_levels` 更新部分をもう少し詰めて、この outlier を確認する価値がある。

### Idea 038

Idea:

- term loop で `current_offsets/current_lengths` も carry し、各 term で `rule_offsets/rule_lengths` を全軸 `take` し直さず、chosen axis だけ差分更新する。

Why:

- current 実装では term ごとに
  - `current_offsets = take(rule_offsets, level_indices)`
  - `current_lengths = take(rule_lengths, level_indices)`
  を全軸分やり直していた。
- `_next_term_extra_levels(...)` は「chosen axis を 1 つ進めて trailing axes を level 1 へ戻す」形なので、
  - 軸 `< chosen`: そのまま
  - 軸 `== chosen`: 1 段だけ進める
  - 軸 `> chosen`: base level へ戻す
  の差分更新で済むはず、という仮説。

Implementation:

- [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py) の `_next_term_extra_levels(...)` が `chosen axis` を返すように変更。
- `_smolyak_plan_integral(...)` の term loop state を
  - `(acc, current_extra_levels)`
  から
  - `(acc, current_extra_levels, current_offsets, current_lengths)`
  へ広げた。
- `current_offsets/current_lengths` は全軸 gather せず、chosen axis と base rule から `where(...)` で差分更新する形にした。

Status:

- `rejected`

Validation:

- `python3 -m py_compile python/jax_util/functional/smolyak.py python/jax_util/functional/integrate.py`
- `python3 -m pytest python/tests/functional/test_integrate.py python/tests/functional/test_smolyak.py -q`
- result: `30 passed`

Result:

- GPU probe で current 正本と比較。
- baseline:
  - `single_8d_l4_f32`: `32.51 ms`
  - `batch_8d_l4_f32`: `36.76 ms`, throughput `6963.68/s`
  - `single_16d_l3_f32`: `48.45 ms`
  - `batch_16d_l3_f32`: `46.44 ms`, throughput `5512.14/s`
- modified:
  - `single_8d_l4_f32`: `46.49 ms`
  - `batch_8d_l4_f32`: `44.31 ms`, throughput `5777.65/s`
  - `single_16d_l3_f32`: `42.87 ms`
  - `batch_16d_l3_f32`: `52.77 ms`, throughput `4851.30/s`

Failed Because:

- 全軸 gather は減ったが、その代わり term loop の carry に `current_offsets/current_lengths` という `O(d)` ベクトル 2 本を追加してしまった。
- 結果として `while/fori_loop` の loop-carried state が太り、GPU 実行ではむしろ不利になった可能性が高い。
- これは「Python レベルでの再計算削減」が、そのまま XLA 実行時の利益に変わらない典型例だった。

Interpretation:

- term traversal では「毎回 gather しない」ことより、「loop carry を小さく保つ」ことのほうが重要。
- 以後の改造は
  - state を増やさない
  - もしくは state を増やしても scalar / small carry に留める
  方向へ寄せるべき。
- この案は revert した。

### Idea 039

Idea:

- 1D Clenshaw-Curtis / difference-rule storage を device FFT ではなく host NumPy FFT で構築し、最後にだけ device へ載せる。

Why:

- 前回 baseline の hard failure は `integrator_init` に集中し、代表例として
  - `2D level24 float32`: `_difference_rule_storage_device(...)` で OOM
  - `2D level27 float32`: cuFFT plan failure
  が出ていた。
- 1D rule 構築は init-only work なので、runtime kernel である必要が薄い。

Implementation:

- [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py) に
  - `_dct_type_one_numpy(...)`
  - `_clenshaw_curtis_weights_numpy(...)`
  - `_clenshaw_curtis_rule_numpy(...)`
  - `_difference_rule_storage_host(...)`
  を追加。
- `SmolyakIntegrator.__init__` の rule storage builder を
  - `_difference_rule_storage_device(...)`
  から
  - `_difference_rule_storage_host(...)`
  へ切り替えた。

Status:

- `done`

Validation:

- `python3 -m py_compile python/jax_util/functional/smolyak.py`
- `JAX_PLATFORMS=cpu PYTHONPATH=python python3 -m pytest python/tests/functional/test_integrate.py python/tests/functional/test_smolyak.py -q`
- result: `31 passed`

Representative Result:

- previous baseline:
  - `vmap_1d_l25_f64`: `integrator_init_seconds = 38.76 s`
  - `vmap_16d_l3_f32`: `integrator_init_seconds = 2.64 s`
- current probe:
  - `vmap_1d_l25_f64`: `init = 8.19 s`, `warm = 4.21 s`, throughput `237.72/s`
  - `vmap_16d_l3_f32`: `init = 0.0117 s`, `warm = 0.1430 s`, throughput `6994.77/s`

Tradeoffs:

- large rule storages still emit pinned-host allocation warnings during host->device staging
- but the catastrophic cuFFT-side init failures disappeared in the tested range

Interpretation:

- init bottleneckにはかなり効く。
- これは 1D 特化ではなく、全次元が共有する 1D rule builder の改善として意味がある。
- 現時点では `accepted`.

### Idea 040

Idea:

- `_decode_points_and_weights(...)` に `axis_count <= 4` の small-dim specialized path を入れ、generic mixed-radix decode の余計な stride 構築と weight matrix を避ける。

Why:

- previous baseline の pathological success / timeout frontier は
  - `2D level24+`
  - `4D level21`
  のような low-dim high-level 帯に強く出ていた。
- low dimension では decode 式を直接書いたほうが GPU integer path が素直になる可能性がある。

Implementation:

- [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py) の `_decode_points_and_weights(...)` に
  - `axis_count == 2`
  - `axis_count == 3`
  - `axis_count == 4`
  の specialized branch を追加。
- points は axis ごとに `take` して `stack`
- weights は axis ごとに `take` して積を取る

Status:

- `done`

Validation:

- `python3 -m py_compile python/jax_util/functional/smolyak.py`
- `JAX_PLATFORMS=cpu PYTHONPATH=python python3 -m pytest python/tests/functional/test_integrate.py python/tests/functional/test_smolyak.py -q`
- result: `31 passed`

Representative Result:

- baseline current code before specialization:
  - `vmap_2d_l18_f32`: `4970.48/s`
  - `vmap_4d_l12_f32`: `1057.05/s`
  - `vmap_16d_l3_f32`: `6805.65/s`
- after specialization:
  - `vmap_2d_l18_f32`: `5055.13/s`
  - `vmap_4d_l12_f32`: `1066.90/s`
  - `vmap_16d_l3_f32`: `6994.77/s`

Interpretation:

- improvement は大きくないが、low-dim と mid-dim の representative case で一貫して微増だった。
- 実装複雑度も限定的なので `accepted`.

### Idea 041

Idea:

- small-dim specialized decode の `take(..., mode=\"clip\")` を direct indexing (`rule_nodes[idx]`) に置き換える。

Why:

- bounds-safe gather を外せば HLO が軽くなる可能性がある。

Status:

- `rejected`

Result:

- representative probe で
  - `vmap_2d_l18_f32`: `5005/s -> 4859/s`
  - `vmap_16d_l3_f32`: `7083/s -> 7061/s`
  と、少なくとも明確な改善は出なかった。

Interpretation:

- generic path で以前だめだった傾向は small-dim path でも変わらなかった。
- revert 済み。

### Idea 042

Idea:

- scalar-valued integrand (`shape == ()` or `(1,)`) に対して singleton output axis を潰した fast path を入れる。

Why:

- scaling experiment の主な integrand は length-1 output なので、`tensordot` 周りが少し軽くなる可能性がある。

Status:

- `rejected`

Result:

- representative probe で
  - `vmap_1d_l25_f64`: `240.48/s -> 240.68/s` で実質横ばい
  - `vmap_2d_l18_f32`: `5005/s -> 4899/s`
  - `vmap_16d_l3_f32`: `7083/s -> 6471/s`
  と broad に悪化した。

Interpretation:

- singleton axis を消す利得より、extra branch / altered HLO のほうが重かった。
- revert 済み。

### Idea 043

Idea:

- decode の整数演算を「安全な範囲では `int32`」に落とす。

Why:

- GPU では `int64` より `int32` のほうが有利なことが多く、current decode は整数演算比率が高い。

Status:

- `rejected`

Result:

- representative probe で
  - `vmap_1d_l25_f64`: `237.72/s -> 224.71/s`
  - `vmap_2d_l18_f32`: `5055/s -> 4789/s`
  - `vmap_4d_l12_f32`: `1066.9/s -> 1017.5/s`
  - `vmap_16d_l3_f32`: `6994.8/s -> 6783.4/s`
  と全体に悪化した。

Interpretation:

- 期待に反して、この実装では `int32` 化は有利に働かなかった。
- cast / layout / generated code の都合で、単純な bit-width 縮小が勝ちにならない例だった。
- revert 済み。
