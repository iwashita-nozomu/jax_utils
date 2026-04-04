# Smolyak Vmap XLA Memory

`SmolyakIntegrator` を `vmap(integrate(f_i))` で実運用する前提で、2026-04-02 時点までに分かった

- temp buffer の主因
- HLO / buffer assignment の見方
- 有効だった tuning
- 次に試す価値が高い改善案

をまとめる。

詳細な経緯は `notes/experiments/` に分散しているが、ここでは今後も引く実務メモだけを残す。

## Current Picture

- 既定 `chunk_size` は `16384`。これは [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L727) で設定されている。
- `batched_suffix_ndim=0` なら、integrator 内部の点列は基本的に `dim x chunk` であり、suffix 展開そのものが主犯ではない。
- それでも outer `vmap(1000)` を掛けると、compiled temp は `single` に対して約 `330x` に跳ねる。
- `single` と `batch` の HLO テキストはかなり似ているが、`memory_analysis()` と `buffer-assignment` では大差が出る。
- 現時点で batch 側 temp の主犯は `f32[1000, 16384]` で、これは `batch_size x chunk_size` に対応する。

## Local Evidence

### 0. Current Recheck on 2026-04-03

current `tensordot` 正本に戻したあとでも、結論は変わっていない。

dump:

- [/tmp/xla_dump_smolyak_single_cur/module_1201.jit_single_integral.sm_8.9_gpu_after_optimizations-memory-usage-report.txt](/tmp/xla_dump_smolyak_single_cur/module_1201.jit_single_integral.sm_8.9_gpu_after_optimizations-memory-usage-report.txt)
- [/tmp/xla_dump_smolyak_single_cur/module_1201.jit_single_integral.sm_8.9_gpu_after_optimizations-buffer-assignment.txt](/tmp/xla_dump_smolyak_single_cur/module_1201.jit_single_integral.sm_8.9_gpu_after_optimizations-buffer-assignment.txt)
- [/tmp/xla_dump_smolyak_batch_cur/module_1209.jit_batched_integrals.sm_8.9_gpu_after_optimizations-memory-usage-report.txt](/tmp/xla_dump_smolyak_batch_cur/module_1209.jit_batched_integrals.sm_8.9_gpu_after_optimizations-memory-usage-report.txt)
- [/tmp/xla_dump_smolyak_batch_cur/module_1209.jit_batched_integrals.sm_8.9_gpu_after_optimizations-buffer-assignment.txt](/tmp/xla_dump_smolyak_batch_cur/module_1209.jit_batched_integrals.sm_8.9_gpu_after_optimizations-buffer-assignment.txt)

結果:

- `single total bytes used = 232358` (`226.9 KiB`)
- `batch total bytes used = 65842138` (`62.79 MiB`)
- current batch の `preallocated-temp` でも最大は `f32[1000,16384]`
- `single` 側の対応物は `f32[16384]`

したがって、current code でも問題は

- `chunk_size=16384` そのもの
- ではなく
- `outer vmap batch axis` が掛かった結果、`chunk` と直積になった `f32[1000,16384]`

である。

補助的な op count もこの読みと整合する。

- `single`: `gather=14`, `while=135`, `dot_general=4`, `fusion=28`
- `batch`: `gather=24`, `while=153`, `dot_general=8`, `fusion=40`

つまり HLO の構造差は「別物」になるほど大きくはないが、batch 化で `while/gather/dot` 周辺が肥大化し、buffer assignment で巨大 temp が現れる。

### 0.5 Wrapper-Level `custom_vmap` Recheck on 2026-04-03

公開 API を

- `vmap(lambda f: integrate(f, integrator))(f_s)`

のまま保つため、[integrate.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/integrate.py) に wrapper-level の `custom_vmap` を入れた。  
非 batched direct call は従来どおり plain dispatch、array-backed な `eqx.Module` 系だけが batched path を通る。

結果ファイル:

- [smolyak_integrate_custom_vmap_hlo_20260403.json](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_hlo/results/smolyak_integrate_custom_vmap_hlo_20260403.json)
- [/tmp/xla_dump_smolyak_integrate_batch_20260403/module_1203.jit_compiled_batch.sm_8.9_gpu_after_optimizations-memory-usage-report.txt](/tmp/xla_dump_smolyak_integrate_batch_20260403/module_1203.jit_compiled_batch.sm_8.9_gpu_after_optimizations-memory-usage-report.txt)
- [/tmp/xla_dump_smolyak_integrate_batch_20260403/module_1203.jit_compiled_batch.sm_8.9_gpu_after_optimizations-buffer-assignment.txt](/tmp/xla_dump_smolyak_integrate_batch_20260403/module_1203.jit_compiled_batch.sm_8.9_gpu_after_optimizations-buffer-assignment.txt)
- [/tmp/smolyak_integrate_custom_vmap_probe.json](/tmp/smolyak_integrate_custom_vmap_probe.json)

対象ケース:

- GPU
- `dimension=1`
- `level=12`
- `dtype=float32`
- `chunk_size=16384`
- outer `vmap` batch size `1000`

構造:

- batching rule は `f` だけ batched、`integrator` は固定
- 内部では `lax.map(batch_size=auto_tile)` を使って problem batch を分割
- このケースでは `auto_tile=239` になっていた
- XLA dump では [integrate.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/integrate.py) 起点の outer `while` が確認できる

compiled memory:

- `single temp_size_in_bytes = 199248`
- `batch temp_size_in_bytes = 274000`

これは旧 plain `vmap` の

- `batch temp_size_in_bytes = 65801040`

に対して約 `240x` 小さい。

buffer assignment で見える最大 temp も変わった。

- 旧 plain `vmap`: `f32[1000,16384]`
- 新 wrapper batching:
  - `2 x s64[16384]`
  - `2 x f32[16384,1]`
  - `2 x f32[16384]`
  - `f32[4,239,1]`
  - `f32[239,1]`

つまり主犯はもう `batch_size x chunk_size` ではなく、`tile_size x chunk_size` と小さい slice 群に置き換わっている。

runtime trade-off:

- 旧 plain `vmap` probe: [/tmp/smolyak_tensordot_probe.json](/tmp/smolyak_tensordot_probe.json)
  - `batch warm_runtime_ms ≈ 6.184`
  - `throughput ≈ 161708 integrals/s`
- 新 wrapper batching probe: [/tmp/smolyak_integrate_custom_vmap_probe.json](/tmp/smolyak_integrate_custom_vmap_probe.json)
  - `batch warm_runtime_ms ≈ 27.211`
  - `throughput ≈ 36749 integrals/s`

したがって、

- memory には非常に効く
- ただし runtime は約 `4.4x` 悪化した

と読むべき。

現時点の解釈:

- wrapper-level batching という設計方向自体は正しい
- ただし `lax.map` tile 化だけでは throughput の代償がまだ大きい
- 次はこの wrapper 境界を保ったまま、
  - tile 内 kernel の形
  - auto tile heuristic
  - `while` の入れ子の減らし方
  を詰めるのが筋

### 0.6 Wrapper-Level Batching Reprobe on 2026-04-03

上の wrapper-level batching は memory 面では正しかったが、最初の heuristic は tile を小さく切りすぎていた。
[integrate.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/integrate.py) で次の 2 点を入れて reprobe した。

- target tile budget を `16 MiB -> 32 MiB`、下限を `4 MiB -> 8 MiB` に引き上げた
- `auto_tile >= axis_size` なら `lax.map(...)` を使わず plain `jax.vmap(...)` を通す fast path を追加した

reprobe:

- [/tmp/smolyak_integrate_custom_vmap_reprobe_20260403.json](/tmp/smolyak_integrate_custom_vmap_reprobe_20260403.json)
- [/tmp/xla_dump_integrate_tile479/module_1203.jit_run.sm_8.9_gpu_after_optimizations-memory-usage-report.txt](/tmp/xla_dump_integrate_tile479/module_1203.jit_run.sm_8.9_gpu_after_optimizations-memory-usage-report.txt)
- [/tmp/xla_dump_integrate_tile479/module_1203.jit_run.sm_8.9_gpu_after_optimizations-buffer-assignment.txt](/tmp/xla_dump_integrate_tile479/module_1203.jit_run.sm_8.9_gpu_after_optimizations-buffer-assignment.txt)

代表ケース:

- GPU
- `dimension=1`
- `level=12`
- `dtype=float32`
- outer batch size `1000`

結果:

- old wrapper batching:
  - `auto_tile=239`
  - `warm_runtime_ms ≈ 27.211`
  - `throughput ≈ 36749 integrals/s`
- current wrapper batching:
  - `auto_tile=479`
  - `warm_runtime_ms ≈ 15.992`
  - `throughput ≈ 62532 integrals/s`

この改善後も compiled temp は小さいままだった。

- current batch `Total bytes used = 350954` (`342.7 KiB`)
- 主な temp は
  - `2 x s64[16384]`
  - `2 x f32[16384]`
  - `2 x f32[16384,1]`
  - `f32[2,479,1]`
  - `f32[479]`
  - `f32[479,1]`

つまり

- old plain `vmap` の主犯 `f32[1000,16384]`
- old wrapper batching の小 temp

の中間ではなく、`small-temp` 側を維持したまま runtime をかなり戻せた。

別ケースでも完全な 1D 専用改善ではないことを確認した。

- `4D, level4, float32, batch=1000`
  - old wrapper batching throughput `≈ 9782/s`
  - current wrapper batching throughput `≈ 10145/s`

また、小 batch では `auto_tile >= axis_size` になって plain `vmap` fast path へ落ちる。

- `1D, level12, batch=32`: `tile=32`, `warm_ms ≈ 3.340`
- `1D, level12, batch=128`: `tile=128`, `warm_ms ≈ 3.641`

現時点の解釈:

- wrapper-level batching という境界は維持してよい
- ただし小 batch まで `lax.map` に落とすのは損
- 次の主戦場は
  - `smolyak.py` 本体の term / decode / point-block 側
  - とくに tile 化された batch path の中で残る `while/gather` の削減


### 1. Single vs Batch Memory Analysis

実験ファイル:

- [smolyak_single_vs_batch_20260402.memory.json](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_hlo/results/smolyak_single_vs_batch_20260402.memory.json)
- [smolyak_single_vs_batch_20260402.summary.json](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_hlo/results/smolyak_single_vs_batch_20260402.summary.json)

対象ケース:

- GPU
- `dimension=1`
- `level=12`
- `dtype=float32`
- `batch_size=1000`

結果:

- `single temp_size_in_bytes = 199248`
- `batch temp_size_in_bytes = 65801040`

したがって、問題は入力常駐サイズより compiled temp 側にある。

### 2. Buffer Assignment で見えた主犯

dump:

- [/tmp/xla_dump_smolyak_single_20260402/module_1201.jit_single_integral.sm_8.9_gpu_after_optimizations-buffer-assignment.txt](/tmp/xla_dump_smolyak_single_20260402/module_1201.jit_single_integral.sm_8.9_gpu_after_optimizations-buffer-assignment.txt)
- [/tmp/xla_dump_smolyak_batch_20260402/module_1207.jit_batched_integrals.sm_8.9_gpu_after_optimizations-buffer-assignment.txt](/tmp/xla_dump_smolyak_batch_20260402/module_1207.jit_batched_integrals.sm_8.9_gpu_after_optimizations-buffer-assignment.txt)

batch 側の大きい値:

- `loop_multiply_fusion = f32[1000,16384]`
- `gemm_fusion_dot.2 = f32[8,1000,1]`

この `f32[1000,16384]` が `62.50 MiB` を占め、batch の `preallocated-temp` の大半を食っている。

### 3. Code Correspondence

対応する実装箇所:

- gather に相当する点列取り出し: [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L501)
- `vmap(f)` と重みつき縮約: [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L649) [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L650)
- chunk loop 骨格: [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L673) [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L677)

重要な読み方:

- `16384` は `chunk_size`
- `1000` は outer `vmap` の batch size
- したがって、いまの大バッファは `batch_size x chunk_size`

current dump の metadata では、batch 側の `loop_multiply_fusion` は

- `jit(batched_integrals)/jit(main)/vmap(while)/body/while/body/dot_general`

に対応している。  
つまり、問題は単なる gather 単体ではなく

- gather で作った point block
- `vmap(f)` で出した関数値 block
- `dot_general` による weighted reduction

が outer batch 軸と結合したところで発生している。

### 4. HLO Text だけでは見抜きにくい

実験ファイル:

- [smolyak_single_vs_batch_20260402.jsonl](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_hlo/results/smolyak_single_vs_batch_20260402.jsonl)

`single` と `batch` の HLO 行数は

- `single: 146`
- `batch: 147`

で、見た目の差は小さい。  
しかし compiled temp は大きく違うので、今後は HLO テキストだけでなく

- `Executable.memory_analysis()`
- `after_optimizations-buffer-assignment.txt`
- `after_optimizations-memory-usage-report.txt`

をセットで見る。

## XLA Tuning Findings

代表比較:

- `1D level25 float64`
- `1D level27 float32`

結果群:

- [xla_tuning_20260402](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_scaling/results/xla_tuning_20260402)
- [xla_tuning_20260402_valid](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_scaling/results/xla_tuning_20260402_valid)

要点:

- `JAX_COMPILER_ENABLE_REMAT_PASS=false` は有効
  - `1D level27 float32` で `init 90.65s -> 71.02s`
  - `batch runtime` はほぼ同等か微改善
  - `peak` はやや増えることがある
- `--xla_gpu_enable_while_loop_double_buffering=false` は一部で `peak` を少し下げるが、init が悪化しやすい
- `cuda_malloc_async + use_cuda_host_allocator=true` は今回の case では timeout が増えた
- `--xla_memory_scheduler=kBrkga` はこの jaxlib/XLA では unsupported

現時点の実務メモ:

- 第一候補: `jax_compiler_enable_remat_pass=false`
- 条件付き候補: `xla_gpu_enable_while_loop_double_buffering=false`
- 非推奨: `cuda_malloc_async + use_cuda_host_allocator=true`
- 未対応: `xla_memory_scheduler=kBrkga`

## Immediate Implication

`vmap(integrate(f_i))` が本命 API なら、問題は

- `suffix` 展開ではなく
- gather 後に `batch_size x chunk_size` の中間を作る計算形

である。

従って、次の改善の主軸は XLA フラグではなく

- その巨大中間を作らない式変形
- gather と reduce の fused kernel 化
- batch と chunk をタイル化して streaming する実装

に寄る。

## Ten Next Ideas

以下は次に検討する価値が高い順に近い。  
各項目について「なぜ候補か」「ローカル状況との接点」を短く書く。

### 1. `batch x chunk` の巨大中間を作らない custom kernel 化

- 今の主犯は `f32[batch, chunk]`。
- JAX Pallas は GPU で block 単位の kernel を書けるので、gather と weighted reduction を streaming に書き下ろす候補になる。
- Smolyak では `points -> f(points) -> weight multiply -> reduction` が本体なので、この一連を 1 kernel へ寄せる価値が高い。

Source:

- https://docs.jax.dev/en/latest/pallas/index.html
- https://docs.jax.dev/en/latest/pallas/quickstart.html
- https://docs.jax.dev/en/latest/pallas/pipelining.html

### 2. outer `vmap` を保ったまま batch 軸を内部タイル化する batched API を積分器内部へ持つ

- user API は `vmap` 前提のまま、中では `batch_tile x chunk_tile` で処理する。
- これは `vmap` をやめる案ではなく、`vmap` 相当の batched semantics を積分器内へ吸収して temp を制御する案。
- 現状の compiled temp が `batch_size x chunk_size` に比例するので、ここを `batch_tile x chunk_tile` に落とすのが本丸。

Source:

- Pallas BlockSpec / grid docs: https://docs.jax.dev/en/latest/jax.experimental.pallas.html
- Pallas software pipelining: https://docs.jax.dev/en/latest/pallas/pipelining.html

### 3. `jax.remat` を自動 pass ではなく手動で局所適用する

- 公式 docs は自動 remat pass より手動 `jax.remat` が有利なことがあると明記している。
- 今の観測でも `jax_compiler_enable_remat_pass=false` が有効だったので、次は `decode_points_and_weights` や `prefix_chunk_body` の一部へ手動 remat を試すのが筋。

Source:

- https://docs.jax.dev/en/latest/gpu_memory_allocation.html
- https://docs.jax.dev/en/latest/gradient-checkpointing.html

### 4. gather と weight multiply の間にできる大配列を custom primitive / custom lowering で潰す

- buffer assignment では gather の出力と `loop_multiply_fusion` が主要な temp。
- JAX/XLA の一般 fusion で足りないなら、custom call / primitive で「gather + multiply + partial reduce」を一体化する価値がある。
- これは Pallas より下の選択肢だが、最終的な本命になりうる。

Source:

- OpenXLA lowering overview: https://openxla.org/xla/hlo_to_thunks
- Pallas design docs: https://docs.jax.dev/en/latest/pallas/design/design.html

### 5. `chunk_size` を固定値ではなく `batch_size` 依存で決める

- 現在の主バッファは `batch_size x chunk_size`。
- `chunk_size=16384` は scalar path では自然でも、`vmap(1000)` に対しては大きすぎる可能性がある。
- まずは heuristic として `target_temp_budget_bytes` から逆算する adaptive chunking を試す価値がある。

Source:

- OpenXLA buffer assignment / temp dump docs: https://openxla.org/xla/hlo_to_thunks
- HLO dump docs: https://openxla.org/xla/hlo_dumps

### 6. batch 軸の layout を明示的に制御して gather / dot の食い違いを減らす

- JAX は device-local layout control を持っている。
- 今の `f32[1000,16384]{1,0}` と GEMM 側の都合が完全に噛み合っていない可能性があるので、layout を試す余地がある。
- 効果は不確実だが、今のように batch 軸が太いと layout 差が temp に効くことがある。

Source:

- https://docs.jax.dev/en/latest/notebooks/layout.html

### 7. device memory profiling を runner に組み込み、compile 後の live buffer を callgraph で残す

- `memory_analysis()` は executable 単位の集計しか見えない。
- `jax.profiler.save_device_memory_profile()` なら Python stack 付きで live buffer を追える。
- `jit` 境界は opaque だが、runner 上で compile 前後・実行後の profile を残せば「何の段階で増えたか」の切り分けに使える。

Source:

- https://docs.jax.dev/en/latest/device_memory_profiling.html
- https://docs.jax.dev/en/latest/_autosummary/jax.profiler.device_memory_profile.html

### 8. `vmap` path 専用の HLO / buffer-assignment dump を自動生成する benchmark mode を整備する

- いまは ad hoc に `/tmp` へ dump している。
- 以後も使うので、`single` と `batch` を同一 case で自動比較し、`memory-usage-report` と `buffer-assignment` を成果物として残す mode を持つと良い。
- 実装改善前後の regression check にも使える。

Source:

- HLO dump docs: https://openxla.org/xla/hlo_dumps
- XLA tooling docs: https://openxla.org/xla/tools

### 9. batched integrand が線形・指数型なら specialized path を持つ

- 今回の benchmark は `exp(dot(c, x))`。
- これは一般 `f` より構造が強く、`X @ C^T` へ落ちる。
- benchmark だけでも specialized evaluator を持てば、「積分器本体の問題」と「benchmark integrand の都合」を切り分けられる。
- 一般 API とは別の benchmark mode としてなら有用。

Source:

- local observation from buffer assignment: `gemm_fusion_dot.2` is already recognized as GEMM-like in batched benchmark

### 10. 1D rule storage / decode 周りを `Ref` と scratch memory で再設計する

- Pallas docs は block ごとに HBM から SRAM へ持ち込んで処理する前提を推している。
- 現在は gathered point block と weight block を generic JAX array として扱っているため、compiler が大きな temp を取りやすい。
- `Ref` / `BlockSpec` / `scratch_shapes` を使う設計に寄せると、streaming 化の足場になる。

Source:

- https://docs.jax.dev/en/latest/jax.experimental.pallas.html
- https://docs.jax.dev/en/latest/pallas/gpu/reference.html
- https://docs.jax.dev/en/latest/pallas/pipelining.html

## Representative Sources

- JAX GPU memory allocation:
  - https://docs.jax.dev/en/latest/gpu_memory_allocation.html
- JAX buffer donation:
  - https://docs.jax.dev/en/latest/buffer_donation.html
- JAX device memory profiling:
  - https://docs.jax.dev/en/latest/device_memory_profiling.html
- JAX Pallas:
  - https://docs.jax.dev/en/latest/pallas/index.html
  - https://docs.jax.dev/en/latest/pallas/quickstart.html
  - https://docs.jax.dev/en/latest/pallas/pipelining.html
  - https://docs.jax.dev/en/latest/pallas/gpu/reference.html
- OpenXLA HLO / buffer assignment:
  - https://openxla.org/xla/hlo_dumps
  - https://openxla.org/xla/hlo_to_thunks

## 2026-04-04 Bottleneck Identification

current code の representative case として

- `dimension=2`
- `level=18`
- `dtype=float32`
- outer batch size `1000`

で fresh dump を取り直した。

files:

- [/tmp/xla_dump_smolyak_current_2d18/module_0011.jit_compiled_single.sm_8.9_gpu_after_optimizations-memory-usage-report.txt](/tmp/xla_dump_smolyak_current_2d18/module_0011.jit_compiled_single.sm_8.9_gpu_after_optimizations-memory-usage-report.txt)
- [/tmp/xla_dump_smolyak_current_2d18/module_0013.jit_compiled_batch.sm_8.9_gpu_after_optimizations-memory-usage-report.txt](/tmp/xla_dump_smolyak_current_2d18/module_0013.jit_compiled_batch.sm_8.9_gpu_after_optimizations-memory-usage-report.txt)
- [/tmp/xla_dump_smolyak_current_2d18/module_0013.jit_compiled_batch.sm_8.9_gpu_after_optimizations-buffer-assignment.txt](/tmp/xla_dump_smolyak_current_2d18/module_0013.jit_compiled_batch.sm_8.9_gpu_after_optimizations-buffer-assignment.txt)
- [/tmp/xla_dump_smolyak_current_2d18/module_0013.jit_compiled_batch.thunk_sequence.txt](/tmp/xla_dump_smolyak_current_2d18/module_0013.jit_compiled_batch.thunk_sequence.txt)

### What Dominates Time

静的 HLO なので厳密な wall-time 比率までは出ないが、buffer assignment と thunk sequence から支配演算はかなり絞れる。

- batch path の最大 temp は `gemm_fusion_dot.141 = f32[479,16384]`
- これは約 `31,391,744 bytes`
- `compiled_batch` 全体の live bytes `~32.22 MiB` のほぼ大半を占める

thunk sequence でも主系列は

- `input_concatenate_fusion_1`
- `gemm_fusion_dot_141`
- `input_reduce_fusion`

になっている。

したがって current `vmap` path の支配演算は

1. `values[tile_batch, chunk]` の batched 関数値行列生成
2. その後段の `exp + weight multiply + reduce`

である。

source 対応は

- points block 構築: [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L813)
- non-suffix path の `vmap(f)` と縮約: [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L822) [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py#L824)
- wrapper-level tile loop: [integrate.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/integrate.py#L140)

### What Does Not Dominate

- `broadcast_in_dim` 自体はもう主犯ではない
- wrapper batching の `wrapped_slice` / `dynamic_update_slice` は見えるが、サイズ・launch 規模とも二次要因
- Smolyak 固有の `divide/remainder/gather/while` は single path では主役だが、current batch path では `gemm + reduce` より一段下

single path では事情が違い、`compiled_single` の temp は `~195.6 KiB` しかない。こちらは

- `input_convert_gather_reduce_fusion`
- `input_reduce_fusion`
- small `while`

が中心で、Smolyak decode / gather が相対的に重い。

### Interpretation

「積分の本質」は `f(x)` の大量評価と重み付き和なので、`gemm/reduce` が支配すること自体は自然である。
ただし current implementation は、その本質計算を

- まず `values[tile_batch, chunk]` として materialize
- そのあと縮約

という形で実行している。ここはまだ改善余地がある。

### Immediate Design Space

一般 `f` を保ったまま `values[tile_batch, chunk]` を小さくする手段は 3 段ある。

1. plain JAX:
   - `chunk -> point_tiles`
   - tile ごとに `vmap(f)`
   - tile partial sum を即時加算
2. Pallas:
   - point / weight load
   - `f` evaluation
   - partial reduction
   を block kernel 化
3. C++ / CUDA custom call:
   - さらに低レベルだが、現時点ではやりすぎ

現時点の推奨順は

1. tile-local partial sum を plain JAX で試す
2. まだ temp / launch overhead が残るなら Pallas
3. custom call は最後

である。
