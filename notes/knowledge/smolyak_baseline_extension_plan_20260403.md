# Smolyak Baseline Extension Plan

## Goal

`smolyak_scaling` の baseline を、今後しばらく比較の基準として使える形に拡張する。

今回の前提は次の通り。

- 競合を避けるため、まずはコードを増やさず設計だけを固める
- 比較対象は `Smolyak` と `Monte Carlo`
- 実運用を意識して `single` と `vmap` の両方を測る
- compile 時間は副次的でよく、steady-state runtime と timeout frontier を重視する
- mode 切替を乱立させず、実験 runner の case 定義を整理して拡張する

## Current Baseline Limits

現状の baseline は `Smolyak x {single, vmap}` の 2 モードだけで、次がまだ弱い。

- `Monte Carlo` との同一 runner 上での同条件比較
- `single` と `vmap` の phase ごとの時間切り分け
- timeout の発生相
  - `init`
  - compile
  - first execute
  - steady-state execute
- fairness 指標
  - same-budget
  - same-output-shape
  - same-batch-size

## Proposed Mode Layout

一晩回す baseline の mode は 4 本にする。

1. `smolyak_single`
2. `smolyak_vmap`
3. `mc_single`
4. `mc_vmap`

runner 側の case 設計としては、`mode` を 1 つの文字列にするよりも、内部では次の 2 軸で持つほうがよい。

- `integrator_family in {smolyak, monte_carlo}`
- `execution_variant in {single, vmap}`

ただし JSONL や summary 出力には、読みやすさのために

- `mode = smolyak_single`
- `mode = smolyak_vmap`
- `mode = mc_single`
- `mode = mc_vmap`

を持たせるのがよい。

こうしておくと、

- scheduler の case 列挙は単純
- summary では 4 モードの表をそのまま作れる
- 後で `sobol_single` のような別積分器を足しても拡張しやすい

## Detailed Case Schema

baseline 拡張後の case は、最低限次の field を持つ。

- `case_id`
- `dimension`
- `level`
- `dtype_name`
- `integrator_family`
- `execution_variant`
- `mode`
- `budget_kind`
- `budget_value`
- `budget_reference`

期待する意味は次の通り。

- `integrator_family`
  - `smolyak`
  - `monte_carlo`
- `execution_variant`
  - `single`
  - `vmap`
- `mode`
  - `smolyak_single`
  - `smolyak_vmap`
  - `mc_single`
  - `mc_vmap`
- `budget_kind`
  - 当面は `same_budget`
- `budget_value`
  - `Smolyak` なら `num_evaluation_points`
  - `MC` なら `num_samples`
- `budget_reference`
  - `smolyak_num_points`

### Result Schema Additions

結果 JSONL には、現状項目に加えて次を足す。

- `mode`
- `integrator_family`
- `problem_batch_size`
- `failure_phase`
- `runner_failure_kind`
- `failure_source`
- `lowering_seconds`
- `first_execute_seconds`
- `warm_execute_seconds`
- `host_copy_seconds`
- `sampling_seconds`
  - `MC` のみ
- `integration_seconds`
  - `MC` / `Smolyak` 共通の「評価と縮約」に寄る時間

### Why Keep `level` For Monte Carlo

`MC` にも `level` を残す理由は、集計軸を揃えるためである。

- `Smolyak` は `(dimension, level)` が自然
- `MC` は本来 `(dimension, sample_count)` が自然

だが baseline 比較では

- `(dimension, level)` を主 index
- `MC` はその level に対応する same-budget sample count

にすると frontier 表、heatmap、失敗表が揃う。

## Scheduler Design

現行 scheduler はそのまま使い、case 列挙だけ拡張する。

つまり、scheduler は

- `dimension`
- `level`
- `dtype`
- `mode`

の直積を回すだけでよい。

### Recommended Worker Layout

overnight baseline はまず現状通り

- `gpu_indices = [0,1,2]`
- `workers_per_gpu = 1`

を既定とする。

理由:

- 高 level の 1D / `float64` は init が重く、同 GPU 2 worker は競合が強い
- まずは timeout frontier を clean に見たい

### Mode Ordering

mode の並び順は次がよい。

1. `smolyak_single`
2. `smolyak_vmap`
3. `mc_single`
4. `mc_vmap`

この順の利点:

- `Smolyak` の point count を先に知り、それを MC budget に流しやすい
- `single` / `vmap` の差が隣接し、ログが読みやすい
- `MC` の shared-sample 実装を後段に置ける

## Monte Carlo Detailed Design

### Core Estimator

`MC` は区間 `[-0.5, 0.5]^d` 上の一様分布を使う。

推定量は

- `I ≈ volume * mean(f(x_i))`

で、ここで volume は常に `1` なので、実装上は単純平均でよい。

### Same-Budget Sample Rule

各 `(dimension, level)` について

- `sample_count = smolyak_num_evaluation_points(dimension, level)`

を使う。

ただし初期実装では、安全のため次の補助 field を残す。

- `raw_budget_value`
- `effective_budget_value`

将来、`MC` だけ上限を設けたくなった時に後方解析できる。

### RNG / Sample Sharing

#### Single

- case ごとに 1 個の seed
- `sample_count x dimension` の sample matrix を生成

#### Vmap

まず baseline では、問題 batch 全体で sample 点を共有する。

つまり

- sample matrix は `sample_count x dimension`
- 1000 問題は同じ sample 点列を使う

これは `Smolyak` の

- 共通点
- 複数関数評価

と比較しやすく、また `outer vmap` の temp を直接比較しやすい。

### MC Output Fields

`MC` には次の field を追加したい。

- `num_samples`
- `sampling_seconds`
- `sample_points_device_nbytes`
- `shared_samples_across_batch`
- `mc_seed`

## Timing Design

JAX の benchmarking guidance では、dispatch が非同期なので実測時に `block_until_ready()` が必要とされている。baseline でも phase ごとに同期点を明示する必要がある。  
Source: https://docs.jax.dev/en/latest/benchmarking.html

### Proposed Timing Breakdown

1. `spawn_to_worker_seconds`
2. `jax_import_seconds`
3. `integrator_init_seconds`
4. `sampling_seconds`
5. `device_transfer_seconds`
6. `lowering_seconds`
7. `first_execute_seconds`
8. `warm_execute_seconds`
9. `host_copy_seconds`

### Measurement Rules

- 各 phase の終端で `block_until_ready()` を入れる
- host copy は `np.asarray(...)` の前後で測る
- compile と first execute は分ける
- timeout 時には最後に到達した phase を `failure_phase` に記録する

### Failure Phase Enumeration

標準化したい phase は次。

- `spawn`
- `jax_import`
- `init`
- `sampling`
- `transfer`
- `lowering`
- `first_execute`
- `warm_execute`
- `host_copy`

## XLA / JAX Investigation Findings

一次情報ベースで、いま効きそうな候補を baseline 観点で整理する。

### `jax_compiler_enable_remat_pass`

JAX docs では、自動 remat pass を切った方が良い場合があるとされている。  
Source: https://docs.jax.dev/en/latest/gpu_memory_allocation.html

この repo での既観測とも合っており、優先度は高い。

### `xla_gpu_enable_while_loop_double_buffering`

OpenXLA flags guidance に存在する。current HLO では `while` が目立つので、runtime / temp 両面で試す価値がある。  
Source: https://openxla.org/xla/flags_guidance

### `xla_latency_hiding_scheduler_rerun`

OpenXLA 側で latency-hiding scheduler の再実行回数を持つ。compile は重くなるが、runtime が支配的な夜間 baseline の tuning 軸としては候補。  
Source: https://openxla.org/xla/flags_guidance

### `xla_memory_scheduler`

OpenXLA flags guidance にあるが、runtime 直接改善というより memory/compile trade-off として扱う。baseline 既定値にはしない。  
Source: https://openxla.org/xla/flags_guidance

### Device Memory Profile

JAX docs には `save_device_memory_profile()` があり、OOM や temp 膨張の切り分けに使える。  
Source: https://docs.jax.dev/en/latest/device_memory_profiling.html

baseline 本番に常時は入れないが、frontier case 再現時の sidecar probe として有用。

### `lax.map(batch_size=...)`

JAX docs では、`lax.map(..., batch_size=...)` は小さい `vmap` を順に回す形でメモリ削減に使える。  
Source: https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html

これは baseline runner の mode には入れず、積分器側の将来改善案として扱う。

### `custom_vmap`

JAX docs では `custom_vmap` で batching rule を書けるが、reverse-mode autodiff はそのままではサポートしない。  
Source: https://docs.jax.dev/en/latest/_autosummary/jax.custom_batching.custom_vmap.html

これも baseline 拡張そのものではなく、積分 wrapper 改造の設計メモとして扱う。

## Concrete File-Level Plan

コード変更時の着地点を先に固定しておく。

### `experiments/functional/smolyak_scaling/run_smolyak_scaling.py`

ここでやること:

- `SUPPORTED_MODES` 追加
- case builder を 4 モード対応へ拡張
- `MC` sample count 計算を追加
- phase timing 追加
- `failure_phase` 追加
- CSV 出力追加

### `experiments/smolyak_experiment/cases.py`

ここでやること:

- baseline 用 integrand family の正本を整理
- `Smolyak` / `MC` で同じ family を引けるようにする

### `notes/knowledge/*`

ここでやること:

- tuning campaign と baseline campaign の分担を明文化
- overnight run ごとの意思決定ログを残す

## What To Investigate Before Coding

実装前に、最低限次を確認する。

1. `MC vmap` で shared-sample が本当に公平か
2. `MC` に点生成時間を含めるかを baseline 規約として固定する
3. CSV の粒度
   - raw only
   - raw + summary
   - raw + summary + frontier
4. timeout case を再実行する sidecar probe の要否

## Recommended Immediate Next Step

コード変更の順は次が最も安全。

1. `mode` を 4 本に増やす
2. `MC` same-budget sample count 実装
3. timing phase 拡張
4. summary / CSV 拡張
5. overnight baseline
6. 翌朝 timeout frontier を見て XLA tuning run を別で切る

## Monte Carlo Baseline Plan

### Main Idea

`Monte Carlo` の点数は、`Smolyak` の各 level に対して same-budget で揃える。

つまり `(dimension=d, level=l)` の `Smolyak` case に対して、

- `sample_count = smolyak_num_evaluation_points(d, l)`

を基本にする。

これは「level 1-50 に対して 50 パターンの MC 点数を用意する」というユーザ意図と一致している。

### Why Same-Budget First

同じ runner で baseline を取る段階では、最初の比較軸は

- same-budget

が最も素直である。

理由:

- `Smolyak` は level から点数が決まる
- `MC` は sample count から計算量が決まる
- runtime / timeout / throughput の比較では、まず「同じ評価回数」が最も解釈しやすい

### Monte Carlo Integrand Policy

被積分関数は `Smolyak` baseline と同じ family を使う。

- `single`
  - 係数ベクトル 1 本
- `vmap`
  - 係数ベクトル 1000 本

これで

- 積分器だけを変えた比較
- outer `vmap` の重さの比較

になる。

### Monte Carlo RNG Policy

case ごとに deterministic seed を割り当てる。

最低限ほしい field:

- `mc_base_seed`
- `mc_stream_id`
- `mc_sample_count`

`vmap` では

- batch 全体で同じ sample 点を共有するか
- 問題ごとに別 sample を引くか

を早めに固定する必要がある。

baseline 用としてはまず

- batch 全体で sample 点共有

がよい。

理由:

- 点生成コストを比較から外しやすい
- `Smolyak` と同様に「共通点に対する複数関数評価」という形に寄せられる
- outer `vmap` の temp 問題を比較しやすい

### Monte Carlo Case Table

baseline 一晩 run の case 列挙は次を基本にする。

- `dimension in 1..100`
- `level in 1..50`
- `dtype in {float16, bfloat16, float32, float64}`
- `mode in 4 modes`

ここで `mc_*` では level 自体は sample budget のインデックスとして扱う。

つまり `MC` でも `level` field は残すが、意味は

- `Smolyak` の level に対応する same-budget bucket

である。

この方が summary と frontier 表を揃えやすい。

## XLA / Runtime Investigation Plan

compile 時間は今回は主目的ではないので、runtime 改善に効きそうな候補を優先順位つきで整理する。

### Priority A

1. `jax_compiler_enable_remat_pass`
   - 現状でも効き目があった候補
   - compile より runtime / temp のバランスを見る
   - 値: `true`, `false`

2. `xla_gpu_enable_while_loop_double_buffering`
   - current HLO は `while` が目立つので相性がある
   - 値: `true`, `false`

3. `xla_latency_hiding_scheduler_rerun`
   - memory schedule が runtime にどう効くかは case 依存
   - 値: `1`, `3`, `5`

### Priority B

4. `xla_memory_scheduler`
   - memory 削減には効く可能性があるが、runtime 直接改善の主役ではない
   - compile は重くなる
   - 値: `kDefault`, `kBrkga`

5. `XLA_PYTHON_CLIENT_PREALLOCATE`
   - runtime 高速化というより fragmentation / OOM 安定性観点
   - 値: `true`, `false`

6. `XLA_PYTHON_CLIENT_MEM_FRACTION`
   - multi-process 共存向け
   - 一晩 run の安定性確認用

### Priority C

7. `XLA_PYTHON_CLIENT_ALLOCATOR=platform`
   - 最小メモリ観察には useful
   - 遅いので baseline 既定値にはしない

### Recommended Tuning Order

1. baseline
2. `remat_pass=false`
3. `double_buffering=false`
4. `remat_pass=false + double_buffering=false`
5. `latency_hiding_scheduler_rerun=3`
6. `latency_hiding_scheduler_rerun=5`

`xla_memory_scheduler` は compile が重いので、前 4 条件のあとでよい。

## Timing Points To Add

今の baseline に追加したい計測点は次。

### Process / Setup

- process spawn 到着
- worker environment 適用完了
- JAX import 完了
- integrator object 構築開始

### Integrator Init

- 1D rule generation 完了
- integrator dataclass / module 完了
- device transfer 完了

### Compilation

- lowering start
- lowering stop
- first compiled execute start
- first compiled execute stop

### Steady-State

- warm repeat start
- warm repeat stop
- host copy start
- host copy stop

### Failure Phase Tag

timeout や失敗には phase を付ける。

- `init`
- `transfer`
- `lowering`
- `first_execute`
- `warm_execute`
- `host_copy`

これがあると「time frontier」の原因がすぐ分かる。

## Additional Detailed Ideas

### 1. Summary を 4 モード比較の形に固定する

最低限出したい列:

- `mode`
- `num_success`
- `num_failure`
- `mean_runtime`
- `max_runtime`
- `frontier_dimension_at_level`
- `frontier_level_at_dimension`

### 2. JSONL とは別に中間 CSV を残す

残したい CSV:

- raw case table
- timeout-only table
- frontier table
- level-wise runtime summary
- dtype-wise throughput summary

### 3. timeout のあとも phase 別に再集計できるようにする

`failure_kind` に加えて

- `failure_phase`
- `runner_failure_kind`
- `failure_source`

を標準化する。

### 4. Same-budget と same-accuracy を分離する

baseline 一晩 run は same-budget でよいが、report では後処理で

- same-budget
- matched-accuracy

の両方を出せるようにしたい。

### 5. `vmap` batch size を baseline field として明示する

将来 1000 固定を外しても比較しやすいように、

- `problem_batch_size`

を常に持つ。

### 6. `single` と `vmap` を同じ runner で回しつつ、summary は分ける

case 列挙は一緒でよいが、summary は

- single frontier
- vmap frontier

で別表にする。

### 7. `Smolyak` と `MC` の公平性を metadata に残す

たとえば:

- `budget_kind = same_budget`
- `budget_reference = smolyak_num_points`

### 8. `MC` 点生成コストを別計測する

`MC` は点生成まで含めるかどうかで runtime 解釈が変わる。

だから

- `sampling_seconds`
- `integration_seconds`

を分けたい。

### 9. OOM と timeout を frontier の別軸として扱う

success frontier だけでなく

- first timeout frontier
- first OOM frontier

を別に集計する。

### 10. baseline と exploratory tuning run を分ける

`baseline` は固定条件、
`tuning` は XLA flag 比較、
`campaign` は設計探索、
という三層を崩さない。

## Suggested File / Output Layout

baseline 拡張後の結果は少なくとも次を置くとよい。

- `smolyak_scaling_*.json`
- `smolyak_scaling_*.jsonl`
- `smolyak_scaling_raw_cases.csv`
- `smolyak_scaling_frontier.csv`
- `smolyak_scaling_timeouts.csv`
- `smolyak_scaling_summary_by_mode.csv`

## Suggested Implementation Order

1. `mode` 軸を 4 モードへ拡張
2. `MC` same-budget sample count 生成
3. phase timing 拡張
4. failure phase tagging
5. CSV 出力
6. overnight baseline
7. XLA tuning A/B

## Notes About What Not To Do

- `Smolyak` と `MC` を別 runner に分けない
  - 同じ scheduler / timeout / JSONL で比較したい
- compile 時間を短くするために計算を省略しない
- `single` と `vmap` を混ぜた summary だけで判断しない
- timeout 後に原因 phase を捨てない

## References

- JAX GPU memory allocation: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
- JAX profiling device memory: https://docs.jax.dev/en/latest/device_memory_profiling.html
- JAX benchmarking guide: https://docs.jax.dev/en/latest/benchmarking.html
- OpenXLA flags guidance: https://openxla.org/xla/flags_guidance
