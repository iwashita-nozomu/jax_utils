# Smolyak Tuning Notes 2026-03-16

## Context

- branch:
  - `work/smolyak-tuning-20260316`
- related results branch:
  - `results/functional-smolyak-scaling-tuned`

## Source

- tuned large-scale run JSONL:
  - `/workspace/.worktrees/results-functional-smolyak-scaling-tuned/experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260316T093835Z.jsonl`
- tuned large-scale run log:
  - `/workspace/.worktrees/results-functional-smolyak-scaling-tuned/experiments/functional/smolyak_scaling/results/run_smolyak_scaling_20260316T093835Z.log`
- HLO CPU case:
  - `/workspace/.worktrees/work-smolyak-tuning-20260316/experiments/functional/smolyak_hlo/results/smolyak_hlo_20260316T110024Z.json`
  - `/workspace/.worktrees/work-smolyak-tuning-20260316/experiments/functional/smolyak_hlo/results/smolyak_hlo_20260316T110024Z_summary.json`
- HLO GPU case:
  - `/workspace/.worktrees/work-smolyak-tuning-20260316/experiments/functional/smolyak_hlo/results/smolyak_hlo_20260316T110043Z.json`
  - `/workspace/.worktrees/work-smolyak-tuning-20260316/experiments/functional/smolyak_hlo/results/smolyak_hlo_20260316T110043Z_summary.json`
- bottleneck scan:
  - `/workspace/.worktrees/work-smolyak-tuning-20260316/experiments/functional/smolyak_hlo/results/smolyak_bottleneck_scan_20260316.json`

## Observations

- `2026-03-16T10:39Z` 時点で JSONL は `160` 行。
- status の内訳はおおむね
  - `ok: 26`
  - `failed: 134`
- failure kind の内訳はおおむね
  - `error: 102`
  - `worker_terminated: 30`
  - `oom: 2`
- 実行済み dtype はこの時点では `float16` のみ。
- `level=0` は想定通り
  - `dimension must be positive`
  - `level must be positive`
  の入力検証エラーのみ。
- `level=1` は
  - `dimension=26` まで成功
  - `dimension=27, 28` で `CUDA OOM`
  - `dimension>=29` は主に `worker_terminated`
  という切り替わりになっている。
- `level=1` の成功ケースでは `num_points=1`。
- にもかかわらず `integrator_init_seconds` は次元に対して急増している。
  - `d=24`: 約 `7.89 s`
  - `d=25`: 約 `15.56 s`
  - `d=26`: 約 `32.03 s`
- 同じケースで `avg_integral_seconds` はおおむね `0.003 - 0.005 s` 程度に留まっている。
- 小さい HLO case でも `stablehlo.while`, `func.call`, `stablehlo.gather` が目立つ。
- HLO size の一例:
  - CPU (`dimension=4`, `level=3`, `float32`)
    - `preferred_text.utf8_bytes max`: `20401`
    - `hlo_proto_bytes max`: `64458`
    - `generated_code_size_in_bytes max`: `0`
    - `temp_size_in_bytes max`: `264260`
  - GPU (`dimension=2`, `level=1`, `float16`)
    - `preferred_text.utf8_bytes max`: `15336`
    - `hlo_proto_bytes max`: `46709`
    - `generated_code_size_in_bytes max`: `27132`
    - `temp_size_in_bytes max`: `168792`
- `level=1`, `num_points=1`, `num_terms=1` に固定した bottleneck scan でも、
  `integrator_init_seconds` が次元とともに急増する一方で、
  `lower_seconds`, `compile_seconds`, `first_execute_seconds` は比較的緩やか。
- CPU scan (`float32`) の一例:
  - `d=16`
    - `integrator_init_seconds`: `0.051 s`
    - `compile_seconds`: `0.775 s`
    - `stablehlo_utf8_bytes`: `56905`
    - `hlo_proto_bytes`: `193366`
    - `argument_size_in_bytes`: `524784`
  - `d=20`
    - `integrator_init_seconds`: `3.041 s`
    - `compile_seconds`: `1.067 s`
    - `stablehlo_utf8_bytes`: `70765`
    - `hlo_proto_bytes`: `252456`
    - `argument_size_in_bytes`: `8389232`
  - `d=24`
    - `integrator_init_seconds`: `32.771 s`
    - `compile_seconds`: `1.174 s`
    - `stablehlo_utf8_bytes`: `85351`
    - `hlo_proto_bytes`: `319201`
    - `argument_size_in_bytes`: `134218480`
- tuned large-scale run (`float16`, `level=1`) の `storage_bytes` もほぼ 2 倍則で増える。
  - `d=24`: `67109440`
  - `d=25`: `134218328`
  - `d=26`: `268436080`
- `level=1` でも `max_rule_level = dimension + level - 1 = dimension` なので、
  1D Clenshaw-Curtis 則は最大 `2^(dimension - 1) + 1` ノードまで事前構築されている。
  - `d=24`: 最大 1D ノード数 `8388609`
  - `d=26`: 最大 1D ノード数 `33554433`
- tuned large-scale run の並列 worker は 3 本とも存在している。
  - `gpu-0`, `gpu-1`, `gpu-2` に対して case は round-robin に割り当てられている。
  - JSONL 上でも `worker_label` と `assigned_gpu_index` は `0/1/2` に分かれている。
- 観測時点の worker process は 3 本とも CPU 使用率がほぼ `100%` で、
  RSS も大きい一方、`nvidia-smi pmon` では active kernel が見えなかった。
- 完了済み case でも `visible_device_id` は常に `0` だったが、
  これは各 worker が `CUDA_VISIBLE_DEVICES` により local device 0 として見ていると解釈できる。

## Interpretation

- `num_points=1` で `d=27` から OOM に入るので、主問題は積分点数そのものではない可能性が高い。
- 現時点での一次ボトルネックは、
  `initialize_smolyak_integrator()` が `prepared_level` までではなく
  `dimension + prepared_level - 1` までの 1D 差分則 storage を丸ごと事前構築している点である。
- とくに `level=1` では Smolyak 的な評価点数は 1 点でも、
  1D rule table の最大 level は `dimension` まで伸びるため、
  host 初期化と device 転送のコストが次元に対して実質指数増大している。
- したがって、ここで観測される `OOM` や `worker_terminated` は
  「Smolyak 点数が多い」よりも
  「1D rule table を作りすぎている」
  ことの影響が大きい。
- JIT / HLO 側にもコストはあるが、
  現時点の micro benchmark では主因ではない。
  `stablehlo` text や `hlo proto` は増えるものの、
  `integrator_init_seconds` の立ち上がりほど急ではない。
- したがって、旧版で支配的だった
  - host 側の巨大 point cloud
  とは別種のボトルネックへ移っていると考えられる。
- 現在の tuned run でも dtype が `float16` しか進んでいないので、case ordering が比較のしやすさをまだ損ねている可能性がある。
- HLO size は小さい case でも十分に可視化できるので、
  大規模 run を止めずに別 worktree で compile / lowering 側の兆候を追える。
- GPU の小さい case では `generated_code_size_in_bytes` が非ゼロで取れており、
  compile 後コードサイズの肥大化を以後の切り分け指標として使える。
- 「GPU1,2 が使われていない」ことを、そのまま multiprocessing bug とみなすのは早い。
  現時点の観測では、
  - case 配布は 3 worker に行われている
  - 3 worker は同時に CPU を使っている
  - しかし GPU kernel 実行は sparse
  なので、まずは host 側初期化支配が本命である。
- ただし runner の並列化実装はかなり重い。
  各 GPU 用 thread が各 case ごとに fresh `ProcessPoolExecutor(max_workers=1)` を作るので、
  process spawn / JAX import / device context 再生成のオーバーヘッドが case ごとに入る。
  これは一次ボトルネックではないにせよ、二次ボトルネックとして大きい。

## Idea

- まずは 1D rule table の事前構築戦略を見直す。
  `dimension + level - 1` 全体を materialize するのではなく、
  実際に必要な level の rule を遅延評価または外側 loop で参照する構造へ寄せる。
- `level=1` に限定した micro benchmark を切り出し、
  - integrator 初期化
  - JIT / lowering
  - 1 回目実行
  - 反復実行
  を分離して測る。
- `num_points=1` の経路だけに絞った HLO / compile cost 観察を行い、
  再帰アンローリングの影響を切り分ける。
- HLO 実験は
  - CPU で size と op 構造を見る
  - GPU で generated code size と temp size を見る
  という二段構えにすると、大規模 run と干渉しにくい。
- 二次ボトルネックとしては、現在の実行パスに残る
  - `gather`
  - `while`
  - `slice/concatenate`
  を順に潰していく。
  ただし、一次ボトルネックを消す前にここを最適化しても効果は限定的と考えられる。
- case ordering は
  - `level -> dimension -> dtype`
  もしくは
  - `dimension -> level -> dtype`
  に変更して、途中停止しても全 dtype を比較できるようにする。
- `worker_terminated` が `OOM` の後段エラーなのか、
  compile side の異常終了なのかをもう少し細かく分類できるようにする。
- runner 側は、GPU ごとに long-lived な worker process を 1 本ずつ持つ構造へ変える候補がある。
  そうすれば
  - process spawn
  - JAX import
  - device context 再生成
  の繰り返しを減らせる。
- 実験環境側の暫定対策として、
  - `workers_per_gpu`
  - worker ごとの CPU affinity 自動分割
  - `OMP_NUM_THREADS=1` などの thread oversubscription 抑制
  を runner に入れておく価値が高い。
- もし GPU 分離そのものをさらに検証したいなら、
  worker 内で
  - physical GPU UUID
  - `jax.devices()` の実体
  - `CUDA_VISIBLE_DEVICES`
  を結果 JSON へ残すと確実に判断できる。

## Follow-up

- 次に確認したいのは
  - `level=1` での compile/init scaling
  - dtype interleave の必要性
  - `worker_terminated` の実体
  の 3 点。

## Tooling

### Source

`2026-03-16` 時点で、この環境で実際に確認できたツールは次の通り。

- available
  - `nvidia-smi`
  - `nvcc`
  - `ncu`
  - `python3`
  - `pytest`
  - `rg`
- unavailable
  - `jq`
  - `nsys`
  - `py-spy`
  - `strace`
  - `perf`

### Interpretation

- 今回の Smolyak 解析では、すでに入っているツールだけでも
  - GPU 使用率の観察
  - 実行プロセスの確認
  - HLO dump / summary
  - テスト実行
  は十分行えた。
- 一方で、今回いちばん欲しいのは GPU kernel 単体の統計ではなく
  - host 側初期化
  - compile / lowering
  - GPU 実行
  の時間関係をまとめて見られるツールである。
- その意味では、`ncu` よりも `nsys` の優先度が高い。
- JSONL 集計や failure 種別の簡易整理には `jq` があるとかなり楽になる。
- Python 側の hot spot 確認には `py-spy` が有力。
- host 側の system-call / memory 周りを見るなら `strace` や `perf` も候補になる。

### Idea

Dockerfile 更新時の候補優先度は、現時点では次の順がよさそう。

1. `nsys`
   - CPU/GPU timeline を同時に見たい。
2. `jq`
   - JSONL の quick inspection を楽にしたい。
3. `py-spy`
   - Python 側の初期化 hot spot を見たい。
4. `strace`
   - process / memory / file I/O の偏りを確認したい。
5. `perf`
   - host 側のより低レベルな profiling 用。

### Note

- Dockerfile を更新するときは、
  - 実験で本当に使ったツール
  - 追加したい理由
  - 代替手段の有無
  を一緒に整理してから入れる。
- 今回の判断では、`nsys` と `jq` がまず有力。
