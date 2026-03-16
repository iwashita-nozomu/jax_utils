# Smolyak Tuning Worktree Extraction

## Context

- branch:
  - `work/smolyak-tuning-20260316`
- extracted_to_main_from:
  - `/workspace/.worktrees/work-smolyak-tuning-20260316`
- downstream_results_branch:
  - `results/functional-smolyak-scaling-tuned`
- scope:
  - Smolyak 積分器の軽量化
  - HLO 解析の拡充
  - GPU 可視性の切り分け

## Conventions

- `Source:`
  - 実際に見たファイル、結果、ログ
- `Interpretation:`
  - Source からの解釈
- `Decision:`
  - 最終的に残す判断

## HLO Analysis Expansion

### Source

- HLO dump:
  - `/workspace/.worktrees/work-smolyak-tuning-20260316/python/jax_util/hlo/dump.py`
- HLO summary:
  - `/workspace/.worktrees/work-smolyak-tuning-20260316/scripts/hlo/summarize_hlo_jsonl.py`
- HLO experiment:
  - `/workspace/.worktrees/work-smolyak-tuning-20260316/experiments/functional/smolyak_hlo/run_smolyak_hlo_case.py`
- HLO results:
  - `/workspace/.worktrees/work-smolyak-tuning-20260316/experiments/functional/smolyak_hlo/results/smolyak_hlo_20260316T110024Z.json`
  - `/workspace/.worktrees/work-smolyak-tuning-20260316/experiments/functional/smolyak_hlo/results/smolyak_hlo_20260316T110043Z.json`
  - `/workspace/.worktrees/work-smolyak-tuning-20260316/experiments/functional/smolyak_hlo/results/smolyak_bottleneck_scan_20260316.json`
- tuning note:
  - `/workspace/.worktrees/work-smolyak-tuning-20260316/notes/experiments/smolyak_tuning_20260316.md`

### Interpretation (HLO Analysis)

- HLO 解析拡充の成果は commit 済みで、`results/functional-smolyak-scaling-tuned` にすでに含まれている。
- tuning worktree を残す理由として、HLO 関連の tracked file はもうない。
- 主な知見は:
  - `stablehlo.while`
  - `func.call`
  - `stablehlo.gather`
  が目立ち、積分本体の算術より制御フローと index 処理の比重が高い、というものだった。

## GPU Visibility Probe

### Source

- untracked probe script:
  - `/workspace/.worktrees/work-smolyak-tuning-20260316/experiments/functional/smolyak_scaling/debug_gpu_visibility.py`

### Interpretation (GPU Visibility Probe)

- probe は 2 系統を比較していた:
  - `ProcessPoolExecutor(..., initializer=...)`
  - `subprocess.run(...)`
- どちらも child 側では
  - `CUDA_VISIBLE_DEVICES=<assigned>`
  - `jax.devices() == [cuda:0]`
  - `jax.devices("gpu")` の local id は `[0]`
  を返す設計だった。
- したがって、この probe が残していた結論は
  - 「child ごとの GPU 可視性分離は成立している」
  - 「GPU 1,2 が遊んで見える主因は可視性バグではなく、CPU 側初期化支配」
  である。

### Decision

- probe script 自体は debug 用の一時ファイルなので、main や results branch には持ち込まない。
- 結論だけ本ノートへ残す。

## Small CPU Smoke Results

### Source

- untracked results:
  - `smolyak_scaling_cpu_20260316T082844Z.json/.jsonl`
  - `smolyak_scaling_cpu_20260316T083540Z.json/.jsonl`
  - `smolyak_scaling_cpu_20260316T084926Z.json/.jsonl`

### Interpretation (Small CPU Smoke Results)

- 3 run とも条件は同じだった:
  - `platform=cpu`
  - `dimensions=[1]`
  - `levels=[1, 2]`
  - `dtype_names=['float32']`
  - `num_cases=2`
- 3 run とも `status={'ok': 2}` で、最初の case は
  - `dimension=1`
  - `level=1`
  - `dtype_name=float32`
  - `num_points=1`
  - `integrator_init_seconds ≈ 0.009 - 0.012`
  - `avg_integral_seconds ≈ 1.0e-4 - 1.4e-4`
  だった。
- これらは tuning 中の手元確認結果であり、branch の本質的な成果物ではない。

### Decision

- これらの JSON/JSONL は持ち込まない。
- 内容は「小さい CPU smoke は通っていた」という確認として本ノートに要約だけ残す。

## Branch Containment

### Source

- `git rev-list --left-right --count work/smolyak-tuning-20260316...results/functional-smolyak-scaling-tuned`
- result:
  - `0 17`

### Interpretation

- `work/smolyak-tuning-20260316` にしかない commit は残っていない。
- tracked history は `results/functional-smolyak-scaling-tuned` に完全に包含されている。

## Final Decision

- tuning worktree の tracked な成果は `results/functional-smolyak-scaling-tuned` に包含済み。
- 未追跡の `debug_gpu_visibility.py` と小さい CPU smoke JSON/JSONL は、本ノートへの要約吸い出しで十分。
- したがって、`/workspace/.worktrees/work-smolyak-tuning-20260316` は削除してよい。
