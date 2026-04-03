# Worktree Note: Runner Single Path 2026-04-03

## Goal

- `experiment_runner` の実行経路を `StandardRunner` + spawn/fresh child process に統一する。
- `subprocess_scheduler.py` を削除する。
- `subprocess_scheduler` 側で成熟していた host-side lifecycle / monitor / JSONL helper のよい部分だけを標準系へ移す。

## Implemented

- `python/experiment_runner/result_io.py` を追加し、JSON 互換変換と JSONL append/read を集約した。
- `python/experiment_runner/runner.py` に monitor 接続、start/finish callback、fresh child process 前提の host-visible 観測点を追加した。
- `python/experiment_runner/resource_scheduler.py` に worker label、GPU slot、CPU affinity の metadata 構築を追加した。
- `experiments/functional/smolyak_scaling/run_smolyak_scaling.py` を `StandardRunner` / `StandardFullResourceScheduler` ベースへ移行した。
- `python/experiment_runner/subprocess_scheduler.py` と専用テスト群を削除した。

## Verification

- `python3 -m pyright python/experiment_runner python/tests/experiment_runner experiments/functional/smolyak_scaling/run_smolyak_scaling.py`
- `python3 -m pytest -q python/tests/experiment_runner`
- `PYTHONPATH=/workspace/.worktrees/work-runner-single-path-20260403/python python3 experiments/functional/smolyak_scaling/run_smolyak_scaling.py --platform cpu --dimensions 2:2 --levels 1:1 --dtypes float32 --num-repeats 1 --num-accuracy-problems 1 --output /tmp/smolyak_scaling_smoke.json`

## Notes

- CPU smoke では JAX の CUDA plugin 初期化警告が stderr に出るが、run 自体は成功し、JSON / JSONL を生成した。
- worktree root の一時 `WORKTREE_SCOPE.md` は carry-over note へ置き換える前提で管理した。
