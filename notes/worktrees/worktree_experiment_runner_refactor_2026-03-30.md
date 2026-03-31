# Experiment Runner Refactor Worktree Extraction

## Context

- branch:
  - `work/experiment-runner-refactor-20260330`
- worktree:
  - `/workspace/.worktrees/work-experiment-runner-refactor-20260330`
- base:
  - `12dee4834f106b37a23b3a81c20456658af33815`
- purpose:
  - `python/experiment_runner/` を standalone module として再整理し、fresh process 実行、GPU env 制御、軽量 monitor を main へ持ち帰れる形にする

## Scope

- `python/experiment_runner/`
- `python/tests/experiment_runner/`
- `documents/experiment_runner.md`
- `notes/experiments/experiment_runner_usage.md`
- `notes/themes/experiment_runner_main_integration.md`
- `notes/themes/experiment_runner_realtime_monitor.md`

## Commits On This Branch

1. `97ddea9 feat: add lightweight experiment runner monitor`
1. `3a6ef36 refactor: isolate experiment runner cases in fresh processes`
1. `2929c5c refactor: centralize experiment runner gpu env control`

## What Was Carried To Main

- `StandardRunner` を case ごとの fresh spawned child process 実行へ変更した。
- `resource_scheduler.py` へ `GPUEnvironmentConfig` を追加し、GPU 可視性と allocator 系 env の責務を scheduler 側へ寄せた。
- `subprocess_scheduler.py` へ monitor 接続点を追加した。
- `monitor.py` を追加し、軽量 HTTP + JSON API の runtime monitor を導入した。
- `gpu_runner.py` は削除し、GPU 固有制御は `resource_scheduler.py` 側へ統合した。
- `python/tests/experiment_runner/` に monitor / runner / GPU runner 系の回帰 test を追加した。
- `documents/experiment_runner.md` と `notes/experiments/experiment_runner_usage.md` を現行 API に合わせて更新した。
- theme note として monitor 設計および main integration 方針を `notes/themes/` へ持ち帰った。

## Validation

- branch side:
  - `python3 -m py_compile python/experiment_runner/*.py`
  - `pytest -q python/tests/experiment_runner`
  - result: `22 passed, 1 skipped`
- main carry-over side:
  - `python3 -m py_compile python/experiment_runner/*.py`
  - `pytest -q python/tests/experiment_runner`
  - `git diff --check`
  - result: `22 passed, 1 skipped`

## Action Log Summary

- `Idea:` GPU / JAX の import-sensitive な state を case 間で漏らさないため、runner を process pool ではなく fresh child process 基準へ寄せる。
- `Decision:` `StandardRunner` は case ごとに spawn child process を起動する実装へ置き換えた。
- `Decision:` GPU allocator と preallocation の設定は experiment script 側で直接組まず、`GPUEnvironmentConfig` へ集約する。
- `Decision:` 観測は重い外部監視基盤に広げず、`monitor.py` の軽量 HTTP surface に留める。
- `Interpretation:` `gpu_runner.py` の独立モジュールは責務が薄くなったため、`resource_scheduler.py` へ吸収する方が整理しやすい。
- `Decision:` `notes/experiments/` と `notes/themes/` は main 側の恒久パスへ張り替え、`.worktrees/...` の一時リンクは残さない。

## Key Observations

- fresh process 化により、GPU 実験で `CUDA_VISIBLE_DEVICES` や `JAX_PLATFORMS` のケース間汚染を避けやすくなった。
- GPU env の組み立てを scheduler 側へ寄せると、experiment script 側の責務がかなり薄くなる。
- lightweight monitor は `subprocess_scheduler.py` 系と相性がよく、host が `Popen` を握る構造をそのまま活用できる。
- note の carry-over では、実装 note だけでなく worktree extraction note も main 側へ必要になる。

## Follow-Up

- `experiments/smolyak_experiment/` を新しい `experiment_runner` の責務境界に合わせて段階的に移行する。
- monitor を常用する script と、使わない script の最小 wiring を整理する。
- branch を閉じる前に `notes/branches/work_experiment_runner_refactor_20260330.md` の status と入口リンクを最終状態へ更新する。
