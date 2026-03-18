```markdown
# Smolyak Improvement Worktree Reset

## Context

- branch:
  - `work/smolyak-improvement-20260318`
- previous_worktree:
  - `/workspace/.worktrees/work-smolyak-improvement-20260318`
- reset_reason:
  - Smolyak merge conflict の解消時に `main` 側 helper API と experiment branch 実装を結合した前提が誤っていた

## Decision

- `python/jax_util/functional/smolyak.py` は `results/functional-smolyak-scaling-tuned` branch の module をそのまま `main` に戻す
- その統合前提で始めた旧 worktree の未コミット差分は破棄する
- Smolyak の次の改造作業は、修正後 `main` を起点に worktree を作り直して再開する

## Scope Difference

- 旧 worktree は `smolyak_grid` や追加 helper API を前提にしていた
- 新しい worktree では、その前提を持ち込まず、experiment branch から main に戻した module を起点に改造する

## Carry-Over

- 参照元 results branch:
  - `results/functional-smolyak-scaling-tuned`
- correction note:
  - [worktree_results_smolyak_scaling_tuned_2026-03-18.md](/workspace/notes/worktrees/worktree_results_smolyak_scaling_tuned_2026-03-18.md)

## Next Step

- corrected `main` から `work/smolyak-improvement-20260318` を切り直し、新しい `WORKTREE_SCOPE.md` を置く

## 2026-03-18: 活動要約

- `main` へ experiment-runner 関連のテスト・ドキュメントを統合
- ワークツリー運用の見直し（`.worktrees/` の管理）を確認

```
