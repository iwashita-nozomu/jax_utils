# WORKTREE_SCOPE

## Worktree Summary

- Branch: `work/nn-df-zero-20260404`
- Worktree path: `/workspace/.worktrees/nn-df-zero-20260404`
- Purpose: ラグランジアンの一回微分に基づく最急法のうち、parameter update を `DF=0` の解として与える枠組みを、文献調査、数理検証、設計、実装の順で別サブモジュールとして進める
- Owner or agent: Codex

## Editable Directories

- `python/jax_util/neuralnetwork/lagrangian_df_zero/`
- `python/tests/neuralnetwork/`
- `documents/design/`
- `notes/themes/`
- `notes/worktrees/`
- `references/`

## Runtime Output Directories

- `experiments/` は準備段階では未使用

## Read-Only Or Avoid Directories

- `python/jax_util/neuralnetwork/neuralnetwork.py`
- `python/jax_util/neuralnetwork/train.py`
- `python/jax_util/neuralnetwork/sequential_train.py`
- `python/jax_util/neuralnetwork/dynamic_programming.py`
- `python/experiment_runner/`
- `experiments/smolyak_*`
- `reports/`
- `scripts/security/`

## Required References Before Editing

- [documents/worktree-lifecycle.md](/workspace/documents/worktree-lifecycle.md)
- [documents/coding-conventions-project.md](/workspace/documents/coding-conventions-project.md)
- [documents/coding-conventions-python.md](/workspace/documents/coding-conventions-python.md)
- [documents/research-workflow.md](/workspace/documents/research-workflow.md)
- [documents/experiment-workflow.md](/workspace/documents/experiment-workflow.md)
- [documents/design/jax_util/README.md](/workspace/documents/design/jax_util/README.md)
- [python/jax_util/neuralnetwork/neuralnetwork.py](/workspace/python/jax_util/neuralnetwork/neuralnetwork.py)
- [python/jax_util/neuralnetwork/sequential_train.py](/workspace/python/jax_util/neuralnetwork/sequential_train.py)
- [python/jax_util/neuralnetwork/dynamic_programming.py](/workspace/python/jax_util/neuralnetwork/dynamic_programming.py)

## Main Carry-Over Targets

- `python/jax_util/neuralnetwork/lagrangian_df_zero/`
- `python/tests/neuralnetwork/`
- `documents/design/jax_util/`
- `notes/themes/`
- `references/`
- `notes/worktrees/worktree_nn_df_zero_2026-04-04.md`

## Working Notes During Execution

- Action log path: `notes/worktrees/worktree_nn_df_zero_2026-04-04.md`

## Required Checks Before Commit

- `python3 -m pyright /workspace/.worktrees/nn-df-zero-20260404/python/jax_util/neuralnetwork/lagrangian_df_zero`
- `python3 -m pytest -q python/tests/neuralnetwork`
- `python3 -m mdformat --check documents/design/README.md documents/design/jax_util/README.md documents/design/jax_util/lagrangian_df_zero.md notes/themes/README.md notes/themes/06_lagrangian_df_zero.md notes/themes/bellman_bp.md notes/worktrees/worktree_nn_df_zero_2026-04-04.md references/README.md references/lagrangian_df_zero/README.md references/lagrangian_df_zero/catalog.md`
- `python3 scripts/tools/check_markdown_lint.py documents/design/README.md documents/design/jax_util/README.md documents/design/jax_util/lagrangian_df_zero.md notes/themes/README.md notes/themes/06_lagrangian_df_zero.md notes/themes/bellman_bp.md notes/worktrees/worktree_nn_df_zero_2026-04-04.md references/README.md references/lagrangian_df_zero/README.md references/lagrangian_df_zero/catalog.md`
- `git diff --check`

## Additional Rules

- 既存 `neuralnetwork` の公開面は直接壊さない
- `DF=0` 型の更新則は新サブモジュール内に閉じる
- 文献調査、数理的検証、設計 review を通す前に本実装へ入らない
- 依存パッケージを試す場合は一時 install に留め、carry-over 前に Docker 正本更新要否を明記する
