# Worktree Scope Report

調査日時: 2026-03-18

概要: リポジトリ内のすべての Git worktree を走査し、各ワークツリーのルートに `WORKTREE_SCOPE.md` が存在するかを確認しました。ワークツリー側でスコープが欠落している場合は、作業に着手する前に `WORKTREE_SCOPE.md` を作成してください。

結果:

- `/workspace`: MISSING_SCOPE
- `/workspace/.worktrees/nn-develop-20260318`: HAS_SCOPE
- `/workspace/.worktrees/work-experiment-runner-generalization-20260317`: MISSING_SCOPE
- `/workspace/.worktrees/work-smolyak-improvement-20260318`: HAS_SCOPE

推奨アクション:

- `work-experiment-runner-generalization-20260317` ワークツリーに `WORKTREE_SCOPE.md` が存在しません。作業の目的・終了条件・必要な依存を明記したスコープをワークツリー内へ追加してください。
- ルート作業中の `/workspace` は通常 main 作業ツリーです。ここにも直接作業を行う場合は `WORKTREE_SCOPE.md` を用意するか、ワークツリーを切って作業を行ってください。

注記: ワークツリーのスコープ不足は main へ統合する際の混乱とコンフリクトを招きます。スコープを整備した上で、今回のようにエージェントを総動員して徹底的にレビュー・統合してください。
