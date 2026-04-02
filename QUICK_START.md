# Quick Start

このファイルは、人間がこの repo で作業を始めるための最短手順です。
詳細な背景説明より、今すぐ必要な入口とコマンドだけを置きます。

## 1. 最初に読む

- [README.md](/workspace/README.md)
- [documents/worktree-lifecycle.md](/workspace/documents/worktree-lifecycle.md)
- [documents/conventions/README.md](/workspace/documents/conventions/README.md)

実験を行う場合は、追加で次を読みます。

- [experiments/README.md](/workspace/experiments/README.md)
- [documents/experiment-workflow.md](/workspace/documents/experiment-workflow.md)

## 2. 新しい作業を始める

新しい作業は `main` で直接始めません。必ず worktree を作ります。

```bash
bash scripts/setup_worktree.sh work/<topic>-YYYYMMDD
cd .worktrees/work-<topic>-YYYYMMDD
```

例:

```bash
bash scripts/setup_worktree.sh work/solver-fix-20260402
cd .worktrees/work-solver-fix-20260402
```

worktree を作ったら、最初に `WORKTREE_SCOPE.md` を埋めます。

- 何を直すか
- どのディレクトリを編集するか
- どのチェックを通すか
- `main` に何を持ち帰るか

## 3. 実装前の確認

```bash
bash scripts/guide.sh
bash scripts/view_conventions.sh naming
make ci-quick
```

`make ci-quick` と `make ci`、`bash scripts/run_pytest_with_logs.sh` は既定で `JAX_PLATFORMS=cpu` を使います。GPU 前提の確認をしたい場合だけ、明示的に環境変数を上書きします。

フルチェックは次です。

```bash
make ci
```

## 4. よく使うコマンド

```bash
# 規約一覧
bash scripts/view_conventions.sh

# pytest をログ付きで実行
bash scripts/run_pytest_with_logs.sh

# 包括 review を実行
bash scripts/run_comprehensive_review.sh

# Markdown とリンクを確認
python3 scripts/tools/check_markdown_lint.py documents
python3 scripts/tools/audit_and_fix_links.py documents
```

## 5. 実験の基本

- 実験コードは `experiments/<topic>/` に置きます。
- 実行ごとの生成物は `experiments/<topic>/result/<run_name>/` に置きます。
- 1 回の実験 report は `experiments/report/<run_name>.md` に置きます。
- partial run を正式結果として再開・継ぎ足しすることを禁止します。

## 6. 終了時の整理

```bash
git status --short
git worktree list
```

不要になった worktree は、carry-over を `main` に持ち帰ってから削除します。

```bash
git worktree remove .worktrees/work-<topic>-YYYYMMDD
git branch -d work/<topic>-YYYYMMDD
```

## 7. 参照先

- [scripts/README.md](/workspace/scripts/README.md)
- [documents/README.md](/workspace/documents/README.md)
- [documents/TROUBLESHOOTING.md](/workspace/documents/TROUBLESHOOTING.md)
