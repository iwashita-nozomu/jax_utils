# scripts

`scripts/` は、開発・review・実験運用を補助する実行入口をまとめる場所です。
ここには「いま使うスクリプト」だけを書き、存在しない補助 README や古い数え上げ表は置きません。

## 最初に使うもの

### worktree

- [setup_worktree.sh](/workspace/scripts/setup_worktree.sh)
  - 新しい branch / worktree を作る正本です。
- [tools/create_worktree.sh](/workspace/scripts/tools/create_worktree.sh)
  - 互換ラッパーです。
- [guide.sh](/workspace/scripts/guide.sh)
  - 作業の入口と状況確認に使います。

### CI / review

- [ci/run_all_checks.sh](/workspace/scripts/ci/run_all_checks.sh)
  - `pytest`、`pyright`、`pydocstyle`、`ruff` を実行します。
- [run_pytest_with_logs.sh](/workspace/scripts/run_pytest_with_logs.sh)
  - pytest の実行ログを保存します。
- [run_comprehensive_review.sh](/workspace/scripts/run_comprehensive_review.sh)
  - 包括 review 用のチェックをまとめて実行します。

### ドキュメント / 整理

- [tools/check_markdown_lint.py](/workspace/scripts/tools/check_markdown_lint.py)
- [tools/audit_and_fix_links.py](/workspace/scripts/tools/audit_and_fix_links.py)
- [tools/fix_markdown_docs.py](/workspace/scripts/tools/fix_markdown_docs.py)
- [tools/find_similar_documents.py](/workspace/scripts/tools/find_similar_documents.py)
- [tools/tfidf_similar_docs.py](/workspace/scripts/tools/tfidf_similar_docs.py)

### 実験補助

- [hlo/summarize_hlo_jsonl.py](/workspace/scripts/hlo/summarize_hlo_jsonl.py)
- [tools/check_worktree_scopes.sh](/workspace/scripts/tools/check_worktree_scopes.sh)

## よく使うコマンド

```bash
# 新しい worktree
bash scripts/setup_worktree.sh work/<topic>-YYYYMMDD

# 軽量 CI
make ci-quick

# フル CI
make ci

# pytest をログ付きで実行
bash scripts/run_pytest_with_logs.sh

# 包括 review
bash scripts/run_comprehensive_review.sh

# Markdown とリンク確認
python3 scripts/tools/check_markdown_lint.py documents
python3 scripts/tools/audit_and_fix_links.py documents
```

## 実行環境

- shell スクリプトは `python3` があればそれを優先し、なければ `python` を使います。
- Python 実行時の基本 import path は `PYTHONPATH=/workspace/python` です。
- repo 全体の pytest 系スクリプトは既定で `JAX_PLATFORMS=cpu` を使います。GPU 検証を行うときだけ呼び出し側で明示的に上書きします。
- ローカル仮想環境の作成を禁止します。依存追加時は `docker/Dockerfile` と `docker/requirements.txt` を同時に更新します。

## 参照先

- [README.md](/workspace/README.md)
- [QUICK_START.md](/workspace/QUICK_START.md)
- [documents/tools/README.md](/workspace/documents/tools/README.md)
- [documents/WORKFLOW_INVENTORY.md](/workspace/documents/WORKFLOW_INVENTORY.md)
- [documents/worktree-lifecycle.md](/workspace/documents/worktree-lifecycle.md)
