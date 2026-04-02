# ワークフロー目録

この文書は、現在の自動化入口と、まだ人手 review が必要な作業を整理するための正本です。
存在しないスクリプトや終了した運用は書きません。

## 自動化済みの入口

- `scripts/setup_worktree.sh`
  - `origin/main` から branch と worktree を作成し、`WORKTREE_SCOPE.md` を配置します。
- `scripts/tools/create_worktree.sh`
  - `setup_worktree.sh` 互換の薄いラッパーです。
- `scripts/ci/run_all_checks.sh`
  - `pytest`、`pyright`、`pydocstyle`、`ruff` を一括実行します。
- `scripts/run_pytest_with_logs.sh`
  - pytest 実行ログを `python/tests/logs/` に保存します。
- `scripts/run_comprehensive_review.sh`
  - 包括 review 用の静的解析、テスト、補助 validator をまとめて実行します。
- `scripts/tools/check_worktree_scopes.sh`
  - worktree ごとの `WORKTREE_SCOPE.md` を確認します。
- `scripts/tools/check_markdown_lint.py`
  - Markdown 体裁を確認します。
- `scripts/tools/audit_and_fix_links.py`
  - Markdown 内リンクを検査・修正します。
- `scripts/tools/fix_markdown_docs.py`
  - Markdown の機械的な修正を行います。
- `scripts/tools/find_similar_documents.py`
  - 類似文書候補を抽出します。

## 人手 review が必要な作業

- 実験結果の採否判断
  - `summary.json`、`cases.jsonl`、report の内容解釈は人が行います。
- worktree を閉じる前の carry-over 抽出
  - `notes/`、`diary/`、最小 result の持ち帰り先判断は人が行います。
- 規約変更の正本反映
  - `documents/` のどこを更新するかは人が決めます。
- 歴史メモや旧 skill 草稿の整理
  - `notes/`、隠し草稿、古い review の削除判断は人が行います。

## いま不足している自動化

- `documents/` と `scripts/README.md` の stale 記述を継続検出する link / path checker
- `notes/` と歴史文書に残った worktree 絶対パスの定期監査
- worktree 削除前に carry-over 先が `main` から参照可能かを確認する checker
- 実験 report の最小必須項目を確認する schema / lint

## 使い分け

- 日常の実装確認は `make ci-quick`
- 仕上げ前の確認は `make ci`
- workflow 全体の点検は `bash scripts/run_comprehensive_review.sh`
- 実験運用は [experiment-workflow.md](/workspace/documents/experiment-workflow.md)
- worktree 運用は [worktree-lifecycle.md](/workspace/documents/worktree-lifecycle.md)
