# コーディレクトリ構成案: Coding Agent 対応（詳細版）

以下は本プロジェクト（現在の `documents/`, `scripts/`, `reports/` 構成を前提）に対して、コーディング Agent 機能を安全かつ運用しやすく追加するためのディレクトリ構成案です。

目的:
- Agent 機能を役割ごとに分離して実装・テスト・監査を容易にする。
- ドキュメントとログを一箇所に集約してトレーサビリティを確保する。

提案構成（リポジトリルートからの相対）:

- `agents/`  <-- Agent 実装とランタイム
  - `agents/README.md`                : Agent の全体設計と運用手順
  - `agents/planner/`                 : タスク分解・探索順決定・実行計画生成
  - `agents/executor/`                : 変更適用（差分編集、worktree 操作）
  - `agents/verifier/`                : 静的解析・テスト・安全判定
  - `agents/retriever/`               : ファイル探索順制御（ランキング、候補絞り込み）
  - `agents/policies/`                : 変更可能パス、禁止操作、承認条件
  - `agents/agents_config.yaml`       : 共通設定（max steps, timeout, allowed roots）
  - `agents/logs/`                    : 実行ログ（イベント時系列）

- `scripts/agent_tools/`  <-- 既存スクリプトを移行/ラップ
  - `fix_markdown_docs.py`
  - `audit_and_fix_links.py`
  - `find_similar_documents.py` / `tfidf_similar_docs.py`
  - `create_worktree.sh`
  - CLI ラッパー: `agents-runner.py`（Agent ワークフローを手動・CI から起動）
  - `rank_target_files.py`（探索優先度スコア計算）

- `documents/agent/`  <-- 設計・ポリシー・研究ノート
  - `AGENT_RESEARCH.md`
  - `AGENT_DIRECTORY_PROPOSAL.md`
  - `AGENT_FILE_EXPLORATION_POLICY.md`
  - `AGENT_USAGE.md` (既存) を移行して拡張

- `reports/agents/`  <-- Agent 実行レポート
  - `reports/agents/<run-id>/decision_log.md`
  - `reports/agents/<run-id>/edits.patch`
  - `reports/agents/<run-id>/verification.txt`
  - `reports/agents/<run-id>/retrospective.md`

- `.github/workflows/agent-check.yml`  : PR 時に Agent が行った変更を検証する CI（失敗時は PR をブロック）

運用ルール（短い一覧）:

- main には Agent が自動で直接 push しない（ドキュメント更新は例外）。Agent の変更は `refs/heads/agent/*` ブランチで PR 化する。
- `agents/agents_config.yaml` により、Agent が操作できるパス、許容するファイルタイプ、最大変更行数などを制限する。
- すべての Agent 実行は `reports/agents/<timestamp>-<uuid>/` に出力してアーカイブする。

探索順ポリシー（要約）:

1. スコープ文書（`WORKTREE_SCOPE.md` / issue）
2. 規約（`documents/`）
3. 設計（`documents/design/`）
4. 実装の定義側（`python/**/__init__.py`, `core.py`）
5. 利用側（`tests/`, 呼び出し元）

この順序を固定すると、探索の迷走が減ります。

移行手順（簡易）:

1. `scripts/tools/*` の Agent 関連スクリプトを `scripts/agent_tools/` に整理。
2. `agents/` 下に `planner/executor/verifier/retriever` の最小実装（stub）を作る。
3. `retriever` に探索優先度ランキングを実装する。
4. CI ワークフロー（`agent-check.yml`）を追加して、Agent ブランチを自動検証する。

理由:

- 既存の `scripts/tools` は単機能で有用だが、Agent の安全運用には実行履歴・設定・ログを管理する専用領域があると運用上優位です。
