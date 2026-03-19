# コーディレクトリ構成案: Coding Agent 対応

以下は本プロジェクト（現在の `documents/`, `scripts/`, `reports/` 構成を前提）に対して、コーディング Agent 機能を安全かつ運用しやすく追加するためのディレクトリ構成案です。

目的:
- Agent 機能を役割ごとに分離して実装・テスト・監査を容易にする。
- ドキュメントとログを一箇所に集約してトレーサビリティを確保する。

提案構成（リポジトリルートからの相対）:

- `agents/`  <-- Agent 実装とランタイム
  - `agents/README.md`                : Agent の全体設計と運用手順
  - `agents/planner/`                 : タスク分解・計画生成モジュール（dry-run 提案を生成）
  - `agents/executor/`                : 変更適用モジュール（patch 作成、ワークツリー操作）
  - `agents/verifier/`                : テスト・静的解析・安全判定ルール
  - `agents/agents_config.yaml`       : 権限、許容ルール、ブラックリスト／ホワイトリスト設定
  - `agents/logs/`                    : agent-run の証跡（プロンプト、決定、結果、テスト出力）

- `scripts/agent_tools/`  <-- 既存スクリプトを移行/ラップ
  - `fix_markdown_docs.py`
  - `audit_and_fix_links.py`
  - `find_similar_documents.py` / `tfidf_similar_docs.py`
  - `create_worktree.sh`
  - CLI ラッパー: `agents-runner.py`（Agent ワークフローを手動・CI から起動するための統合 CLI）

- `documents/agent/`  <-- 設計・ポリシー・研究ノート
  - `AGENT_RESEARCH.md`
  - `AGENT_DIRECTORY_PROPOSAL.md`
  - `AGENT_USAGE.md` (既存) を移行して拡張

- `reports/agents/`  <-- Agent 実行レポート
  - `reports/agents/<run-id>/`  : 各実行ごとの証跡（diff、logs、tests）

- `.github/workflows/agent-check.yml`  : PR 時に Agent が行った変更を検証する CI（失敗時は PR をブロック）

運用ルール（短い一覧）:

- main には Agent が自動で直接 push しない（ドキュメント更新は例外）。Agent の変更は必ず `refs/heads/agent/*` の一時ブランチを作り PR を作成する。CI を通ったら人の承認でマージする。
- `agents/agents_config.yaml` により、Agent が操作できるパス、許容するファイルタイプ、最大変更行数などを制限する。
- すべての Agent 実行は `reports/agents/<timestamp>-<uuid>/` に出力してアーカイブする。

移行手順（簡易）:

1. `scripts/tools/*` の Agent 関連スクリプトを `scripts/agent_tools/` に整理。
2. `agents/` 下に Planner/Executor/Verifier の最小実装（stub）を置き、既存スクリプトを呼び出すラッパーを作成。
3. CI ワークフロー（`agent-check.yml`）を追加して、Agent が作成したブランチを自動でテストする。

理由:

- 既存の `scripts/tools` は単機能で有用だが、Agent の安全運用には実行履歴・設定・ログを管理する専用領域があると運用上優位です。
