# Agent Team — Coordination Guide

このリポジトリの恒久的なエージェントチーム運用ポリシーです。

正本:

- チーム定義: `agents/README.md`
- 構成: `agents/agents_config.yaml`
- タスク workflow: `agents/TASK_WORKFLOWS.md`
- 運用ガイド: `documents/AGENTS_COORDINATION.md`

## 恒久チームの必須ロール

- `intent_analyst`: ユーザー要求をスコープ、受け入れ条件、曖昧点へ変換する。
- `coordinator`: スコープ固定、役割分担、handoff 管理、エスカレーションを担当する。
- `editor`: スコープ内の実装・文書変更を担当する。
- `change_reviewer`: 実装の各チャンクを逐次レビューし、次チャンクに進めるか判断する。
- `final_reviewer`: editor と独立した最終レビューを行う。
- `verifier`: `scripts/ci/pre_review.sh` や `make ci` を実行し、検証結果を残す。
- `auditor`: 証跡を `reports/agents/<run-id>/` に集約し、クローズアウトを記録する。

## 条件付きの専門ロール

- `researcher`: アルゴリズム調査、外部ドキュメント調査、インターネット検索を担当する。
- `scheduler`: 大規模変更のマイルストーン、依存関係、順序管理を担当する。
- `infra_steward`: CI、Docker、experiment runner、基盤自動化の保守と拡張を担当する。

## コンテキスト分離

- `editor` と `change_reviewer` は private context を共有しない。
- `editor` と `final_reviewer` も private context を共有しない。
- reviewer 側は `intent_brief.md`、`research_notes.md`、`schedule.md`、diff、検証結果のみを見る。

## 標準アーティファクト

各エージェント実行では、最初に以下を作成してください。

```bash
python3 scripts/agent_tools/bootstrap_agent_run.py \
  --task "describe the task" \
  --owner "<agent-or-human>"
```

生成される標準ファイル:

- `reports/agents/<run-id>/intent_brief.md`
- `reports/agents/<run-id>/decision_log.md`
- `reports/agents/<run-id>/review_log.md`
- `reports/agents/<run-id>/team_manifest.yaml`
- `reports/agents/<run-id>/verification.txt`
- `reports/agents/<run-id>/retrospective.md`

## 運用方針

- 各作業はワークツリー単位で行い、`WORKTREE_SCOPE.md` がない場合は開始しない。
- 変更は小さく分割し、handoff ごとに根拠と未解決事項を残す。
- `main` へ直接 push しない。
- Markdown を編集した場合は `mdformat` を実行する。
- close 前に検証を通し、結果を `verification.txt` に残す。

## Reviewer チェックリスト

- [ ] `python -m pyright ./python/jax_util` が通る
- [ ] `python -m pytest python/tests/ -q` が通る
- [ ] `python -m pydocstyle python/jax_util` の結果を確認した
- [ ] `python -m ruff check python/jax_util --select E,F,I,D,UP` の結果を確認した
- [ ] 仕様変更がある場合、関連ドキュメントを更新した
- [ ] スコープ外変更や未解決リスクを `decision_log.md` に残した

再審査が必要な場合は `review:needs-revision` を付与し、改善箇所を具体的に指摘してください。

## 議論と監査

- 重要な意思決定は `decision_log.md` に残す。
- チーム間の議論が必要な場合は `.github/agents/discussion.md` に要点を残す。
- クローズ時には `retrospective.md` を更新する。
