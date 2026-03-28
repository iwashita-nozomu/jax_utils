# エージェントチーム運用と役割分担

目的: 恒久的なエージェントチームを、毎回同じ入口と同じ権限モデルで運用すること。

## 正本

- チーム定義と role permission: `agents/agents_config.json`
- agent 間やり取りの規約: `agents/COMMUNICATION_PROTOCOL.md`
- runtime 実装: `scripts/agent_tools/agent_team.py`
- 人間向け入口: `agents/README.md`
- task workflow カタログ: `agents/TASK_WORKFLOWS.md`

この文書は運用の入口に留め、詳細な role 定義は正本へ集約する。

agent 間の handoff、review、response、escalation の書き方は `agents/COMMUNICATION_PROTOCOL.md` を参照する。

## 現在のチーム構成

- Always-on: `manager`, `manager_reviewer`, `designer`, `design_reviewer`, `implementer`, `change_reviewer`, `final_reviewer`, `verifier`, `auditor`
- Specialists: `researcher`, `research_reviewer`, `scheduler`, `schedule_reviewer`, `infra_steward`, `infra_reviewer`
- `manager` が intent、scope、specialist activation、permission 判断、handoff を統合管理する
- `designer` が coding 前の設計を担当し、`design_reviewer` の review と修正反映を通す
- 各 execution role は reviewer の feedback を受けて修正してから次段へ進む
- repo を直接編集できるのは `implementer` だけ
- それ以外の role は artifact-only で、`reports/agents/<run-id>/` の自分の成果物だけを更新する

## 基本フロー

1. `bootstrap_agent_run.py` で run bundle を作る。
1. `manager` が `intent_brief.md`、scope、specialist role を確定する。
1. `manager_reviewer` が intake を review し、`manager` が feedback を反映する。
1. 必要なら `researcher` / `research_reviewer`、`scheduler` / `schedule_reviewer`、`infra_steward` / `infra_reviewer` を順に通す。
1. `designer` が設計を作り、`design_reviewer` が review し、`designer` が feedback を反映する。
1. `implementer` が `WORKTREE_SCOPE.md` の editable directories 内で実装する。
1. `change_reviewer` が chunk review を行い、`implementer` が feedback を反映する。
1. `final_reviewer` が独立レビューを行い、`implementer` が必要なら最後の修正を行う。
1. `verifier` が quality gate を実行する。
1. `auditor` が closeout を `retrospective.md` に残す。

## コマンド

```bash
python3 scripts/agent_tools/bootstrap_agent_run.py \
  --task "<task>" \
  --owner "<owner>" \
  --workspace-root "$PWD"
```

```bash
python3 scripts/agent_tools/validate_role_write_scope.py \
  --report-dir reports/agents/<run-id> \
  --workspace-root "$PWD" \
  --report-snapshot-out /tmp/agent-report-before.json \
  --workspace-snapshot-out /tmp/agent-workspace-before.json
```

```bash
python3 scripts/agent_tools/validate_role_write_scope.py \
  --role change_reviewer \
  --report-dir reports/agents/<run-id> \
  --report-snapshot-in /tmp/agent-report-before.json \
  --workspace-snapshot-in /tmp/agent-workspace-before.json \
  --workspace-root "$PWD"
```

`pre_review.sh` では、必要なら以下で verifier の write scope も検査できる。

```bash
AGENT_REPORT_DIR=reports/agents/<run-id> \
AGENT_ROLE=verifier \
AGENT_ENFORCE_WRITE_SCOPE=1 \
bash scripts/ci/pre_review.sh
```

## 運用ルール

- `agents/agents_config.json` を role 定義の単一正本として扱う。
- `scripts/agent_tools/agent_team.py` を runtime 実装の単一正本として扱う。
- role 定義を他文書へコピーしない。必要ならリンクする。
- GitHub Actions では reviewer-return loop を含む handoff spine を通し、specialist role も workflow input から有効化できるようにする。
- `implementer` 以外が repo ファイルを触る workflow は作らない。
- artifact-only role を検査するときは、role 実行前の report snapshot と workspace snapshot の両方を取り、role 実行後の差分で write scope を判定する。
- `WORKTREE_SCOPE.md` がある作業では、repo 変更を editable directories に限定する。
