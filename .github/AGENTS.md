# Agent Team — Coordination Guide

このリポジトリでは、恒久チームの正本を 1 か所に寄せます。

## 正本

- チーム定義と write policy: `agents/agents_config.json`
- agent 間 communication の規約: `agents/COMMUNICATION_PROTOCOL.md`
- runtime 実装: `scripts/agent_tools/agent_team.py`
- 人間向け入口: `agents/README.md`
- task workflow: `agents/TASK_WORKFLOWS.md`
- repo 運用ガイド: `documents/AGENTS_COORDINATION.md`

## 運用の要点

- 常時ロールは `manager`, `manager_reviewer`, `designer`, `design_reviewer`, `implementer`, `change_reviewer`, `final_reviewer`, `verifier`, `auditor`。
- 条件付きロールは `researcher`, `research_reviewer`, `scheduler`, `schedule_reviewer`, `infra_steward`, `infra_reviewer`。
- `manager` が intent 取り込み、scope 固定、specialist activation、permission 判断、handoff 管理を統合して持つ。
- `designer` は coding の前に必須で、`design_reviewer` の review と修正反映を通ってから `implementer` に渡す。
- 各 execution role は reviewer の feedback を受けて修正してから次段へ進む。
- repo を直接編集できるのは `implementer` だけ。
- reviewer 群、researcher 群、scheduler 群、infra 群、verifier、auditor は artifact-only とし、`reports/agents/<run-id>/` 内の自分の成果物だけを更新する。
- `implementer` と reviewer 群は private context を共有しない。

## 実行入口

```bash
python3 scripts/agent_tools/bootstrap_agent_run.py \
  --task "describe the task" \
  --owner "<agent-or-human>" \
  --workspace-root "$PWD"
```

permission を確認したい場合:

```bash
python3 scripts/agent_tools/validate_role_write_scope.py \
  --report-dir reports/agents/<run-id> \
  --workspace-root "$PWD" \
  --report-snapshot-out /tmp/agent-report-before.json \
  --workspace-snapshot-out /tmp/agent-workspace-before.json
```

```bash
python3 scripts/agent_tools/validate_role_write_scope.py \
  --role final_reviewer \
  --report-dir reports/agents/<run-id> \
  --report-snapshot-in /tmp/agent-report-before.json \
  --workspace-snapshot-in /tmp/agent-workspace-before.json \
  --workspace-root "$PWD"
```
