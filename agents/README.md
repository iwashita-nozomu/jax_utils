# Permanent Agent Team

This directory defines the repo's long-lived agent team.

## Canonical Sources

- Team definition and role permissions: `agents/agents_config.json`
- Inter-agent communication rules: `agents/COMMUNICATION_PROTOCOL.md`
- Runtime implementation used by bootstrap and permission checks: `scripts/agent_tools/agent_team.py`
- Task workflow catalog: `agents/TASK_WORKFLOWS.md`
- Machine-readable task catalog: `agents/task_catalog.yaml`

Keep detailed role logic in the canonical sources above. Other docs should link back here instead of duplicating role definitions.

## Team Shape

- Always-on roles: `manager`, `manager_reviewer`, `designer`, `design_reviewer`, `implementer`, `change_reviewer`, `final_reviewer`, `verifier`, `auditor`
- Specialist roles: `researcher`, `research_reviewer`, `scheduler`, `schedule_reviewer`, `infra_steward`, `infra_reviewer`
- `manager` is the integrated control role for intake, scoping, specialist activation, permissions, and escalation.
- `designer` always runs before `implementer`.
- Every execution role has a paired reviewer, and the reviewed role must apply review feedback before handoff proceeds.
- Only `implementer` may modify repository files.
- Every other role is artifact-only and writes only to the run bundle under `reports/agents/<run-id>/`.
- Execution roles and their paired reviewers remain isolated from each other's private working context.

## Standard Commands

```bash
python3 scripts/agent_tools/bootstrap_agent_run.py \
  --task "describe the task" \
  --owner "<agent-or-human>" \
  --workspace-root "$PWD"
```

```bash
python3 scripts/agent_tools/bootstrap_agent_run.py \
  --task "large algorithm change" \
  --owner "<agent-or-human>" \
  --enable researcher \
  --enable scheduler \
  --workspace-root "$PWD"
```

`--enable researcher` のように specialist の execution role を有効化すると、paired reviewer も同じ activation group として自動的に含まれます。

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

## Operating Rules

- Reuse this same team shape across Codex, GitHub Actions, and any other agent runtime.
- GitHub Actions models the handoff spine with reviewer-return loops and can activate specialists through workflow inputs when a run needs them.
- Treat `agents/agents_config.json` as the single source of truth for roles, handoffs, and write policies.
- Treat `agents/COMMUNICATION_PROTOCOL.md` as the single source of truth for handoff, review, response, and escalation messages.
- Keep repo edits inside `WORKTREE_SCOPE.md` editable directories whenever `implementer` is active.
- Capture both a report-dir snapshot and a workspace-change snapshot before an artifact-only role runs, then validate against both after the role writes.
- Record scope, risk, and acceptance decisions in the report bundle.
