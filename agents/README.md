# Agent System Canon

このディレクトリは、repo 全体で共有する agent workflow と skill の正本です。

**🇯🇵 日本語ガイドが必要ですか？ → [USER_GUIDE_JA.md](./USER_GUIDE_JA.md)**

## Canonical Sources

- Team definition and role permissions: `agents/agents_config.json`
- Inter-agent communication rules: `agents/COMMUNICATION_PROTOCOL.md`
- Runtime implementation used by bootstrap and permission checks: `scripts/agent_tools/agent_team.py`
- Task workflow catalog: `agents/TASK_WORKFLOWS.md`
- Machine-readable task catalog: `agents/task_catalog.yaml`
- Codex-first workflow supplement: `agents/CODEX_WORKFLOWS.md`
- Shared skill canon: `agents/skills/README.md`

Keep detailed role logic in the canonical sources above. Other docs should link back here instead of duplicating role definitions.

## Canonical Layout

- `AGENTS.md` is the repo-wide entrypoint for agentic tools.
- `agents/CODEX_WORKFLOWS.md` defines Codex-first subagent operation.
- `agents/README.md` is the only human-facing summary that may restate the team shape.
- `agents/skills/` is the only canonical documentation surface for shared skills.
- `.agents/skills/` contains thin skill shims for tools that auto-discover `SKILL.md`.
- `.codex/agents/` contains project-scoped Codex custom agents.
- `CLAUDE.md`, `.github/AGENTS.md`, and `.github/copilot-instructions.md` are thin adapters.
- `.github/skills/` contains GitHub-facing automation scripts and implementation entrypoints. It is not the skill documentation canon.

Other docs should link here and to the machine-readable/runtime canon instead of copying role lists or handoff details.

## Team Shape

- Always-on roles: `manager`, `manager_reviewer`, `designer`, `design_reviewer`, `implementer`, `change_reviewer`, `final_reviewer`, `verifier`, `auditor`
- Specialist roles: `researcher`, `research_reviewer`, `experimenter`, `experiment_reviewer`, `report_reviewer`, `scheduler`, `schedule_reviewer`, `infra_steward`, `infra_reviewer`
- `manager` is the integrated control role for intake, scoping, specialist activation, permissions, and escalation.
- `designer` always runs before `implementer`.
- Every execution role has a paired reviewer, and the reviewed role must apply review feedback before handoff proceeds.
- Experiment-driven tasks use an explicit loop: `experimenter -> experiment_reviewer -> report_reviewer -> experimenter / implementer -> reviewers -> experimenter`.
- `experiment_reviewer` critiques comparison fairness, quantitative summaries, and overclaim risk before the next code change is justified.
- `report_reviewer` independently critiques the user-facing report, its evidence traceability, and whether the report needs rewrite, extra validation, or rerun.
- Only `implementer` may modify repository source files.
- `experimenter` may write only run artifacts, experiment reports, and runtime output directories explicitly listed in `WORKTREE_SCOPE.md`.
- `manager`, reviewers, `researcher`, `scheduler`, `infra_steward`, `verifier`, and `auditor` are artifact-only and write only to the run bundle under `reports/agents/<run-id>/`.
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
  --enable experimenter \
  --enable scheduler \
  --workspace-root "$PWD"
```

`--enable researcher` や `--enable experimenter` のように specialist の execution role を有効化すると、paired reviewer も同じ activation group として自動的に含まれます。

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
- Treat Codex as the primary interactive runtime while this repo is in Codex-first operation.
- Keep GitHub Actions and any other automation aligned with this canonical handoff spine when workflow logic changes.
- Treat `agents/agents_config.json` as the single source of truth for roles, handoffs, and write policies.
- Treat `agents/COMMUNICATION_PROTOCOL.md` as the single source of truth for handoff, review, response, and escalation messages.
- Treat `agents/CODEX_WORKFLOWS.md` and `.codex/agents/*.toml` as the single source of truth for Codex subagent usage.
- Treat `agents/skills/` as the single source of truth for shared skill descriptions and usage boundaries.
- Treat this file as the only human-facing summary of role lists and team shape.
- Keep `AGENTS.md`, `CLAUDE.md`, `documents/AGENTS_COORDINATION.md`, `.github/AGENTS.md`, and `.github/copilot-instructions.md` as thin entrypoints that link here instead of repeating the team definition.
- Keep repo edits inside `WORKTREE_SCOPE.md` editable directories whenever `implementer` is active.
- Keep experiment outputs inside `WORKTREE_SCOPE.md` runtime output directories whenever `experimenter` is active.
- Capture both a report-dir snapshot and a workspace-change snapshot before an artifact-only role runs, then validate against both after the role writes.
- Record scope, risk, and acceptance decisions in the report bundle.
- `Research-Driven Change` と実験結果を伴う依頼では、`critical-review` を通さずに完了扱いにしません。
- `experiments/report/<run_name>.md` を生成する依頼では、`report-review` を通さずに完了扱いにしません。
- repo-wide な文書整理や workflow 改造では、変更後 review を通さずに完了扱いにしません。

## Template Library

`agents/templates/` contains **18 standardized templates** used directly by `bootstrap_agent_run.py` and related runtime tools. Templates are **NOT referenced in documentation** by design—they are discovered and used dynamically by execution tools.

**Template Inventory by Role:**

| Template | Role | Purpose |
|----------|------|---------|
| `intent_brief.md` | manager | Task intake and scoping |
| `design_brief.md` | designer | Design proposal brief |
| `design_review.md` | design_reviewer | Design feedback and approval |
| `research_notes.md` | researcher | Research findings and analysis |
| `research_review.md` | research_reviewer | Research quality and fairness critique |
| `experiment_log.md` | experimenter | Experiment setup, metrics, and execution log |
| `experiment_report.md` | experimenter | Results, analysis, and conclusions |
| `experiment_review.md` | experiment_reviewer | Quantitative rigor and overclaim check |
| `report_review.md` | report_reviewer | User-facing report quality and evidence traceability check |
| `change_review.md` | change_reviewer | Code quality and integration checklist |
| `final_review.md` | final_reviewer | Go/no-go decision and sign-off |
| `schedule.md` | scheduler | Timeline, dependencies, risk mitigation |
| `schedule_review.md` | schedule_reviewer | Schedule feasibility and contingencies |
| `infra_notes.md` | infra_steward | Infrastructure changes and rationale |
| `infra_review.md` | infra_reviewer | Infrastructure impact assessment |
| `management_review.md` | manager | Final integration and release notes |
| `decision_log.md` | manager | Decision audit trail and context |
| `retrospective.md` | manager | Post-execution lessons and next steps |

**Usage:** Templates are loaded by name at runtime. No external documentation or cross-references needed.
