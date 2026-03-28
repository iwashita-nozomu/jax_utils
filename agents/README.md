# Permanent Agent Team

This directory is the source of truth for the repo's long-lived agent team.

## Purpose

- Keep the same long-lived agent team available for every task.
- Reuse the same team shape across Codex, GitHub Actions, and any other agent runtime.
- Make agent runs reproducible.
- Store evidence for every run under `reports/agents/`.

## Always-On Roles

- `intent_analyst`: converts the user request into scope, acceptance criteria, and ambiguity notes.
- `coordinator`: owns planning, delegation, cross-role handoffs, and escalation.
- `editor`: owns code or document changes inside the approved scope.
- `change_reviewer`: reviews each implementation chunk or milestone before the next chunk proceeds.
- `final_reviewer`: performs an independent final review before verification.
- `verifier`: owns automated checks and confirms that required gates ran.
- `auditor`: owns the final evidence bundle and closeout record.

## Specialist Roles

- `researcher`: looks up algorithms, external APIs, papers, and internet references when local context is not enough.
- `scheduler`: manages milestones and dependency tracking for large or multi-stage changes.
- `infra_steward`: owns CI, Docker, experiment runners, automation, and infra expansion work.

## Context Isolation

- `editor`, `change_reviewer`, and `final_reviewer` must not share private scratchpads or hidden working context.
- Reviewers only consume approved artifacts such as `intent_brief.md`, `research_notes.md`, `schedule.md`, diffs, and test evidence.
- This separation is intentional so that review remains independent.

## Activation Guide

- Enable `researcher` when algorithm choice, web research, or external verification is required.
- Enable `scheduler` for large refactors, multi-module edits, or any task that needs milestone tracking.
- Enable `infra_steward` for `experiment_runner`, CI, Docker, automation, or platform expansion work.

## Standard Run Flow

1. Bootstrap a run directory.
1. Capture user intent and acceptance criteria.
1. Enable specialist roles if the task needs research, scheduling, or infra stewardship.
1. Execute the scoped change.
1. Review incrementally, then run an independent final review.
1. Verify the change and save evidence.
1. Close out with a retrospective.

## Bootstrap Command

```bash
python3 scripts/agent_tools/bootstrap_agent_run.py \
  --task "describe the task" \
  --owner "<agent-or-human>"
```

```bash
python3 scripts/agent_tools/bootstrap_agent_run.py \
  --task "large algorithm change" \
  --owner "<agent-or-human>" \
  --enable researcher \
  --enable scheduler
```

```bash
python3 scripts/agent_tools/bootstrap_agent_run.py \
  --task "infra expansion" \
  --owner "<agent-or-human>" \
  --full-team
```

The command creates:

- `reports/agents/<run-id>/intent_brief.md`
- `reports/agents/<run-id>/decision_log.md`
- `reports/agents/<run-id>/review_log.md`
- `reports/agents/<run-id>/team_manifest.yaml`
- `reports/agents/<run-id>/verification.txt`
- `reports/agents/<run-id>/retrospective.md`

When specialist roles are enabled, it also creates:

- `reports/agents/<run-id>/research_notes.md`
- `reports/agents/<run-id>/schedule.md`
- `reports/agents/<run-id>/infra_notes.md`

## Canonical Files

- Team config: `agents/agents_config.yaml`
- Task workflow catalog: `agents/TASK_WORKFLOWS.md`
- Machine-readable task catalog: `agents/task_catalog.yaml`
- GitHub-facing policy: `.github/AGENTS.md`
- Repo coordination guide: `documents/AGENTS_COORDINATION.md`
- Run artifacts: `reports/agents/<run-id>/`

## Operating Rules

- Use this same team structure even when a different agent implementation is used.
- Keep reviewers independent from the editor's private context.
- Do not push directly to `main`.
- Keep work scoped to a worktree or explicitly approved branch.
- Run verification before handoff or closeout.
- Record decisions that affect scope, risk, or acceptance criteria.
