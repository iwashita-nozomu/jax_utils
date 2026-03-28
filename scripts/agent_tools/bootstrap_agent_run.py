#!/usr/bin/env python3
"""Bootstrap a persistent agent-team run directory."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_ROOT = ROOT / "reports" / "agents"
TEMPLATE_ROOT = ROOT / "agents" / "templates"


@dataclass(frozen=True)
class Role:
    """Describe one permanent team role."""

    name: str
    owns: tuple[str, ...]
    required_outputs: tuple[str, ...]
    activation: str = "always"


ALWAYS_ON_ROLES = (
    Role(
        "intent_analyst",
        ("user_intent_capture", "acceptance_criteria", "ambiguity_tracking"),
        ("intent_brief.md",),
    ),
    Role(
        "coordinator",
        ("planning", "delegation", "escalation", "cross_role_handoffs"),
        ("team_manifest.yaml", "decision_log.md"),
    ),
    Role(
        "editor",
        ("code_and_docs_edits", "local_validation"),
        ("decision_log.md",),
    ),
    Role(
        "change_reviewer",
        ("milestone_review", "regression_review", "design_guardrails"),
        ("review_log.md",),
    ),
    Role(
        "final_reviewer",
        ("final_independent_review", "ship_blockers", "risk_summary"),
        ("review_log.md",),
    ),
    Role(
        "verifier",
        ("ci_execution", "quality_gates"),
        ("verification.txt",),
    ),
    Role(
        "auditor",
        ("evidence_capture", "closeout", "retrospective"),
        ("retrospective.md",),
    ),
)

SPECIALIST_ROLES = (
    Role(
        "researcher",
        ("external_research", "algorithm_survey", "source_collection"),
        ("research_notes.md",),
        activation="optional",
    ),
    Role(
        "scheduler",
        ("milestone_plan", "dependency_tracking", "delivery_checkpoints"),
        ("schedule.md",),
        activation="optional",
    ),
    Role(
        "infra_steward",
        ("experiment_runner", "ci_cd", "docker", "automation"),
        ("infra_notes.md",),
        activation="optional",
    ),
)

SPECIALIST_ROLE_NAMES = tuple(role.name for role in SPECIALIST_ROLES)

ARTIFACT_TEMPLATES = {
    "intent_brief.md": "intent_brief.md",
    "decision_log.md": "decision_log.md",
    "review_log.md": "review_log.md",
    "retrospective.md": "retrospective.md",
    "research_notes.md": "research_notes.md",
    "schedule.md": "schedule.md",
    "infra_notes.md": "infra_notes.md",
}

CONTEXT_POLICIES = (
    {
        "roles": ("editor", "change_reviewer"),
        "mode": "isolated",
        "share_only": (
            "intent_brief.md",
            "research_notes.md",
            "schedule.md",
            "review_log.md",
            "diff",
            "verification.txt",
        ),
        "do_not_share": (
            "private_scratchpads",
            "hidden_chain_of_thought",
            "direct_session_memory",
        ),
    },
    {
        "roles": ("editor", "final_reviewer"),
        "mode": "isolated",
        "share_only": (
            "intent_brief.md",
            "research_notes.md",
            "schedule.md",
            "review_log.md",
            "diff",
            "verification.txt",
        ),
        "do_not_share": (
            "private_scratchpads",
            "hidden_chain_of_thought",
            "direct_session_memory",
        ),
    },
)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(
        description="Create a standard reports/agents/<run-id>/ bundle for one agent-team run."
    )
    parser.add_argument("--task", required=True, help="Short task description for the run.")
    parser.add_argument("--owner", required=True, help="Human or agent responsible for the run.")
    parser.add_argument("--run-id", help="Optional explicit run id. Defaults to a timestamped slug.")
    parser.add_argument(
        "--enable",
        action="append",
        choices=SPECIALIST_ROLE_NAMES,
        default=[],
        help="Enable a specialist role. Repeat the flag to enable multiple roles.",
    )
    parser.add_argument(
        "--full-team",
        action="store_true",
        help="Enable every specialist role for this run.",
    )
    parser.add_argument(
        "--report-root",
        default=str(DEFAULT_REPORT_ROOT),
        help="Directory that will contain per-run report folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the run id and paths without writing files.",
    )
    return parser


def slugify(value: str) -> str:
    """Return an ASCII slug that is safe for file paths."""
    ascii_only = value.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_only).strip("-")
    return slug or "task"


def make_run_id(task: str, created_at: datetime) -> str:
    """Build a stable default run id."""
    timestamp = created_at.strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{slugify(task)[:40]}"


def render_template(template_name: str, replacements: dict[str, str]) -> str:
    """Load and fill a text template from agents/templates."""
    content = (TEMPLATE_ROOT / template_name).read_text(encoding="utf-8")
    for key, value in replacements.items():
        content = content.replace(f"{{{{{key}}}}}", value)
    return content


def select_roles(enabled_specialists: list[str], full_team: bool) -> tuple[Role, ...]:
    """Return the active roles for this run."""
    if full_team:
        return ALWAYS_ON_ROLES + SPECIALIST_ROLES
    enabled_set = set(enabled_specialists)
    selected_specialists = tuple(role for role in SPECIALIST_ROLES if role.name in enabled_set)
    return ALWAYS_ON_ROLES + selected_specialists


def iter_artifacts(roles: tuple[Role, ...]) -> tuple[str, ...]:
    """Return unique artifact filenames in deterministic order."""
    ordered_artifacts: list[str] = []
    for role in roles:
        for output in role.required_outputs:
            if output not in ordered_artifacts:
                ordered_artifacts.append(output)
    ordered_artifacts.extend(["team_manifest.yaml", "verification.txt"])
    unique_artifacts: list[str] = []
    for artifact in ordered_artifacts:
        if artifact not in unique_artifacts:
            unique_artifacts.append(artifact)
    return tuple(unique_artifacts)


def build_manifest(
    run_id: str,
    task: str,
    owner: str,
    created_at_iso: str,
    report_dir: Path,
    roles: tuple[Role, ...],
) -> str:
    """Build the team manifest yaml."""
    lines = [
        "run:",
        f"  id: {run_id}",
        f"  task: {task!r}",
        f"  owner: {owner!r}",
        f"  created_at_utc: {created_at_iso}",
        f"  report_dir: {str(report_dir)!r}",
        f"  team_config: {str(ROOT / 'agents' / 'agents_config.yaml')!r}",
        f"  task_catalog: {str(ROOT / 'agents' / 'task_catalog.yaml')!r}",
        "roles:",
    ]
    for role in roles:
        lines.append(f"  - id: {role.name}")
        lines.append(f"    activation: {role.activation}")
        lines.append("    status: pending")
        lines.append("    owns:")
        for responsibility in role.owns:
            lines.append(f"      - {responsibility}")
        lines.append("    required_outputs:")
        for output in role.required_outputs:
            lines.append(f"      - {output}")
    lines.append("context_policies:")
    for policy in CONTEXT_POLICIES:
        lines.append("  - roles:")
        for role_name in policy["roles"]:
            lines.append(f"      - {role_name}")
        lines.append(f"    mode: {policy['mode']}")
        lines.append("    share_only:")
        for artifact in policy["share_only"]:
            lines.append(f"      - {artifact}")
        lines.append("    do_not_share:")
        for artifact in policy["do_not_share"]:
            lines.append(f"      - {artifact}")
    lines.extend(
        [
            "quality_gates:",
            "  - scripts/ci/pre_review.sh",
            "  - make ci",
        ]
    )
    lines.extend(
        [
            "artifacts:",
        ]
    )
    for artifact in iter_artifacts(roles):
        lines.append(f"  - {artifact}")
    return "\n".join(lines) + "\n"


def create_run_bundle(
    report_dir: Path,
    run_id: str,
    task: str,
    owner: str,
    created_at_iso: str,
    roles: tuple[Role, ...],
) -> tuple[str, ...]:
    """Create the standard files for a run."""
    replacements = {
        "RUN_ID": run_id,
        "TASK": task,
        "OWNER": owner,
        "CREATED_AT": created_at_iso,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    created_files = list(iter_artifacts(roles))
    for artifact in created_files:
        if artifact in ARTIFACT_TEMPLATES:
            (report_dir / artifact).write_text(
                render_template(ARTIFACT_TEMPLATES[artifact], replacements),
                encoding="utf-8",
            )
    (report_dir / "team_manifest.yaml").write_text(
        build_manifest(run_id, task, owner, created_at_iso, report_dir, roles),
        encoding="utf-8",
    )
    (report_dir / "verification.txt").write_text(
        "\n".join(
            [
                f"run_id={run_id}",
                f"task={task}",
                f"owner={owner}",
                f"created_at_utc={created_at_iso}",
                "status=pending",
                "",
            ]
        ),
        encoding="utf-8",
    )
    created_files.extend(["team_manifest.yaml", "verification.txt"])
    unique_created_files: list[str] = []
    for artifact in created_files:
        if artifact not in unique_created_files:
            unique_created_files.append(artifact)
    return tuple(unique_created_files)


def main() -> int:
    """Run the bootstrap command."""
    args = build_parser().parse_args()
    created_at = datetime.now(timezone.utc).replace(microsecond=0)
    created_at_iso = created_at.isoformat().replace("+00:00", "Z")
    report_root = Path(args.report_root).resolve()
    run_id = args.run_id or make_run_id(args.task, created_at)
    report_dir = report_root / run_id
    roles = select_roles(args.enable, args.full_team)
    created_files = ()

    if not args.dry_run:
        created_files = create_run_bundle(report_dir, run_id, args.task, args.owner, created_at_iso, roles)

    print(f"RUN_ID={run_id}")
    print(f"REPORT_DIR={report_dir}")
    if args.dry_run:
        print("DRY_RUN=1")
    else:
        print(f"ACTIVE_ROLES={','.join(role.name for role in roles)}")
        print(f"CREATED_FILES={','.join(created_files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
