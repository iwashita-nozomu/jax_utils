#!/usr/bin/env python3
"""Run repeatable Smolyak research loops and write standalone review notes."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_RESULTS_ROOT = SCRIPT_DIR / "results" / "research_loops"
DEFAULT_NOTES_ROOT = WORKSPACE_ROOT / "notes" / "experiments" / "loops"


def _dimension_csv(start: int = 1, stop: int = 50) -> str:
    return ",".join(str(value) for value in range(start, stop + 1))


def _level_csv(stop: int, start: int = 1) -> str:
    return ",".join(str(value) for value in range(start, stop + 1))


@dataclass(frozen=True)
class LoopSpec:
    loop_id: int
    phase: str
    family: str
    dtypes: str
    dimensions: str
    levels: str
    requested_modes: str
    chunk_sizes: str
    batch_size: int
    timeout_seconds: int
    title: str
    rationale: str


def _build_loop_specs() -> list[LoopSpec]:
    families = [
        ("gaussian", "smooth isotropic baseline"),
        ("shifted_anisotropic_gaussian", "smooth but off-center and anisotropic"),
        ("shifted_laplace_product", "non-smooth cusp product"),
        ("balanced_exponential", "zero-mean cancellation stress"),
    ]
    specs: list[LoopSpec] = []
    loop_id = 1

    for family, rationale in families:
        for level_cap in (4, 6, 8, 10, 12):
            specs.append(
                LoopSpec(
                    loop_id=loop_id,
                    phase="frontier",
                    family=family,
                    dtypes="float64",
                    dimensions=_dimension_csv(1, 50),
                    levels=_level_csv(level_cap),
                    requested_modes="auto,points,indexed,batched",
                    chunk_sizes="16384",
                    batch_size=32,
                    timeout_seconds=120,
                    title=f"{family} frontier to level {level_cap}",
                    rationale=f"Map the all-mode frontier for {rationale}.",
                )
            )
            loop_id += 1

    for family, rationale in families:
        for chunk_size in (4096, 8192, 16384, 32768, 65536):
            specs.append(
                LoopSpec(
                    loop_id=loop_id,
                    phase="chunk_sensitivity",
                    family=family,
                    dtypes="float64",
                    dimensions=_dimension_csv(1, 50),
                    levels=_level_csv(10),
                    requested_modes="auto,indexed,batched",
                    chunk_sizes=str(chunk_size),
                    batch_size=32,
                    timeout_seconds=120,
                    title=f"{family} chunk-size sensitivity c={chunk_size}",
                    rationale=f"Check whether chunking rather than quadrature growth limits {rationale}.",
                )
            )
            loop_id += 1

    for family, rationale in families:
        for batch_size in (8, 16, 32, 64, 128):
            specs.append(
                LoopSpec(
                    loop_id=loop_id,
                    phase="batch_sensitivity",
                    family=family,
                    dtypes="float64",
                    dimensions=_dimension_csv(1, 50),
                    levels=_level_csv(10),
                    requested_modes="auto,indexed,batched",
                    chunk_sizes="16384",
                    batch_size=batch_size,
                    timeout_seconds=120,
                    title=f"{family} batch-size sensitivity b={batch_size}",
                    rationale=f"Measure GPU utilization and throughput scaling for {rationale}.",
                )
            )
            loop_id += 1

    for family, rationale in families:
        for level_cap in (4, 6, 8, 10, 12):
            specs.append(
                LoopSpec(
                    loop_id=loop_id,
                    phase="float32_stress",
                    family=family,
                    dtypes="float32",
                    dimensions=_dimension_csv(1, 50),
                    levels=_level_csv(level_cap),
                    requested_modes="auto,indexed,batched",
                    chunk_sizes="16384",
                    batch_size=32,
                    timeout_seconds=120,
                    title=f"{family} float32 stress to level {level_cap}",
                    rationale=f"Check precision loss and mode stability for {rationale}.",
                )
            )
            loop_id += 1

    for family, rationale in families:
        for level_cap in (11, 12, 13, 14, 15):
            specs.append(
                LoopSpec(
                    loop_id=loop_id,
                    phase="high_level_push",
                    family=family,
                    dtypes="float64",
                    dimensions=_dimension_csv(20, 50),
                    levels=_level_csv(level_cap, 1),
                    requested_modes="auto,indexed,batched",
                    chunk_sizes="16384",
                    batch_size=64,
                    timeout_seconds=180,
                    title=f"{family} high-level push to level {level_cap}",
                    rationale=(
                        f"Push the integrator toward the 50D level-{level_cap} target on {rationale}."
                    ),
                )
            )
            loop_id += 1

    if len(specs) != 100:
        raise AssertionError(f"Expected 100 loop specs, found {len(specs)}.")
    return specs


LOOP_SPECS = _build_loop_specs()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Smolyak research loops and emit per-loop review notes.",
    )
    parser.add_argument("--loop-ids", default="1", help="Comma-separated loop ids to run.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--notes-root", default=str(DEFAULT_NOTES_ROOT))
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--override-dimensions", default=None)
    parser.add_argument("--override-levels", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _parse_csv_ints(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def _run_json_command(command: list[str], *, cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with code {completed.returncode}:\n"
            f"cmd={' '.join(command)}\nstdout={completed.stdout[-4000:]}\nstderr={completed.stderr[-4000:]}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Command did not emit valid JSON:\ncmd={' '.join(command)}\nstdout={completed.stdout[-4000:]}"
        ) from exc


def _summarize_results_jsonl(jsonl_path: Path) -> dict[str, Any]:
    records = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    success_records = [record for record in records if record.get("status") == "success"]
    actual_mode_counts: dict[str, int] = {}
    for record in success_records:
        smolyak = record.get("smolyak", {})
        actual_mode = smolyak.get("actual_mode") if isinstance(smolyak, dict) else None
        if actual_mode is None:
            continue
        actual_mode_counts[str(actual_mode)] = actual_mode_counts.get(str(actual_mode), 0) + 1

    failure_counts: dict[str, int] = {}
    for record in records:
        if record.get("status") == "success":
            continue
        failure_kind = str(record.get("failure_kind") or "error")
        failure_counts[failure_kind] = failure_counts.get(failure_kind, 0) + 1

    return {
        "cases_recorded": len(records),
        "cases_succeeded": len(success_records),
        "cases_failed": len(records) - len(success_records),
        "actual_mode_counts": actual_mode_counts,
        "failure_counts": failure_counts,
    }


def _latest_report_dir(loop_root: Path) -> Path | None:
    report_dirs = sorted(path for path in loop_root.glob("report_*") if path.is_dir())
    return report_dirs[-1] if report_dirs else None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _frontier_snapshot(frontier_csv_path: Path, requested_mode: str) -> dict[str, Any] | None:
    rows = _read_csv_rows(frontier_csv_path)
    matching = [row for row in rows if row.get("requested_mode") == requested_mode]
    if not matching:
        return None
    highest_level = max(int(row["level"]) for row in matching)
    highest_rows = [row for row in matching if int(row["level"]) == highest_level]
    best_row = max(
        highest_rows,
        key=lambda row: -1 if not row.get("max_success_dimension") else int(row["max_success_dimension"]),
    )
    return {
        "highest_level": highest_level,
        "max_success_dimension": (
            None if not best_row.get("max_success_dimension") else int(best_row["max_success_dimension"])
        ),
        "first_failure_dimension": (
            None if not best_row.get("first_failure_dimension") else int(best_row["first_failure_dimension"])
        ),
        "row": best_row,
    }


def _best_auto_compare_target(frontier_csv_path: Path) -> tuple[int, int] | None:
    snapshot = _frontier_snapshot(frontier_csv_path, "auto")
    if snapshot is None:
        return None
    max_dim = snapshot["max_success_dimension"]
    highest_level = snapshot["highest_level"]
    if max_dim is None:
        return None
    return int(max_dim), int(highest_level)


def _run_matrix_loop(
    spec: LoopSpec,
    *,
    results_root: Path,
    workers_per_gpu: int,
    override_dimensions: str | None,
    override_levels: str | None,
) -> dict[str, Any]:
    loop_root = results_root / f"loop_{spec.loop_id:03d}_{spec.family}_{spec.phase}"
    dimensions = spec.dimensions if override_dimensions is None else override_dimensions
    levels = spec.levels if override_levels is None else override_levels
    command = [
        "python3",
        "-m",
        "experiments.smolyak_experiment.run_smolyak_mode_matrix",
        "--platform",
        "gpu",
        "--dimensions",
        dimensions,
        "--levels",
        levels,
        "--dtypes",
        spec.dtypes,
        "--families",
        spec.family,
        "--requested-modes",
        spec.requested_modes,
        "--chunk-sizes",
        spec.chunk_sizes,
        "--batch-size",
        str(spec.batch_size),
        "--warm-repeats",
        "1",
        "--timeout-seconds",
        str(spec.timeout_seconds),
        "--workers-per-gpu",
        str(workers_per_gpu),
        "--quiet",
        "--output-dir",
        str(loop_root),
    ]
    completed = subprocess.run(
        command,
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Command did not emit valid JSON:\ncmd={' '.join(command)}\nstdout={completed.stdout[-4000:]}"
            ) from exc
        payload["matrix_summary"] = json.loads(Path(payload["summary_json"]).read_text(encoding="utf-8"))
        payload["partial"] = False
        payload["matrix_returncode"] = 0
    else:
        loop_report_dir = _latest_report_dir(loop_root)
        if loop_report_dir is None:
            raise RuntimeError(
                f"Command failed with code {completed.returncode} and produced no report directory:\n"
                f"cmd={' '.join(command)}\nstdout={completed.stdout[-4000:]}\nstderr={completed.stderr[-4000:]}"
            )
        jsonl_path = loop_report_dir / "results.jsonl"
        if not jsonl_path.exists():
            raise RuntimeError(
                f"Command failed with code {completed.returncode} and no JSONL was found to salvage:\n"
                f"cmd={' '.join(command)}\nstdout={completed.stdout[-4000:]}\nstderr={completed.stderr[-4000:]}"
            )
        report_payload = _run_json_command(
            [
                "python3",
                "-m",
                "experiments.smolyak_experiment.report_smolyak_mode_matrix",
                "--jsonl-path",
                str(jsonl_path),
                "--output-dir",
                str(loop_report_dir),
            ],
            cwd=WORKSPACE_ROOT,
        )
        payload = {
            "run_dir": str(loop_report_dir),
            "jsonl_path": str(jsonl_path),
            "summary_json": None,
            "report_payload": report_payload,
            "matrix_summary": _summarize_results_jsonl(jsonl_path),
            "partial": True,
            "matrix_returncode": completed.returncode,
            "matrix_stdout_tail": completed.stdout[-4000:],
            "matrix_stderr_tail": completed.stderr[-4000:],
        }
    payload["command"] = command
    payload["executed_dimensions"] = dimensions
    payload["executed_levels"] = levels
    return payload


def _run_compare(
    spec: LoopSpec,
    *,
    results_root: Path,
    dimension: int,
    level: int,
) -> dict[str, Any]:
    compare_root = results_root / f"loop_{spec.loop_id:03d}_{spec.family}_{spec.phase}" / "compare"
    command = [
        "python3",
        "-m",
        "experiments.smolyak_experiment.compare_smolyak_vs_mc",
        "--platform",
        "gpu",
        "--dimension",
        str(dimension),
        "--level",
        str(level),
        "--dtype",
        spec.dtypes.split(",")[0],
        "--family",
        spec.family,
        "--gaussian-alpha",
        "0.8",
        "--anisotropic-alpha-start",
        "0.2",
        "--anisotropic-alpha-stop",
        "1.4",
        "--shift-start",
        "-0.25",
        "--shift-stop",
        "0.25",
        "--laplace-beta-start",
        "1.0",
        "--laplace-beta-stop",
        "6.0",
        "--coeff-start",
        "-1.5",
        "--coeff-stop",
        "1.5",
        "--chunk-size",
        spec.chunk_sizes.split(",")[0],
        "--warm-repeats",
        "2",
        "--mc-seeds",
        "8",
        "--output-dir",
        str(compare_root),
    ]
    payload = _run_json_command(command, cwd=WORKSPACE_ROOT)
    payload["command"] = command
    return payload


def _critical_review_lines(
    *,
    matrix_summary: dict[str, Any],
    frontier_csv_path: Path,
    aggregate_csv_path: Path,
    compare_payload: dict[str, Any] | None,
) -> list[str]:
    lines: list[str] = []
    actual_mode_counts = matrix_summary.get("actual_mode_counts", {})
    failure_counts = matrix_summary.get("failure_counts", {})
    if failure_counts:
        lines.append(f"- Failure modes observed: {failure_counts}")
    else:
        lines.append("- No failures were recorded in this loop; the current cap may be too conservative.")

    auto_snapshot = _frontier_snapshot(frontier_csv_path, "auto")
    if auto_snapshot is not None and auto_snapshot["first_failure_dimension"] is None:
        lines.append(
            f"- `auto` reached the current loop cap at level {auto_snapshot['highest_level']}; the next loop should raise level or dimension limits."
        )

    aggregate_rows = _read_csv_rows(aggregate_csv_path)
    auto_rows = [row for row in aggregate_rows if row.get("requested_mode") == "auto"]
    if auto_rows:
        low_util_rows = [
            row
            for row in auto_rows
            if row.get("median_batch_avg_gpu_util")
            and float(row["median_batch_avg_gpu_util"]) < 10.0
        ]
        if low_util_rows:
            lines.append(
                f"- {len(low_util_rows)} auto-mode cells had median average GPU utilization below 10%; batching or compile amortization is still weak there."
            )

    if compare_payload is not None:
        summary = compare_payload.get("summary", {})
        if summary.get("smolyak_more_accurate_same_budget"):
            lines.append("- Smolyak beat Monte Carlo on same-budget error for the chosen compare case.")
        else:
            lines.append("- Monte Carlo matched or beat Smolyak on same-budget error for the chosen compare case.")
        if not summary.get("smolyak_faster_on_warm_runtime"):
            lines.append("- Warm-runtime speed is still a concern versus Monte Carlo on the chosen compare case.")

    if actual_mode_counts.get("points", 0) and not actual_mode_counts.get("indexed", 0):
        lines.append("- The auto policy never switched into indexed mode in this loop; the dense threshold may still be permissive for this regime.")
    return lines


def _measurement_improvement_lines(
    *,
    compare_payload: dict[str, Any] | None,
    frontier_csv_path: Path,
) -> list[str]:
    lines: list[str] = []
    if compare_payload is None:
        lines.append("- Add at least one Monte Carlo compare case for the hardest successful auto-mode cell.")
    else:
        summary = compare_payload.get("summary", {})
        if abs(float(summary.get("smolyak_error", 0.0))) < 1e-12:
            lines.append("- Use absolute error rather than relative error as the primary metric for near-zero analytic targets.")
    auto_snapshot = _frontier_snapshot(frontier_csv_path, "auto")
    if auto_snapshot is not None and auto_snapshot["first_failure_dimension"] is None:
        lines.append("- Increase the level or dimension cap in the next loop because the present frontier saturated the loop bounds.")
    else:
        lines.append("- Keep logging failure-onset dimensions so implementation bugs are not confused with true frontier limits.")
    return lines


def _format_optional_cell(value: Any) -> str:
    if value in (None, "", "None"):
        return "none"
    return str(value)


def _frontier_summary_lines(frontier_csv_path: Path, requested_modes_csv: str) -> list[str]:
    lines: list[str] = []
    for requested_mode in requested_modes_csv.split(","):
        snapshot = _frontier_snapshot(frontier_csv_path, requested_mode)
        if snapshot is None:
            continue
        row = snapshot["row"]
        lines.append(
            (
                f"- `{requested_mode}`: highest level `{snapshot['highest_level']}`, "
                f"max successful dimension `{_format_optional_cell(snapshot['max_success_dimension'])}`, "
                f"first failure `{_format_optional_cell(snapshot['first_failure_dimension'])}`, "
                f"last-success storage `{_format_optional_cell(row.get('last_success_storage_bytes'))}` bytes, "
                f"last-success batch runtime `{_format_optional_cell(row.get('last_success_batch_warm_runtime_ms'))}` ms, "
                f"last-success batch peak GPU memory `{_format_optional_cell(row.get('last_success_batch_peak_mem_used_mb'))}` MiB."
            )
        )
    return lines


def _selected_figure_lines(report_payload: dict[str, Any]) -> list[str]:
    figures = report_payload.get("figures", {})
    if not isinstance(figures, dict):
        return []

    preferred_titles = (
        "Frontier ",
        "Fastest Success Mode ",
        "Auto Frontier Gap ",
        "GPU Util ",
    )
    selected_lines: list[str] = []
    for title, path in figures.items():
        if not isinstance(title, str) or not isinstance(path, str):
            continue
        if title.startswith(preferred_titles[:3]) or (
            title.startswith("GPU Util ") and "mode=auto" in title
        ):
            selected_lines.append(f"- {title}: `{path}`")
    return selected_lines


def _write_loop_note(
    *,
    spec: LoopSpec,
    note_path: Path,
    matrix_payload: dict[str, Any],
    matrix_summary: dict[str, Any],
    compare_payload: dict[str, Any] | None,
) -> None:
    report_payload = matrix_payload.get("report_payload", {})
    frontier_csv_path = Path(report_payload["frontier_csv"])
    aggregate_csv_path = Path(report_payload["aggregate_csv"])
    raw_csv_path = Path(report_payload["raw_csv"])
    report_md_path = Path(report_payload["report_md"])
    partial = bool(matrix_payload.get("partial"))
    executed_dimensions = str(matrix_payload.get("executed_dimensions", spec.dimensions))
    executed_levels = str(matrix_payload.get("executed_levels", spec.levels))

    auto_snapshot = _frontier_snapshot(frontier_csv_path, "auto")
    critical_review = _critical_review_lines(
        matrix_summary=matrix_summary,
        frontier_csv_path=frontier_csv_path,
        aggregate_csv_path=aggregate_csv_path,
        compare_payload=compare_payload,
    )
    measurement_improvements = _measurement_improvement_lines(
        compare_payload=compare_payload,
        frontier_csv_path=frontier_csv_path,
    )
    frontier_summary = _frontier_summary_lines(frontier_csv_path, spec.requested_modes)
    figure_lines = _selected_figure_lines(report_payload)

    compare_section = ["No Monte Carlo compare case was run."]
    if compare_payload is not None:
        compare_section = [
            f"- Compare JSON: `{compare_payload['output_json']}`",
            f"- Compare target: `d={compare_payload['dimension']}`, `level={compare_payload['level']}`, `mode={compare_payload['smolyak']['materialization_mode']}`",
            f"- Same-budget: Smolyak more accurate = `{compare_payload['summary']['smolyak_more_accurate_same_budget']}`",
            f"- Warm runtime: Smolyak faster = `{compare_payload['summary']['smolyak_faster_on_warm_runtime']}`",
            f"- Smolyak absolute error = `{compare_payload['summary']['smolyak_error']}`",
            f"- Monte Carlo same-budget absolute error = `{compare_payload['summary']['monte_carlo_same_budget_error']}`",
            f"- Monte Carlo matched-error absolute error = `{compare_payload['summary']['monte_carlo_error']}`",
        ]

    lines = [
        f"# Smolyak Research Loop {spec.loop_id:03d}",
        "",
        f"Date: {datetime.now(timezone.utc).date().isoformat()}",
        "",
        "## Goal",
        "",
        f"{spec.title}. {spec.rationale}",
        "",
        "## Executed Calculations",
        "",
        f"- Family: `{spec.family}`",
        f"- DTypes: `{spec.dtypes}`",
        f"- Dimensions: `{spec.dimensions}`",
        f"- Levels: `{spec.levels}`",
        f"- Executed dimensions: `{executed_dimensions}`",
        f"- Executed levels: `{executed_levels}`",
        f"- Requested modes: `{spec.requested_modes}`",
        f"- Chunk sizes: `{spec.chunk_sizes}`",
        f"- Batch size: `{spec.batch_size}`",
        f"- Timeout per case: `{spec.timeout_seconds}` seconds",
        f"- Matrix command: `{' '.join(matrix_payload['command'])}`",
        "",
        "## Primary Outputs",
        "",
        f"- Matrix run dir: `{matrix_payload['run_dir']}`",
        f"- Matrix JSONL: `{matrix_payload['jsonl_path']}`",
        f"- Matrix summary JSON: `{matrix_payload['summary_json']}`",
        f"- Matrix Markdown report: `{report_md_path}`",
        f"- Raw CSV: `{raw_csv_path}`",
        f"- Frontier CSV: `{frontier_csv_path}`",
        "",
        "## Result Summary",
        "",
    ]
    if partial:
        lines.extend(
            [
                f"- Matrix status: `partial` (return code `{matrix_payload.get('matrix_returncode')}`)",
                f"- Cases recorded before interruption: `{matrix_summary['cases_recorded']}`",
            ]
        )
    else:
        lines.append(f"- Cases requested: `{matrix_summary['cases_requested']}`")
    lines.extend(
        [
        f"- Cases succeeded: `{matrix_summary['cases_succeeded']}`",
        f"- Cases failed: `{matrix_summary['cases_failed']}`",
        f"- Actual mode counts: `{matrix_summary['actual_mode_counts']}`",
        ]
    )
    if auto_snapshot is not None:
        lines.extend(
            [
                f"- Auto highest level in this loop: `{auto_snapshot['highest_level']}`",
                f"- Auto max successful dimension at that level: `{auto_snapshot['max_success_dimension']}`",
                f"- Auto first failure dimension at that level: `{auto_snapshot['first_failure_dimension']}`",
            ]
        )
    if partial and matrix_payload.get("matrix_stderr_tail"):
        lines.extend(
            [
                f"- Matrix stderr tail: `{str(matrix_payload['matrix_stderr_tail']).strip()}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Monte Carlo Compare",
            "",
            *compare_section,
            "",
            "## Frontier Snapshot",
            "",
            *(frontier_summary if frontier_summary else ["- No frontier rows were available."]),
            "",
            "## Figures",
            "",
            *(figure_lines if figure_lines else ["- No figure paths were recorded."]),
            "",
            "## Critical Review",
            "",
            *critical_review,
            "",
            "## Measurement Improvements",
            "",
            *measurement_improvements,
            "",
            "## Next Step",
            "",
            "- Use the critical-review findings to choose the next implementation or measurement change before rerunning.",
            "",
        ]
    )
    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _selected_specs(loop_ids: Iterable[int]) -> list[LoopSpec]:
    spec_map = {spec.loop_id: spec for spec in LOOP_SPECS}
    missing = [loop_id for loop_id in loop_ids if loop_id not in spec_map]
    if missing:
        raise ValueError(f"Unknown loop ids: {missing}")
    return [spec_map[loop_id] for loop_id in loop_ids]


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    loop_ids = _parse_csv_ints(args.loop_ids)
    specs = _selected_specs(loop_ids)
    results_root = Path(args.results_root).resolve()
    notes_root = Path(args.notes_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    notes_root.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        note_path = notes_root / f"smolyak_research_loop_{spec.loop_id:03d}.md"
        if note_path.exists() and not args.force:
            print(json.dumps({"loop_id": spec.loop_id, "skipped": True, "note_path": str(note_path)}))
            continue
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "loop_id": spec.loop_id,
                        "title": spec.title,
                        "family": spec.family,
                        "levels": spec.levels,
                        "requested_modes": spec.requested_modes,
                    },
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                )
            )
            continue

        matrix_payload = _run_matrix_loop(
            spec,
            results_root=results_root,
            workers_per_gpu=args.workers_per_gpu,
            override_dimensions=args.override_dimensions,
            override_levels=args.override_levels,
        )
        matrix_summary = dict(matrix_payload["matrix_summary"])
        compare_payload = None
        report_payload = matrix_payload.get("report_payload", {})
        frontier_csv_path = Path(report_payload["frontier_csv"])
        compare_target = _best_auto_compare_target(frontier_csv_path)
        if compare_target is not None:
            compare_payload = _run_compare(
                spec,
                results_root=results_root,
                dimension=compare_target[0],
                level=compare_target[1],
            )

        _write_loop_note(
            spec=spec,
            note_path=note_path,
            matrix_payload=matrix_payload,
            matrix_summary=matrix_summary,
            compare_payload=compare_payload,
        )
        print(
            json.dumps(
                {
                    "loop_id": spec.loop_id,
                    "note_path": str(note_path),
                    "matrix_run_dir": matrix_payload["run_dir"],
                    "compare_output_json": None if compare_payload is None else compare_payload["output_json"],
                },
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
