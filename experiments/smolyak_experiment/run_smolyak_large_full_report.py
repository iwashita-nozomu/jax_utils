#!/usr/bin/env python3
"""Run the existing `large` preset in one command and generate a compact report."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent

from python.experiment_runner import (
    CHILD_COMPLETE_PREFIX,
    WorkerSlot,
    append_jsonl_record,
    apply_worker_environment,
    build_worker_slots,
    json_compatible,
    run_cases_with_subprocess_scheduler,
    worker_slot_from_mapping,
)
import experiments.smolyak_experiment.cases as cases
import experiments.smolyak_experiment.results_aggregator as results_aggregator
from experiments.smolyak_experiment.report_smolyak_same_budget_accuracy import _write_line_svg
from experiments.smolyak_experiment.run_smolyak_experiment_simple import (
    SmolyakWorker,
    _generate_final_results,
    _read_jsonl_records,
    get_experiment_config,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing `large` Smolyak preset in isolated subprocesses and "
            "generate JSONL, final JSON, and a compact Markdown/SVG report in one command."
        ),
    )
    parser.add_argument(
        "--size",
        default="large",
        choices=["medium", "large"],
        help="Preset from run_smolyak_experiment_simple.py. Defaults to `large`.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap for debugging. Leave unset for the full preset.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=10,
        help="Per-case timeout. Defaults to 10 seconds so the full preset can terminate in practice.",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Number of subprocess workers to place on each visible GPU.",
    )
    parser.add_argument(
        "--gpu-indices",
        default=None,
        help="Optional comma-separated GPU indices. Defaults to all GPUs from nvidia-smi.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case start/finish logs.",
    )
    parser.add_argument(
        "--child-case-json",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--child-worker-slot-json",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--child-jsonl-output",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def _parse_gpu_indices(spec: str) -> list[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def _discover_gpu_indices() -> list[int]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    return [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]


def _case_params(case: Mapping[str, object]) -> dict[str, object]:
    return {
        "dimension": int(case["dimension"]),
        "level": int(case["level"]),
        "dtype": str(case["dtype"]),
        "trial_index": int(case["trial_index"]),
    }


def _case_label(case: Mapping[str, object]) -> str:
    params = _case_params(case)
    return f"d{params['dimension']}_l{params['level']}_{params['dtype']}_t{params['trial_index']}"


def _context_for_case(case: Mapping[str, object], jsonl_path: Path) -> dict[str, object]:
    return {
        "case_id": str(case["case_id"]),
        "jsonl_path": str(jsonl_path),
    }


def _failure_result(
    case: Mapping[str, object],
    *,
    failure_kind: str,
    error_text: str,
    details: str,
) -> dict[str, object]:
    params = _case_params(case)
    return {
        "case_id": str(case["case_id"]),
        "case_params": params,
        "failure_kind": failure_kind,
        "smolyak": {
            "status": "FAILURE",
            "init_time_ms": 0.0,
            "jax_import_time_ms": None,
            "integrate_time_ms": 0.0,
            "integrate_first_call_ms": None,
            "integrate_second_call_ms": None,
            "compile_time_ms": None,
            "num_evaluation_points": 0,
            "storage_bytes": None,
            "integral_value": 0.0,
            "analytical_value": float(params["dimension"]) / 12.0,
            "absolute_error": None,
            "relative_error": None,
            "error": f"{failure_kind}: {error_text}\n{details}".strip(),
        },
        "monte_carlo": {"error": f"{failure_kind}: {error_text}".strip()},
    }


def _run_case_in_child(
    case_json: str,
    worker_slot_json: str,
    jsonl_output_path: Path,
) -> None:
    case = json.loads(case_json)
    worker_slot = worker_slot_from_mapping(json.loads(worker_slot_json))
    platform = str(case.get("device", "gpu"))
    apply_worker_environment(
        platform=platform,
        worker_slot=worker_slot,
        disable_gpu_preallocation=(platform == "gpu"),
    )

    context = _context_for_case(case, jsonl_output_path)
    worker = SmolyakWorker()
    try:
        t_import_start = time.perf_counter()
        import jax.numpy as jnp
        from python.jax_util.functional.smolyak import SmolyakIntegrator

        t_import_end = time.perf_counter()
        result = worker._run_case(
            case,
            jnp=jnp,
            SmolyakIntegrator=SmolyakIntegrator,
            context=context,
            extra_timers={"t_jax_import_ms": (t_import_end - t_import_start) * 1000.0},
        )
    except Exception as exc:
        message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        result = _failure_result(
            case,
            failure_kind="child_error",
            error_text=message,
            details=traceback.format_exc(limit=8)[-4000:],
        )

    worker._save_result(result, context)
    print(f"{CHILD_COMPLETE_PREFIX}{json.dumps(json_compatible(result), ensure_ascii=True)}", flush=True)


def _build_child_command(
    case: Mapping[str, object],
    worker_slot: WorkerSlot,
    jsonl_output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.smolyak_experiment.run_smolyak_large_full_report",
        "--child-case-json",
        json.dumps(json_compatible(case), ensure_ascii=True),
        "--child-worker-slot-json",
        json.dumps(json_compatible(worker_slot.to_dict()), ensure_ascii=True),
        "--child-jsonl-output",
        str(jsonl_output_path),
    ]


def _parent_failure_result(
    case: Mapping[str, object],
    worker_slot: WorkerSlot,
    failure_kind: str,
    error_text: str,
    details: str,
) -> dict[str, object]:
    del worker_slot
    return _failure_result(case, failure_kind=failure_kind, error_text=error_text, details=details[-4000:])


def _log_case_started(case: Mapping[str, object], worker_slot: WorkerSlot) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"[{started_at}] start {worker_slot.worker_label} {_case_label(case)}", flush=True)


def _log_case_finished(case: Mapping[str, object], worker_slot: WorkerSlot, result: Mapping[str, object]) -> None:
    finished_at = datetime.now(timezone.utc).isoformat()
    smolyak = result.get("smolyak", {})
    status = smolyak.get("status") if isinstance(smolyak, dict) else "UNKNOWN"
    failure_kind = result.get("failure_kind", "-")
    print(
        f"[{finished_at}] done  {worker_slot.worker_label} {_case_label(case)} status={status} failure_kind={failure_kind}",
        flush=True,
    )


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _aggregate_cells(
    records: list[dict[str, Any]],
    *,
    expected_trials: int,
) -> dict[tuple[str, int, int], dict[str, Any]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        params = record.get("case_params", {})
        if not isinstance(params, dict):
            continue
        key = (
            str(params.get("dtype", "unknown")),
            int(params.get("dimension", -1)),
            int(params.get("level", -1)),
        )
        grouped[key].append(record)

    cells: dict[tuple[str, int, int], dict[str, Any]] = {}
    for key, items in grouped.items():
        success_items = [
            item
            for item in items
            if isinstance(item.get("smolyak"), dict) and item["smolyak"].get("status") == "SUCCESS"
        ]
        failure_counter: Counter[str] = Counter()
        for item in items:
            if item in success_items:
                continue
            failure_kind = str(item.get("failure_kind", "FAILURE"))
            if failure_kind == "FAILURE":
                smolyak = item.get("smolyak", {})
                if isinstance(smolyak, dict):
                    failure_kind = str(smolyak.get("error", "FAILURE")).split(":", 1)[0]
            failure_counter[failure_kind] += 1

        smolyak_times = [
            float(item["smolyak"]["integrate_second_call_ms"])
            for item in success_items
            if isinstance(item["smolyak"].get("integrate_second_call_ms"), (int, float))
        ]
        smolyak_errors = [
            float(item["smolyak"]["absolute_error"])
            for item in success_items
            if isinstance(item["smolyak"].get("absolute_error"), (int, float))
        ]
        points = [
            float(item["smolyak"]["num_evaluation_points"])
            for item in success_items
            if isinstance(item["smolyak"].get("num_evaluation_points"), (int, float))
        ]
        mc_times = [
            float(item["monte_carlo"]["time_ms"])
            for item in success_items
            if isinstance(item.get("monte_carlo"), dict) and isinstance(item["monte_carlo"].get("time_ms"), (int, float))
        ]
        mc_errors = [
            float(item["monte_carlo"]["absolute_error"])
            for item in success_items
            if isinstance(item.get("monte_carlo"), dict) and isinstance(item["monte_carlo"].get("absolute_error"), (int, float))
        ]

        cells[key] = {
            "dtype": key[0],
            "dimension": key[1],
            "level": key[2],
            "trial_count": len(items),
            "expected_trials": expected_trials,
            "success_count": len(success_items),
            "complete_trials": len(items) == expected_trials,
            "full_success": len(items) == expected_trials and len(success_items) == expected_trials,
            "failure_counts": dict(sorted(failure_counter.items())),
            "median_points": _median(points),
            "median_smolyak_time_ms": _median(smolyak_times),
            "median_smolyak_abs_error": _median(smolyak_errors),
            "median_mc_time_ms": _median(mc_times),
            "median_mc_abs_error": _median(mc_errors),
        }
        if cells[key]["median_smolyak_time_ms"] is not None and cells[key]["median_mc_time_ms"] is not None:
            cells[key]["runtime_ratio_mc_over_smolyak"] = (
                float(cells[key]["median_mc_time_ms"]) / max(float(cells[key]["median_smolyak_time_ms"]), 1e-30)
            )
        else:
            cells[key]["runtime_ratio_mc_over_smolyak"] = None
        if cells[key]["median_smolyak_abs_error"] is not None and cells[key]["median_mc_abs_error"] is not None:
            cells[key]["error_ratio_mc_over_smolyak"] = (
                float(cells[key]["median_mc_abs_error"]) / max(float(cells[key]["median_smolyak_abs_error"]), 1e-30)
            )
        else:
            cells[key]["error_ratio_mc_over_smolyak"] = None

    return cells


def _frontier_by_dimension(
    cells: dict[tuple[str, int, int], dict[str, Any]],
    *,
    dimensions: list[int],
    levels: list[int],
    dtypes: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dtype in dtypes:
        for dimension in dimensions:
            frontier: dict[str, Any] | None = None
            for level in levels:
                cell = cells.get((dtype, dimension, level))
                if cell is None or not cell["full_success"]:
                    continue
                frontier = cell
            rows.append(
                {
                    "dtype": dtype,
                    "dimension": dimension,
                    "frontier_level": None if frontier is None else int(frontier["level"]),
                    "median_points": None if frontier is None else frontier["median_points"],
                    "runtime_ratio_mc_over_smolyak": None if frontier is None else frontier["runtime_ratio_mc_over_smolyak"],
                    "error_ratio_mc_over_smolyak": None if frontier is None else frontier["error_ratio_mc_over_smolyak"],
                }
            )
    return rows


def _max_dimension_by_level(
    cells: dict[tuple[str, int, int], dict[str, Any]],
    *,
    dimensions: list[int],
    levels: list[int],
    dtypes: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dtype in dtypes:
        for level in levels:
            max_dimension: int | None = None
            for dimension in dimensions:
                cell = cells.get((dtype, dimension, level))
                if cell is not None and cell["full_success"]:
                    max_dimension = dimension
            rows.append({"dtype": dtype, "level": level, "max_full_success_dimension": max_dimension})
    return rows


def _write_markdown_report(
    *,
    config: Any,
    final_results: dict[str, Any],
    cells: dict[tuple[str, int, int], dict[str, Any]],
    frontier_rows: list[dict[str, Any]],
    max_dimension_rows: list[dict[str, Any]],
    output_path: Path,
    figure_paths: dict[str, Path],
) -> None:
    dimensions = list(range(config.min_dimension, config.max_dimension + 1))
    levels = list(range(config.min_level, config.max_level + 1))
    dtypes = list(config.dtypes)
    failure_groups = results_aggregator.filter_failures(_read_jsonl_records(Path(final_results["output_jsonl"])))
    failure_summary = {kind: len(items) for kind, items in failure_groups.items() if kind != "SUCCESS"}
    full_success_cells = [cell for cell in cells.values() if cell["full_success"]]
    best_accuracy = max(
        full_success_cells,
        key=lambda cell: float(cell["error_ratio_mc_over_smolyak"] or float("-inf")),
        default=None,
    )
    best_smolyak_speed = max(
        full_success_cells,
        key=lambda cell: float(cell["runtime_ratio_mc_over_smolyak"] or float("-inf")),
        default=None,
    )

    lines = [
        "# Smolyak Large Preset Report",
        "",
        "## Method",
        "",
        "- ケース集合は既存の `run_smolyak_experiment_simple.py --size large` と同じ preset を使っています。",
        "- 具体的には `dimension=1..20`, `level=1..20`, `dtype=float16,bfloat16,float32,float64`, `trial=3` の直積です。",
        "- 各ケースは子プロセスで 1 件ずつ実行し、timeout/OOM も結果として JSONL に残しています。",
        "- Smolyak は二次関数 `sum(x^2)` を積分し、Monte Carlo は同じ `num_evaluation_points` を使う same-budget baseline です。",
        "",
        "## Figures",
        "",
        f"![Frontier level by dimension (linear)]({figure_paths['frontier_level'].name})",
        "",
        f"![Frontier points by dimension (log)]({figure_paths['frontier_points'].name})",
        "",
        f"![Success rate by level (linear)]({figure_paths['success_rate'].name})",
        "",
        f"![Frontier runtime ratio (linear)]({figure_paths['runtime_ratio'].name})",
        "",
        f"![Frontier error ratio (log)]({figure_paths['error_ratio'].name})",
        "",
        "## Observations",
        "",
        f"- Total task count: {final_results['total_cases']}",
        f"- Successful tasks: {final_results['successful_cases']}",
        f"- Failed tasks: {final_results['failed_cases']}",
        f"- Full-success cells (all 3 trials successful): {len(full_success_cells)}/{len(dimensions) * len(levels) * len(dtypes)}",
        f"- Failure kinds: {json.dumps(failure_summary, ensure_ascii=True, sort_keys=True)}",
    ]
    if best_accuracy is not None:
        lines.append(
            "- Largest frontier accuracy advantage for Smolyak: "
            f"{best_accuracy['dtype']}-d{best_accuracy['dimension']}-l{best_accuracy['level']} "
            f"with MC/Smolyak absolute-error ratio {float(best_accuracy['error_ratio_mc_over_smolyak']):.3f}."
        )
    if best_smolyak_speed is not None:
        lines.append(
            "- Largest frontier speed advantage for Smolyak: "
            f"{best_smolyak_speed['dtype']}-d{best_smolyak_speed['dimension']}-l{best_smolyak_speed['level']} "
            f"with MC/Smolyak runtime ratio {float(best_smolyak_speed['runtime_ratio_mc_over_smolyak']):.3f}."
        )

    lines.extend(
        [
            "",
            "## Max Full-Success Dimension By Level",
            "",
            "| Level | float16 | bfloat16 | float32 | float64 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    by_level_dtype = {(row["level"], row["dtype"]): row["max_full_success_dimension"] for row in max_dimension_rows}
    for level in levels:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(level),
                    str(by_level_dtype.get((level, "float16"))),
                    str(by_level_dtype.get((level, "bfloat16"))),
                    str(by_level_dtype.get((level, "float32"))),
                    str(by_level_dtype.get((level, "float64"))),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Frontier Table",
            "",
            "| Dtype | Dimension | Frontier level | Points | MC/Smolyak runtime ratio | MC/Smolyak error ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in frontier_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dtype"]),
                    str(row["dimension"]),
                    "no result" if row["frontier_level"] is None else str(row["frontier_level"]),
                    "no result" if row["median_points"] is None else str(int(row["median_points"])),
                    "no result" if row["runtime_ratio_mc_over_smolyak"] is None else f"{float(row['runtime_ratio_mc_over_smolyak']):.3f}",
                    "no result" if row["error_ratio_mc_over_smolyak"] is None else f"{float(row['error_ratio_mc_over_smolyak']):.3f}",
                ]
            )
            + " |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report_assets(
    *,
    records: list[dict[str, Any]],
    final_results: dict[str, Any],
    config: Any,
    report_dir: Path,
) -> dict[str, Any]:
    dimensions = list(range(config.min_dimension, config.max_dimension + 1))
    levels = list(range(config.min_level, config.max_level + 1))
    dtypes = list(config.dtypes)
    cells = _aggregate_cells(records, expected_trials=int(config.num_trials))
    frontier_rows = _frontier_by_dimension(cells, dimensions=dimensions, levels=levels, dtypes=dtypes)
    max_dimension_rows = _max_dimension_by_level(cells, dimensions=dimensions, levels=levels, dtypes=dtypes)

    frontier_level_plot = report_dir / "frontier_level_by_dimension_linear.svg"
    frontier_points_plot = report_dir / "frontier_points_by_dimension_log.svg"
    success_rate_plot = report_dir / "success_rate_by_level_linear.svg"
    runtime_ratio_plot = report_dir / "frontier_runtime_ratio_linear.svg"
    error_ratio_plot = report_dir / "frontier_error_ratio_log.svg"

    frontier_by_dtype = {dtype: [row for row in frontier_rows if row["dtype"] == dtype] for dtype in dtypes}
    _write_line_svg(
        x_values=dimensions,
        series=[[row["frontier_level"] for row in frontier_by_dtype[dtype]] for dtype in dtypes],
        series_labels=dtypes,
        title="Max Full-Success Level by Dimension and Dtype",
        y_label="Frontier level",
        output_path=frontier_level_plot,
        log_scale=False,
    )
    _write_line_svg(
        x_values=dimensions,
        series=[[row["median_points"] for row in frontier_by_dtype[dtype]] for dtype in dtypes],
        series_labels=dtypes,
        title="Evaluation Points at the Full-Success Frontier",
        y_label="Points",
        output_path=frontier_points_plot,
        log_scale=True,
    )
    _write_line_svg(
        x_values=levels,
        series=[
            [
                sum(
                    1
                    for dimension in dimensions
                    if cells.get((dtype, dimension, level)) is not None and cells[(dtype, dimension, level)]["full_success"]
                )
                / max(
                    1,
                    sum(1 for dimension in dimensions if cells.get((dtype, dimension, level)) is not None and cells[(dtype, dimension, level)]["complete_trials"]),
                )
                for level in levels
            ]
            for dtype in dtypes
        ],
        series_labels=dtypes,
        title="Fraction of Full-Success Cells by Level",
        y_label="Success rate",
        output_path=success_rate_plot,
        log_scale=False,
    )
    _write_line_svg(
        x_values=dimensions,
        series=[[row["runtime_ratio_mc_over_smolyak"] for row in frontier_by_dtype[dtype]] for dtype in dtypes],
        series_labels=dtypes,
        title="Frontier Runtime Ratio (Monte Carlo / Smolyak)",
        y_label="Runtime ratio",
        output_path=runtime_ratio_plot,
        log_scale=False,
    )
    _write_line_svg(
        x_values=dimensions,
        series=[[row["error_ratio_mc_over_smolyak"] for row in frontier_by_dtype[dtype]] for dtype in dtypes],
        series_labels=dtypes,
        title="Frontier Error Ratio (Monte Carlo / Smolyak)",
        y_label="Error ratio",
        output_path=error_ratio_plot,
        log_scale=True,
    )

    summary = {
        "experiment": "smolyak_large_full_report",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        "final_results": final_results,
        "frontier_rows": frontier_rows,
        "max_dimension_rows": max_dimension_rows,
        "figures": {
            "frontier_level_by_dimension_linear": str(frontier_level_plot),
            "frontier_points_by_dimension_log": str(frontier_points_plot),
            "success_rate_by_level_linear": str(success_rate_plot),
            "frontier_runtime_ratio_linear": str(runtime_ratio_plot),
            "frontier_error_ratio_log": str(error_ratio_plot),
        },
    }
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")

    report_path = report_dir / "report.md"
    _write_markdown_report(
        config=config,
        final_results=final_results,
        cells=cells,
        frontier_rows=frontier_rows,
        max_dimension_rows=max_dimension_rows,
        output_path=report_path,
        figure_paths={
            "frontier_level": frontier_level_plot,
            "frontier_points": frontier_points_plot,
            "success_rate": success_rate_plot,
            "runtime_ratio": runtime_ratio_plot,
            "error_ratio": error_ratio_plot,
        },
    )
    return {
        "summary_json": str(summary_path),
        "report_md": str(report_path),
        "frontier_level_by_dimension_linear": str(frontier_level_plot),
        "frontier_points_by_dimension_log": str(frontier_points_plot),
        "success_rate_by_level_linear": str(success_rate_plot),
        "frontier_runtime_ratio_linear": str(runtime_ratio_plot),
        "frontier_error_ratio_log": str(error_ratio_plot),
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.child_case_json is not None:
        if args.child_worker_slot_json is None or args.child_jsonl_output is None:
            raise ValueError("Child mode requires --child-worker-slot-json and --child-jsonl-output.")
        _run_case_in_child(
            args.child_case_json,
            args.child_worker_slot_json,
            Path(args.child_jsonl_output),
        )
        return

    config = get_experiment_config(args.size)
    config.timeout_seconds = float(args.timeout_seconds)
    case_list = cases.generate_cases(config)
    if args.max_cases is not None:
        if args.max_cases < 1:
            raise ValueError("max-cases must be positive.")
        case_list = case_list[:args.max_cases]

    run_id = int(time.time())
    run_label = datetime.fromtimestamp(run_id, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = SCRIPT_DIR / "results" / args.size
    report_dir = SCRIPT_DIR / "results" / f"{args.size}_reports" / f"report_{run_label}"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    jsonl_file = output_dir / f"results_{run_id}.jsonl"
    final_json_file = output_dir / f"final_results_{run_id}.json"

    if config.device == "gpu":
        gpu_indices = _parse_gpu_indices(args.gpu_indices) if args.gpu_indices else _discover_gpu_indices()
        if not gpu_indices:
            raise RuntimeError("No GPU indices available for the GPU large preset.")
        worker_slots = build_worker_slots("gpu", gpu_indices, args.workers_per_gpu)
    else:
        worker_slots = build_worker_slots("cpu", [], 1)

    print("=" * 70)
    print(f"Smolyak Experiment Full Report - {args.size.upper()}")
    print("=" * 70)
    print(
        f"Config: dim {config.min_dimension}-{config.max_dimension}, "
        f"level {config.min_level}-{config.max_level}, "
        f"dtypes={','.join(config.dtypes)}, "
        f"trials={config.num_trials}, tasks={len(case_list)}, timeout={config.timeout_seconds:.1f}s"
    )
    print(f"Worker slots: {len(worker_slots)}", flush=True)

    started_at = time.time()
    run_cases_with_subprocess_scheduler(
        case_list,
        worker_slots,
        timeout_seconds=int(config.timeout_seconds),
        build_child_command=lambda case, worker_slot: _build_child_command(case, worker_slot, jsonl_file),
        build_parent_failure_result=_parent_failure_result,
        fallback_jsonl_output_path=jsonl_file,
        cwd=WORKSPACE_ROOT,
        on_case_started=None if args.quiet else _log_case_started,
        on_case_finished=None if args.quiet else _log_case_finished,
    )
    elapsed_seconds = time.time() - started_at

    final_results = _generate_final_results(jsonl_file, config, elapsed_seconds)
    final_results["output_jsonl"] = str(jsonl_file)
    final_results["output_json"] = str(final_json_file)
    with final_json_file.open("w", encoding="utf-8") as handle:
        json.dump(final_results, handle, ensure_ascii=True, indent=2, sort_keys=True)

    records = _read_jsonl_records(jsonl_file)
    report_assets = _write_report_assets(
        records=records,
        final_results=final_results,
        config=config,
        report_dir=report_dir,
    )

    print(
        json.dumps(
            {
                "output_jsonl": str(jsonl_file),
                "output_json": str(final_json_file),
                "report_dir": str(report_dir),
                **report_assets,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
