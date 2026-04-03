#!/usr/bin/env python3
"""Run an exhaustive Smolyak mode matrix in isolated subprocesses."""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

from experiments.smolyak_experiment.cases import SUPPORTED_FAMILIES, make_family_bundle
from experiments.smolyak_experiment.weight_schemes import (
    SUPPORTED_WEIGHT_SCHEMES,
    format_dimension_weights,
    resolve_dimension_weights,
)


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


def _default_dimensions() -> str:
    return ",".join(str(value) for value in range(1, 51))


def _default_levels() -> str:
    return ",".join(str(value) for value in range(1, 11))


def _parse_csv_ints(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def _parse_csv_strings(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one string value is required.")
    return values


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Launch every requested Smolyak mode/dimension/level case in a child process and "
            "record both successes and failures."
        ),
    )
    parser.add_argument("--platform", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--dimensions", default=_default_dimensions())
    parser.add_argument("--levels", default=_default_levels())
    parser.add_argument("--dtypes", default="float32,float64")
    parser.add_argument(
        "--families",
        default="gaussian",
        help=f"Comma-separated integrand families: {','.join(SUPPORTED_FAMILIES)}.",
    )
    parser.add_argument(
        "--requested-modes",
        default="auto,points,indexed,lazy-indexed,batched",
        help="Comma-separated requested modes. `auto` preserves current thresholds.",
    )
    parser.add_argument(
        "--max-vectorized-suffix-ndim",
        type=int,
        default=3,
        help="Maximum number of trailing computational axes to vmap together inside batched mode.",
    )
    parser.add_argument(
        "--batched-axis-order-strategy",
        choices=["original", "length"],
        default="original",
        help="Computational axis order used by batched mode before selecting the trailing suffix block.",
    )
    parser.add_argument(
        "--chunk-sizes",
        default="16384",
        help="Comma-separated Smolyak chunk sizes.",
    )
    parser.add_argument("--gaussian-alpha", type=float, default=0.8)
    parser.add_argument("--anisotropic-alpha-start", type=float, default=0.2)
    parser.add_argument("--anisotropic-alpha-stop", type=float, default=1.4)
    parser.add_argument("--shift-start", type=float, default=-0.25)
    parser.add_argument("--shift-stop", type=float, default=0.25)
    parser.add_argument("--laplace-beta-start", type=float, default=1.0)
    parser.add_argument("--laplace-beta-stop", type=float, default=6.0)
    parser.add_argument("--coeff-start", type=float, default=-1.5)
    parser.add_argument("--coeff-stop", type=float, default=1.5)
    parser.add_argument("--dimension-weights", default=None)
    parser.add_argument(
        "--weight-scheme",
        choices=list(SUPPORTED_WEIGHT_SCHEMES),
        default="none",
    )
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--warm-repeats", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--monitor-interval-ms", type=int, default=100)
    parser.add_argument("--monitor-min-duration-ms", type=float, default=600.0)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--gpu-indices", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "results" / "smolyak_mode_matrix"),
        help="Parent directory for a timestamped result directory.",
    )
    parser.add_argument(
        "--resume-jsonl",
        default=None,
        help="Optional existing JSONL to skip already completed case_ids.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--child-case-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--child-worker-slot-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--child-jsonl-output", default=None, help=argparse.SUPPRESS)
    return parser


def _case_id(case: Mapping[str, object]) -> str:
    return str(case["case_id"])


def _case_params(case: Mapping[str, object]) -> dict[str, object]:
    return {
        "family": str(case["family"]),
        "dimension": int(case["dimension"]),
        "level": int(case["level"]),
        "dtype": str(case["dtype"]),
        "requested_mode": str(case["requested_mode"]),
        "chunk_size": int(case["chunk_size"]),
        "max_vectorized_suffix_ndim": int(case["max_vectorized_suffix_ndim"]),
        "batched_axis_order_strategy": str(case["batched_axis_order_strategy"]),
        "dimension_weights": case["dimension_weights"],
        "weight_scheme": str(case["weight_scheme"]),
        "weight_scale": float(case["weight_scale"]),
    }


def _build_cases(args: argparse.Namespace) -> list[dict[str, object]]:
    dimensions = _parse_csv_ints(args.dimensions)
    levels = _parse_csv_ints(args.levels)
    dtypes = _parse_csv_strings(args.dtypes)
    families = _parse_csv_strings(args.families)
    requested_modes = _parse_csv_strings(args.requested_modes)
    chunk_sizes = _parse_csv_ints(args.chunk_sizes)
    cases: list[dict[str, object]] = []
    case_index = 0
    for family in families:
        for dtype in dtypes:
            for chunk_size in chunk_sizes:
                for requested_mode in requested_modes:
                    for level in levels:
                        for dimension in dimensions:
                            case_id = (
                                f"{family}_d{dimension:02d}_l{level:02d}_{dtype}"
                                f"_m{requested_mode}_c{chunk_size}"
                            )
                            cases.append(
                                {
                                    "case_id": case_id,
                                    "index": case_index,
                                    "platform": args.platform,
                                    "family": family,
                                    "dimension": dimension,
                                    "level": level,
                                    "dtype": dtype,
                                    "requested_mode": requested_mode,
                                    "chunk_size": chunk_size,
                                    "max_vectorized_suffix_ndim": args.max_vectorized_suffix_ndim,
                                    "batched_axis_order_strategy": args.batched_axis_order_strategy,
                                    "gaussian_alpha": args.gaussian_alpha,
                                    "anisotropic_alpha_start": args.anisotropic_alpha_start,
                                    "anisotropic_alpha_stop": args.anisotropic_alpha_stop,
                                    "shift_start": args.shift_start,
                                    "shift_stop": args.shift_stop,
                                    "laplace_beta_start": args.laplace_beta_start,
                                    "laplace_beta_stop": args.laplace_beta_stop,
                                    "coeff_start": args.coeff_start,
                                    "coeff_stop": args.coeff_stop,
                                    "dimension_weights": args.dimension_weights,
                                    "weight_scheme": args.weight_scheme,
                                    "weight_scale": args.weight_scale,
                                    "warm_repeats": args.warm_repeats,
                                    "batch_size": args.batch_size,
                                    "monitor_interval_ms": args.monitor_interval_ms,
                                    "monitor_min_duration_ms": args.monitor_min_duration_ms,
                                }
                            )
                            case_index += 1
    return cases


def _load_completed_case_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        case_id = record.get("case_id")
        if isinstance(case_id, str):
            completed.add(case_id)
    return completed


def _failure_result(
    case: Mapping[str, object],
    *,
    failure_kind: str,
    error_text: str,
    details: str,
) -> dict[str, object]:
    params = _case_params(case)
    return {
        "case_id": _case_id(case),
        "case_params": params,
        "status": "failure",
        "failure_kind": failure_kind,
        "error_text": error_text,
        "details": details[-8000:],
        "smolyak": {
            "requested_mode": str(case["requested_mode"]),
            "actual_mode": None,
            "active_axis_count": None,
            "inactive_axis_count": None,
            "axis_level_ceilings": None,
            "num_terms": None,
            "num_evaluation_points": None,
            "storage_bytes": None,
            "vectorized_ndim": None,
            "max_vectorized_points": None,
            "value": None,
            "analytic_value": None,
            "absolute_error": None,
            "relative_error": None,
            "init_ms": None,
            "maxrss_mb": None,
            "single": None,
            "batch": None,
        },
    }


def _classify_failure_text(text: str) -> str:
    lower = text.lower()
    if "timeout" in lower:
        return "timeout"
    if "out of memory" in lower or "resource_exhausted" in lower or "cuda_error_out_of_memory" in lower:
        return "oom"
    if "nan" in lower or "inf" in lower or "diverg" in lower:
        return "numerical"
    return "error"


def _run_case_in_child(
    case_json: str,
    worker_slot_json: str,
    jsonl_output_path: Path,
) -> None:
    case = json.loads(case_json)
    worker_slot = worker_slot_from_mapping(json.loads(worker_slot_json))
    platform = str(case.get("platform", "gpu"))
    apply_worker_environment(
        platform=platform,
        worker_slot=worker_slot,
        disable_gpu_preallocation=(platform == "gpu"),
    )

    try:
        result = _execute_case(case, worker_slot)
    except Exception as exc:
        message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        result = _failure_result(
            case,
            failure_kind=_classify_failure_text(message),
            error_text=message,
            details=traceback.format_exc(limit=12),
        )

    append_jsonl_record(jsonl_output_path, result)
    print(f"{CHILD_COMPLETE_PREFIX}{json.dumps(json_compatible(result), ensure_ascii=True)}", flush=True)


def _build_child_command(
    case: Mapping[str, object],
    worker_slot: WorkerSlot,
    jsonl_output_path: Path,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.smolyak_experiment.run_smolyak_mode_matrix",
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
    normalized_kind = _classify_failure_text(f"{failure_kind}\n{error_text}\n{details}")
    return _failure_result(
        case,
        failure_kind=normalized_kind,
        error_text=f"{failure_kind}: {error_text}",
        details=details,
    )


def _log_case_started(case: Mapping[str, object], worker_slot: WorkerSlot) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"[{started_at}] start {worker_slot.worker_label} {_case_id(case)}", flush=True)


def _log_case_finished(case: Mapping[str, object], worker_slot: WorkerSlot, result: Mapping[str, object]) -> None:
    finished_at = datetime.now(timezone.utc).isoformat()
    print(
        f"[{finished_at}] done  {worker_slot.worker_label} {_case_id(case)} "
        f"status={result.get('status')} failure_kind={result.get('failure_kind', '-')}",
        flush=True,
    )


def _write_summary_json(
    output_path: Path,
    *,
    cases_requested: int,
    results: list[dict[str, object]],
    elapsed_seconds: float,
    report_payload: dict[str, object] | None,
) -> None:
    success_results = [item for item in results if item.get("status") == "success"]
    failure_results = [item for item in results if item.get("status") != "success"]
    failure_counts = Counter(str(item.get("failure_kind", "unknown")) for item in failure_results)
    actual_mode_counts = Counter(
        str(item.get("smolyak", {}).get("actual_mode"))
        for item in success_results
        if isinstance(item.get("smolyak"), dict)
    )
    warm_times = [
        float(item["smolyak"]["single"]["warm_runtime_ms"])
        for item in success_results
        if isinstance(item.get("smolyak"), dict)
        and isinstance(item["smolyak"].get("single"), dict)
        and item["smolyak"]["single"].get("warm_runtime_ms") is not None
    ]
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases_requested": cases_requested,
        "cases_completed": len(results),
        "cases_succeeded": len(success_results),
        "cases_failed": len(failure_results),
        "elapsed_seconds": elapsed_seconds,
        "failure_counts": dict(sorted(failure_counts.items())),
        "actual_mode_counts": dict(sorted(actual_mode_counts.items())),
        "single_warm_runtime_ms": {
            "median": statistics.median(warm_times) if warm_times else None,
            "max": max(warm_times) if warm_times else None,
        },
        "report_payload": report_payload,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")


def _ru_maxrss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def _monitored_repeats(warm_runtime_ms: float, warm_repeats: int, min_duration_ms: float) -> int:
    if warm_runtime_ms <= 0.0 or min_duration_ms <= 0.0:
        return warm_repeats
    return max(warm_repeats, int(math.ceil(min_duration_ms / warm_runtime_ms)))


def _benchmark_single(
    *,
    eqx_module: Any,
    integrator: Any,
    integrand: Any,
    analytic_value: float,
    warm_repeats: int,
) -> dict[str, object]:
    compiled = eqx_module.filter_jit(integrator.integrate)

    def call_once() -> float:
        value = compiled(integrand)
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()
        return float(np.asarray(value).reshape(-1)[0])

    first_start = time.perf_counter()
    first_value = call_once()
    first_stop = time.perf_counter()
    warm_values: list[float] = []
    warm_times: list[float] = []
    for _ in range(warm_repeats):
        start = time.perf_counter()
        warm_value = call_once()
        stop = time.perf_counter()
        warm_values.append(warm_value)
        warm_times.append((stop - start) * 1000.0)
    warm_runtime_ms = float(sum(warm_times) / len(warm_times)) if warm_times else 0.0
    compile_ms = max(0.0, (first_stop - first_start) * 1000.0 - warm_runtime_ms)
    value = warm_values[-1] if warm_values else first_value
    absolute_error = abs(value - analytic_value)
    return {
        "value": value,
        "absolute_error": absolute_error,
        "relative_error": absolute_error / abs(analytic_value) if analytic_value != 0.0 else absolute_error,
        "first_call_ms": (first_stop - first_start) * 1000.0,
        "warm_runtime_ms": warm_runtime_ms,
        "compile_ms": compile_ms,
        "throughput_integrals_per_second": 1000.0 / warm_runtime_ms if warm_runtime_ms > 0.0 else 0.0,
    }


def _execute_case(case: Mapping[str, object], worker_slot: WorkerSlot) -> dict[str, object]:
    del worker_slot
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from experiments.smolyak_experiment.report_smolyak_gpu_sweep import _GpuMonitor, _benchmark_compiled
    from python.jax_util.functional.smolyak import SmolyakIntegrator

    requested_mode = str(case["requested_mode"])
    dimension_weights = resolve_dimension_weights(
        dimension=int(case["dimension"]),
        dimension_weights_csv=cast(str | None, case["dimension_weights"]),
        weight_scheme=str(case["weight_scheme"]),
        weight_scale=float(case["weight_scale"]),
    )

    dtype = getattr(jnp, str(case["dtype"]))
    family_bundle = make_family_bundle(
        jnp=jnp,
        family=str(case["family"]),
        dimension=int(case["dimension"]),
        dtype=dtype,
        gaussian_alpha=float(case["gaussian_alpha"]),
        anisotropic_alpha_start=float(case["anisotropic_alpha_start"]),
        anisotropic_alpha_stop=float(case["anisotropic_alpha_stop"]),
        shift_start=float(case["shift_start"]),
        shift_stop=float(case["shift_stop"]),
        laplace_beta_start=float(case["laplace_beta_start"]),
        laplace_beta_stop=float(case["laplace_beta_stop"]),
        coeff_start=float(case["coeff_start"]),
        coeff_stop=float(case["coeff_stop"]),
    )
    family_integrand = family_bundle.integrand
    analytic_value = family_bundle.analytic_value
    family_metadata = family_bundle.metadata
    eval_one = family_bundle.eval_one
    single_scale = family_bundle.single_scale
    batch_size = int(case["batch_size"])
    scales = jnp.linspace(0.75, 1.25, batch_size, dtype=dtype)

    init_start = time.perf_counter()
    integrator = SmolyakIntegrator(
        dimension=int(case["dimension"]),
        level=int(case["level"]),
        dimension_weights=dimension_weights,
        requested_materialization_mode=requested_mode,
        max_vectorized_suffix_ndim=int(case["max_vectorized_suffix_ndim"]),
        batched_axis_order_strategy=str(case["batched_axis_order_strategy"]),
        dtype=dtype,
        chunk_size=int(case["chunk_size"]),
    )
    init_stop = time.perf_counter()

    single = _benchmark_single(
        eqx_module=eqx,
        integrator=integrator,
        integrand=family_integrand,
        analytic_value=analytic_value,
        warm_repeats=int(case["warm_repeats"]),
    )

    def eval_for_batch(scale: Any) -> Any:
        return eval_one(scale, integrator)

    compiled_single = eqx.filter_jit(eval_for_batch)
    compiled_batch = eqx.filter_jit(jax.vmap(eval_for_batch))

    monitor_factory = (
        (lambda: _GpuMonitor(int(os.environ.get("SMOLYAK_GPU_INDEX", "0")), int(case["monitor_interval_ms"])))
        if str(case["platform"]) == "gpu"
        else None
    )

    single_param = _benchmark_compiled(
        compiled=compiled_single,
        arg=single_scale,
        warm_repeats=int(case["warm_repeats"]),
        monitor_factory=monitor_factory,
        monitor_min_duration_ms=float(case["monitor_min_duration_ms"]),
    )
    batch = _benchmark_compiled(
        compiled=compiled_batch,
        arg=scales,
        warm_repeats=int(case["warm_repeats"]),
        monitor_factory=monitor_factory,
        monitor_min_duration_ms=float(case["monitor_min_duration_ms"]),
    )
    single_param_throughput = (
        1000.0 / float(single_param["warm_runtime_ms"])
        if float(single_param["warm_runtime_ms"]) > 0.0
        else 0.0
    )
    batch_throughput = (
        batch_size * 1000.0 / float(batch["warm_runtime_ms"])
        if float(batch["warm_runtime_ms"]) > 0.0
        else 0.0
    )
    batch["throughput_integrals_per_second"] = batch_throughput
    batch["throughput_speedup_vs_single"] = (
        batch_throughput / single_param_throughput if single_param_throughput > 0.0 else 0.0
    )
    single_param["throughput_integrals_per_second"] = single_param_throughput

    return {
        "case_id": _case_id(case),
        "case_params": {
            **_case_params(case),
            "platform": str(case["platform"]),
        },
        "status": "success",
        "failure_kind": None,
        "family": family_metadata,
        "smolyak": {
            "requested_mode": requested_mode,
            "actual_mode": str(integrator.materialization_mode),
            "dimension_weights": None if dimension_weights is None else list(dimension_weights),
            "dimension_weights_label": format_dimension_weights(dimension_weights),
            "active_axis_count": int(integrator.active_axis_count),
            "inactive_axis_count": int(integrator.dimension - integrator.active_axis_count),
            "axis_level_ceilings": [int(value) for value in np.asarray(integrator.axis_level_ceilings)],
            "num_terms": int(integrator.num_terms),
            "num_evaluation_points": int(integrator.num_evaluation_points),
            "storage_bytes": int(integrator.storage_bytes),
            "vectorized_ndim": int(integrator.vectorized_ndim),
            "max_vectorized_points": int(integrator.max_vectorized_points),
            "max_vectorized_suffix_ndim": int(integrator.max_vectorized_suffix_ndim),
            "batched_axis_order_strategy": str(integrator.batched_axis_order_strategy),
            "value": float(single["value"]),
            "analytic_value": float(analytic_value),
            "absolute_error": float(single["absolute_error"]),
            "relative_error": float(single["relative_error"]),
            "init_ms": (init_stop - init_start) * 1000.0,
            "maxrss_mb": _ru_maxrss_mb(),
            "single": single_param,
            "batch": batch,
        },
    }


def _run_report(jsonl_path: Path, output_dir: Path) -> dict[str, object] | None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.smolyak_experiment.report_smolyak_mode_matrix",
            "--jsonl-path",
            str(jsonl_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "status": "report_failed",
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
    return json.loads(completed.stdout)


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

    output_root = Path(args.output_dir).resolve()
    run_label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / f"report_{run_label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = run_dir / "results.jsonl"

    resume_path = Path(args.resume_jsonl).resolve() if args.resume_jsonl else None
    completed_case_ids = _load_completed_case_ids(resume_path)
    cases = [case for case in _build_cases(args) if _case_id(case) not in completed_case_ids]
    if resume_path is not None and resume_path.exists():
        jsonl_path.write_text(resume_path.read_text(encoding="utf-8"), encoding="utf-8")

    if args.platform == "gpu":
        gpu_indices = _parse_csv_ints(args.gpu_indices) if args.gpu_indices else _discover_gpu_indices()
    else:
        gpu_indices = []
    if args.platform == "gpu" and not gpu_indices:
        raise RuntimeError("No visible GPUs were discovered. Pass --gpu-indices or run with --platform cpu.")
    worker_slots = build_worker_slots(args.platform, gpu_indices, args.workers_per_gpu)

    started_at = time.time()
    results = run_cases_with_subprocess_scheduler(
        cases,
        worker_slots,
        timeout_seconds=float(args.timeout_seconds),
        build_child_command=lambda case, worker_slot: _build_child_command(case, worker_slot, jsonl_path),
        build_parent_failure_result=_parent_failure_result,
        fallback_jsonl_output_path=jsonl_path,
        cwd=WORKSPACE_ROOT,
        on_case_started=None if args.quiet else _log_case_started,
        on_case_finished=None if args.quiet else _log_case_finished,
    )
    elapsed_seconds = time.time() - started_at

    report_payload = None if args.skip_report else _run_report(jsonl_path, run_dir)
    summary_path = run_dir / "summary.json"
    _write_summary_json(
        summary_path,
        cases_requested=len(cases) + len(completed_case_ids),
        results=results,
        elapsed_seconds=elapsed_seconds,
        report_payload=report_payload,
    )

    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "jsonl_path": str(jsonl_path),
                "summary_json": str(summary_path),
                "report_payload": report_payload,
                "cases_launched_this_run": len(cases),
                "cases_resumed": len(completed_case_ids),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
