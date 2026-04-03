#!/usr/bin/env python3
"""Sweep vmap batch sizes for Smolyak GPU experiments."""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .report_smolyak_gpu_sweep import (
    SCRIPT_DIR,
    _benchmark_compiled,
    _family_eval_factory,
    _GpuMonitor,
    _parse_csv_ints,
    _write_line_svg,
)
from .cases import SUPPORTED_FAMILIES
from .weight_schemes import SUPPORTED_WEIGHT_SCHEMES, format_dimension_weights, resolve_dimension_weights


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GPU batch-scaling sweeps for vmap(integrate(f)).",
    )
    parser.add_argument(
        "--platform",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Execution platform.",
    )
    parser.add_argument(
        "--dimensions",
        default="4,8,12,16,20",
        help="Comma-separated dimensions to test.",
    )
    parser.add_argument(
        "--batch-sizes",
        default="2,4,8,16,32",
        help="Comma-separated vmap batch sizes to test.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=3,
        help="Smolyak level to test.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Floating-point dtype.",
    )
    parser.add_argument(
        "--family",
        choices=list(SUPPORTED_FAMILIES),
        default="gaussian",
        help="Integrand family.",
    )
    parser.add_argument(
        "--gaussian-alpha",
        type=float,
        default=0.8,
        help="Base Gaussian alpha for exp(-alpha ||x||^2).",
    )
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
    parser.add_argument(
        "--requested-mode",
        choices=["auto", "points", "indexed", "lazy-indexed", "batched"],
        default="auto",
        help="Requested Smolyak materialization mode.",
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
        "--warm-repeats",
        type=int,
        default=5,
        help="Number of warm runtime repeats.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="Physical GPU index to monitor and target when platform=gpu.",
    )
    parser.add_argument(
        "--monitor-interval-ms",
        type=int,
        default=100,
        help="Sampling interval for nvidia-smi monitoring.",
    )
    parser.add_argument(
        "--monitor-min-duration-ms",
        type=float,
        default=400.0,
        help="Minimum total monitored runtime per single/batch benchmark.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "results" / "smolyak_gpu_batch_scaling"),
        help="Directory for aggregate report assets.",
    )
    return parser


def _write_markdown_report(
    *,
    dimensions: list[int],
    batch_sizes: list[int],
    results_by_dim: dict[int, list[dict[str, Any]]],
    output_path: Path,
    figure_paths: dict[str, Path],
) -> None:
    all_results = [item for values in results_by_dim.values() for item in values]
    best_case = max(
        all_results,
        key=lambda item: float(item["batch"]["throughput_integrals_per_second"]),
    )
    best_speedup = max(
        all_results,
        key=lambda item: float(item["batch"]["throughput_speedup_vs_single"]),
    )
    lines = [
        "# Smolyak GPU Batch Scaling Report",
        "",
        "## Method",
        "",
        "- Each selected dimension reuses the same Smolyak integrator and sweeps `vmap(integrate(f))` batch size.",
        "- `single` is measured once per dimension and reused as the baseline for throughput speedup.",
        "- GPU monitoring samples `nvidia-smi` over additional compiled warm calls until a minimum monitor duration is reached.",
        "",
        "## Summary",
        "",
        f"- Dimensions: {dimensions}",
        f"- Batch sizes: {batch_sizes}",
        (
            "- Best batch throughput: "
            f"d={best_case['dimension']}, batch={best_case['batch_size']} "
            f"with {best_case['batch']['throughput_integrals_per_second']:.2f} integrals/s"
        ),
        (
            "- Best vmap speedup: "
            f"d={best_speedup['dimension']}, batch={best_speedup['batch_size']} "
            f"with {best_speedup['batch']['throughput_speedup_vs_single']:.2f}x"
        ),
        "",
        "## Figures",
        "",
        f"![Throughput]({figure_paths['throughput'].name})",
        "",
        f"![vmap speedup]({figure_paths['speedup'].name})",
        "",
        f"![GPU utilization]({figure_paths['gpu_util'].name})",
        "",
        f"![Pstate]({figure_paths['pstate'].name})",
        "",
        "## Per-Dimension Best Cases",
        "",
        "| Dimension | Best batch size | Best ints/s | Best speedup | Dominant Pstate | Avg GPU util | Peak GPU util |",
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]

    for dimension in dimensions:
        best = max(
            results_by_dim[dimension],
            key=lambda item: float(item["batch"]["throughput_integrals_per_second"]),
        )
        monitor = best["batch"]["monitor"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(dimension),
                    str(best["batch_size"]),
                    f"{best['batch']['throughput_integrals_per_second']:.2f}",
                    f"{best['batch']['throughput_speedup_vs_single']:.2f}",
                    str(monitor.get("dominant_pstate")),
                    (
                        f"{float(monitor['avg_gpu_util']):.2f}"
                        if monitor.get("avg_gpu_util") is not None
                        else "n/a"
                    ),
                    (
                        str(int(monitor["peak_gpu_util"]))
                        if monitor.get("peak_gpu_util") is not None
                        else "n/a"
                    ),
                ]
            )
            + " |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_results_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "dimension",
        "batch_size",
        "level",
        "family",
        "dtype",
        "requested_platform",
        "platform",
        "gpu_index",
        "requested_mode",
        "dimension_weights_label",
        "num_terms",
        "num_evaluation_points",
        "materialization_mode",
        "storage_bytes",
        "vectorized_ndim",
        "max_vectorized_points",
        "max_vectorized_suffix_ndim",
        "batched_axis_order_strategy",
        "single_value",
        "single_first_call_ms",
        "single_warm_runtime_ms",
        "single_compile_ms",
        "single_throughput_integrals_per_second",
        "batch_value",
        "batch_first_call_ms",
        "batch_warm_runtime_ms",
        "batch_compile_ms",
        "batch_throughput_integrals_per_second",
        "batch_throughput_speedup_vs_single",
        "monitor_sample_count",
        "monitor_dominant_pstate",
        "monitor_min_pstate_numeric",
        "monitor_avg_gpu_util",
        "monitor_peak_gpu_util",
        "monitor_avg_mem_util",
        "monitor_peak_mem_util",
        "monitor_peak_mem_used_mb",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            monitor = item["batch"]["monitor"] or {}
            writer.writerow(
                {
                    "dimension": item["dimension"],
                    "batch_size": item["batch_size"],
                    "level": item["level"],
                    "family": item["family"],
                    "dtype": item["dtype"],
                    "requested_platform": item["requested_platform"],
                    "platform": item["platform"],
                    "gpu_index": item["gpu_index"],
                    "requested_mode": item["requested_mode"],
                    "dimension_weights_label": item["dimension_weights_label"],
                    "num_terms": item["num_terms"],
                    "num_evaluation_points": item["num_evaluation_points"],
                    "materialization_mode": item["materialization_mode"],
                    "storage_bytes": item["storage_bytes"],
                    "vectorized_ndim": item["vectorized_ndim"],
                    "max_vectorized_points": item["max_vectorized_points"],
                    "max_vectorized_suffix_ndim": item["max_vectorized_suffix_ndim"],
                    "batched_axis_order_strategy": item["batched_axis_order_strategy"],
                    "single_value": item["single"]["value"],
                    "single_first_call_ms": item["single"]["first_call_ms"],
                    "single_warm_runtime_ms": item["single"]["warm_runtime_ms"],
                    "single_compile_ms": item["single"]["compile_ms"],
                    "single_throughput_integrals_per_second": item["single"]["throughput_integrals_per_second"],
                    "batch_value": item["batch"]["value"],
                    "batch_first_call_ms": item["batch"]["first_call_ms"],
                    "batch_warm_runtime_ms": item["batch"]["warm_runtime_ms"],
                    "batch_compile_ms": item["batch"]["compile_ms"],
                    "batch_throughput_integrals_per_second": item["batch"]["throughput_integrals_per_second"],
                    "batch_throughput_speedup_vs_single": item["batch"]["throughput_speedup_vs_single"],
                    "monitor_sample_count": monitor.get("sample_count"),
                    "monitor_dominant_pstate": monitor.get("dominant_pstate"),
                    "monitor_min_pstate_numeric": monitor.get("min_pstate_numeric"),
                    "monitor_avg_gpu_util": monitor.get("avg_gpu_util"),
                    "monitor_peak_gpu_util": monitor.get("peak_gpu_util"),
                    "monitor_avg_mem_util": monitor.get("avg_mem_util"),
                    "monitor_peak_mem_util": monitor.get("peak_mem_util"),
                    "monitor_peak_mem_used_mb": monitor.get("peak_mem_used_mb"),
                }
            )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    dimensions = _parse_csv_ints(args.dimensions)
    batch_sizes = _parse_csv_ints(args.batch_sizes)

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from python.jax_util.functional.smolyak import SmolyakIntegrator

    observed_platform = jax.devices()[0].platform if jax.devices() else "unknown"
    output_root = Path(args.output_dir)
    started_at = time.time()
    run_label = datetime.fromtimestamp(started_at, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = output_root / f"report_{run_label}"
    report_dir.mkdir(parents=True, exist_ok=True)

    dtype = getattr(jnp, args.dtype)
    results_by_dim: dict[int, list[dict[str, Any]]] = {}

    for dimension in dimensions:
        dimension_weights = resolve_dimension_weights(
            dimension=dimension,
            dimension_weights_csv=args.dimension_weights,
            weight_scheme=args.weight_scheme,
            weight_scale=args.weight_scale,
        )
        integrator = SmolyakIntegrator(
            dimension=dimension,
            level=args.level,
            dimension_weights=dimension_weights,
            requested_materialization_mode=args.requested_mode,
            max_vectorized_suffix_ndim=args.max_vectorized_suffix_ndim,
            batched_axis_order_strategy=args.batched_axis_order_strategy,
            dtype=dtype,
        )
        eval_one, alpha_single, _ = _family_eval_factory(
            jnp=jnp,
            integrator=integrator,
            dimension=dimension,
            dtype=dtype,
            family=args.family,
            gaussian_alpha=args.gaussian_alpha,
            anisotropic_alpha_start=args.anisotropic_alpha_start,
            anisotropic_alpha_stop=args.anisotropic_alpha_stop,
            shift_start=args.shift_start,
            shift_stop=args.shift_stop,
            laplace_beta_start=args.laplace_beta_start,
            laplace_beta_stop=args.laplace_beta_stop,
            coeff_start=args.coeff_start,
            coeff_stop=args.coeff_stop,
        )

        compiled_single = eqx.filter_jit(eval_one)
        monitor_factory = (
            (lambda: _GpuMonitor(args.gpu_index, args.monitor_interval_ms))
            if observed_platform == "gpu"
            else None
        )
        single = _benchmark_compiled(
            compiled=compiled_single,
            arg=alpha_single,
            warm_repeats=args.warm_repeats,
            monitor_factory=monitor_factory,
            monitor_min_duration_ms=args.monitor_min_duration_ms,
        )
        single["throughput_integrals_per_second"] = (
            1000.0 / single["warm_runtime_ms"] if single["warm_runtime_ms"] > 0.0 else 0.0
        )

        dim_results: list[dict[str, Any]] = []
        for batch_size in batch_sizes:
            batch_alphas = (
                jnp.linspace(
                    0.75,
                    1.25,
                    batch_size,
                    dtype=dtype,
                )
            )
            compiled_batch = eqx.filter_jit(jax.vmap(eval_one))
            batch = _benchmark_compiled(
                compiled=compiled_batch,
                arg=batch_alphas,
                warm_repeats=args.warm_repeats,
                monitor_factory=monitor_factory,
                monitor_min_duration_ms=args.monitor_min_duration_ms,
            )
            batch["throughput_integrals_per_second"] = (
                batch_size * 1000.0 / batch["warm_runtime_ms"]
                if batch["warm_runtime_ms"] > 0.0
                else 0.0
            )
            batch["throughput_speedup_vs_single"] = (
                batch["throughput_integrals_per_second"] / single["throughput_integrals_per_second"]
                if single["throughput_integrals_per_second"] > 0.0
                else 0.0
            )
            dim_results.append(
                {
                    "dimension": dimension,
                    "batch_size": batch_size,
                    "level": args.level,
                    "family": args.family,
                    "dtype": args.dtype,
                    "requested_platform": args.platform,
                    "platform": observed_platform,
                    "gpu_index": args.gpu_index,
                    "requested_mode": args.requested_mode,
                    "dimension_weights": None if dimension_weights is None else list(dimension_weights),
                    "dimension_weights_label": format_dimension_weights(dimension_weights),
                    "num_terms": int(integrator.num_terms),
                    "num_evaluation_points": int(integrator.num_evaluation_points),
                    "materialization_mode": str(integrator.materialization_mode),
                    "storage_bytes": int(integrator.storage_bytes),
                    "vectorized_ndim": int(integrator.vectorized_ndim),
                    "max_vectorized_points": int(integrator.max_vectorized_points),
                    "max_vectorized_suffix_ndim": int(integrator.max_vectorized_suffix_ndim),
                    "batched_axis_order_strategy": str(integrator.batched_axis_order_strategy),
                    "single": single,
                    "batch": batch,
                }
            )
        results_by_dim[dimension] = dim_results

    throughput_path = report_dir / "throughput.svg"
    speedup_path = report_dir / "vmap_speedup.svg"
    gpu_util_path = report_dir / "gpu_util.svg"
    pstate_path = report_dir / "pstate.svg"

    _write_line_svg(
        x_values=batch_sizes,
        series=[
            [item["batch"]["throughput_integrals_per_second"] for item in results_by_dim[dimension]]
            for dimension in dimensions
        ],
        series_labels=[f"d={dimension}" for dimension in dimensions],
        title=f"Batch Throughput by vmap Batch Size (level={args.level})",
        x_label="vmap batch size",
        y_label="Integrals per second",
        output_path=throughput_path,
        log_scale=True,
    )
    _write_line_svg(
        x_values=batch_sizes,
        series=[
            [item["batch"]["throughput_speedup_vs_single"] for item in results_by_dim[dimension]]
            for dimension in dimensions
        ],
        series_labels=[f"d={dimension}" for dimension in dimensions],
        title=f"vmap Speedup by Batch Size (level={args.level})",
        x_label="vmap batch size",
        y_label="Speedup",
        output_path=speedup_path,
        log_scale=True,
    )
    _write_line_svg(
        x_values=batch_sizes,
        series=[
            [item["batch"]["monitor"]["avg_gpu_util"] for item in results_by_dim[dimension]]
            for dimension in dimensions
        ],
        series_labels=[f"d={dimension}" for dimension in dimensions],
        title=f"Average GPU Utilization by Batch Size (level={args.level})",
        x_label="vmap batch size",
        y_label="GPU utilization [%]",
        output_path=gpu_util_path,
        log_scale=False,
    )
    _write_line_svg(
        x_values=batch_sizes,
        series=[
            [item["batch"]["monitor"]["min_pstate_numeric"] for item in results_by_dim[dimension]]
            for dimension in dimensions
        ],
        series_labels=[f"d={dimension}" for dimension in dimensions],
        title="Observed Pstate by Batch Size (lower is better)",
        x_label="vmap batch size",
        y_label="Pstate numeric",
        output_path=pstate_path,
        log_scale=False,
    )

    summary = {
        "experiment": "smolyak_gpu_batch_scaling",
        "started_at_utc": datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(Path.cwd().resolve()),
        "requested_platform": args.platform,
        "platform": observed_platform,
        "gpu_index": args.gpu_index,
        "requested_mode": args.requested_mode,
        "level": args.level,
        "dtype": args.dtype,
        "family": args.family,
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
        "dimensions": dimensions,
        "batch_sizes": batch_sizes,
        "warm_repeats": args.warm_repeats,
        "monitor_interval_ms": args.monitor_interval_ms,
        "monitor_min_duration_ms": args.monitor_min_duration_ms,
        "results": [item for values in results_by_dim.values() for item in values],
        "throughput_plot": str(throughput_path),
        "speedup_plot": str(speedup_path),
        "gpu_util_plot": str(gpu_util_path),
        "pstate_plot": str(pstate_path),
    }
    raw_results = [item for values in results_by_dim.values() for item in values]
    raw_csv_path = report_dir / "raw_results.csv"
    _write_results_csv(raw_results, raw_csv_path)
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")

    report_path = report_dir / "report.md"
    _write_markdown_report(
        dimensions=dimensions,
        batch_sizes=batch_sizes,
        results_by_dim=results_by_dim,
        output_path=report_path,
        figure_paths={
            "throughput": throughput_path,
            "speedup": speedup_path,
            "gpu_util": gpu_util_path,
            "pstate": pstate_path,
        },
    )

    print(
        json.dumps(
            {
                "report_dir": str(report_dir),
                "summary_json": str(summary_path),
                "report_md": str(report_path),
                "raw_csv": str(raw_csv_path),
                "throughput_plot": str(throughput_path),
                "speedup_plot": str(speedup_path),
                "gpu_util_plot": str(gpu_util_path),
                "pstate_plot": str(pstate_path),
                "num_cases": len(summary["results"]),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
