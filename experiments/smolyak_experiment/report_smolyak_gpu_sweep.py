#!/usr/bin/env python3
"""Run dimension sweeps for Smolyak on GPU, including vmap and Pstate monitoring."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from experiments.smolyak_experiment.cases import SUPPORTED_FAMILIES, make_family_bundle
from experiments.smolyak_experiment.weight_schemes import (
    SUPPORTED_WEIGHT_SCHEMES,
    format_dimension_weights,
    resolve_dimension_weights,
)

SCRIPT_DIR = Path(__file__).resolve().parent


def _default_dimensions() -> str:
    return ",".join(str(value) for value in range(1, 21))


def _parse_csv_ints(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Smolyak GPU throughput sweeps with vmap and Pstate monitoring.",
    )
    parser.add_argument(
        "--platform",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Execution platform. GPU is the default for this report.",
    )
    parser.add_argument(
        "--dimensions",
        default=_default_dimensions(),
        help="Comma-separated dimensions to test.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=3,
        help="Smolyak level to test across all dimensions.",
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
        help="Integrand family for the sweep.",
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
    parser.add_argument(
        "--dimension-weights",
        default=None,
        help="Optional comma-separated integer weights for anisotropic Smolyak term selection.",
    )
    parser.add_argument(
        "--weight-scheme",
        choices=list(SUPPORTED_WEIGHT_SCHEMES),
        default="none",
        help="Optional built-in anisotropic weight schedule.",
    )
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=1.0,
        help="Positive multiplier applied to the built-in weight schedule.",
    )
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
        "--vmap-batch-size",
        type=int,
        default=8,
        help="Batch size for vmap(integrate(f)).",
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
        help=(
            "Minimum total monitored runtime per single/batch benchmark. "
            "Additional compiled warm calls are issued for monitoring only."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "results" / "smolyak_gpu_sweep"),
        help="Directory for aggregate report assets.",
    )
    return parser


def _scalar(value: Any) -> float:
    array = value
    if hasattr(array, "block_until_ready"):
        array.block_until_ready()
    return float(np.asarray(array).reshape(-1)[0])


def _family_eval_factory(
    *,
    jnp: Any,
    integrator: Any,
    dimension: int,
    dtype: Any,
    family: str,
    gaussian_alpha: float,
    anisotropic_alpha_start: float,
    anisotropic_alpha_stop: float,
    shift_start: float,
    shift_stop: float,
    laplace_beta_start: float,
    laplace_beta_stop: float,
    coeff_start: float,
    coeff_stop: float,
) -> tuple[Callable[[Any], Any], Any, Any]:
    family_bundle = make_family_bundle(
        jnp=jnp,
        family=family,
        dimension=dimension,
        dtype=dtype,
        gaussian_alpha=gaussian_alpha,
        anisotropic_alpha_start=anisotropic_alpha_start,
        anisotropic_alpha_stop=anisotropic_alpha_stop,
        shift_start=shift_start,
        shift_stop=shift_stop,
        laplace_beta_start=laplace_beta_start,
        laplace_beta_stop=laplace_beta_stop,
        coeff_start=coeff_start,
        coeff_stop=coeff_stop,
    )

    def eval_one(scale: Any) -> Any:
        return family_bundle.eval_one(scale, integrator)

    return eval_one, family_bundle.single_scale, None


class _GpuMonitor:
    def __init__(self, gpu_index: int, interval_ms: int) -> None:
        self.gpu_index = gpu_index
        self.interval_ms = interval_ms
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None
        self._samples: list[dict[str, Any]] = []

    @staticmethod
    def _parse_int_field(text: str, suffix: str) -> int:
        stripped = text.strip()
        if stripped.endswith(suffix):
            stripped = stripped[: -len(suffix)].strip()
        return int(stripped)

    def _reader(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        for raw_line in self._process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 6:
                continue
            try:
                self._samples.append(
                    {
                        "timestamp": parts[0],
                        "index": int(parts[1]),
                        "pstate": parts[2],
                        "gpu_util": self._parse_int_field(parts[3], "%"),
                        "mem_util": self._parse_int_field(parts[4], "%"),
                        "mem_used_mb": self._parse_int_field(parts[5], "MiB"),
                    }
                )
            except ValueError:
                continue

    def __enter__(self) -> "_GpuMonitor":
        command = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=timestamp,index,pstate,utilization.gpu,utilization.memory,memory.used",
            "--format=csv,noheader",
            f"--loop-ms={self.interval_ms}",
        ]
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def summary(self) -> dict[str, Any]:
        if not self._samples:
            return {
                "sample_count": 0,
                "dominant_pstate": None,
                "min_pstate_numeric": None,
                "avg_gpu_util": None,
                "peak_gpu_util": None,
                "avg_mem_util": None,
                "peak_mem_util": None,
                "peak_mem_used_mb": None,
            }

        pstate_counter = Counter(sample["pstate"] for sample in self._samples)
        dominant_pstate = pstate_counter.most_common(1)[0][0]
        pstate_numeric = [
            int(sample["pstate"][1:])
            for sample in self._samples
            if sample["pstate"].startswith("P") and sample["pstate"][1:].isdigit()
        ]
        gpu_utils = [sample["gpu_util"] for sample in self._samples]
        mem_utils = [sample["mem_util"] for sample in self._samples]
        mem_used = [sample["mem_used_mb"] for sample in self._samples]
        return {
            "sample_count": len(self._samples),
            "dominant_pstate": dominant_pstate,
            "min_pstate_numeric": min(pstate_numeric) if pstate_numeric else None,
            "avg_gpu_util": float(sum(gpu_utils) / len(gpu_utils)),
            "peak_gpu_util": int(max(gpu_utils)),
            "avg_mem_util": float(sum(mem_utils) / len(mem_utils)),
            "peak_mem_util": int(max(mem_utils)),
            "peak_mem_used_mb": int(max(mem_used)),
        }


def _benchmark_compiled(
    *,
    compiled: Callable[[Any], Any],
    arg: Any,
    warm_repeats: int,
    monitor_factory: Callable[[], _GpuMonitor] | None,
    monitor_min_duration_ms: float,
) -> dict[str, Any]:
    first_start = time.perf_counter()
    first_value = compiled(arg)
    if hasattr(first_value, "block_until_ready"):
        first_value.block_until_ready()
    first_stop = time.perf_counter()
    first_ms = (first_stop - first_start) * 1000.0

    warm_times: list[float] = []
    monitor_summary: dict[str, Any] | None = None

    if monitor_factory is None:
        for _ in range(warm_repeats):
            start = time.perf_counter()
            value = compiled(arg)
            if hasattr(value, "block_until_ready"):
                value.block_until_ready()
            stop = time.perf_counter()
            warm_times.append((stop - start) * 1000.0)
            first_value = value
    else:
        for _ in range(warm_repeats):
            start = time.perf_counter()
            value = compiled(arg)
            if hasattr(value, "block_until_ready"):
                value.block_until_ready()
            stop = time.perf_counter()
            warm_times.append((stop - start) * 1000.0)
            first_value = value

        warm_runtime_ms = float(sum(warm_times) / len(warm_times)) if warm_times else 0.0
        monitored_repeats = warm_repeats
        if warm_runtime_ms > 0.0 and monitor_min_duration_ms > 0.0:
            monitored_repeats = max(
                warm_repeats,
                int(math.ceil(monitor_min_duration_ms / warm_runtime_ms)),
            )

        with monitor_factory() as monitor:
            for _ in range(monitored_repeats):
                value = compiled(arg)
                if hasattr(value, "block_until_ready"):
                    value.block_until_ready()
                first_value = value
            monitor_summary = monitor.summary()
        if monitor_summary is not None:
            monitor_summary["monitored_repeats"] = monitored_repeats

    warm_runtime_ms = float(sum(warm_times) / len(warm_times)) if warm_times else 0.0
    return {
        "value": _scalar(first_value),
        "first_call_ms": first_ms,
        "warm_runtime_ms": warm_runtime_ms,
        "compile_ms": max(0.0, first_ms - warm_runtime_ms),
        "monitor": monitor_summary,
    }


def _svg_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _scale_y(value: float | None, min_value: float, max_value: float, height: float, log_scale: bool) -> float:
    if value is None or not math.isfinite(value):
        return 0.0
    if log_scale:
        if value <= 0.0:
            return 0.0
        if max_value <= min_value:
            return height
        return (math.log10(value) - math.log10(min_value)) / (math.log10(max_value) - math.log10(min_value)) * height
    if max_value <= min_value:
        return height
    return (value - min_value) / (max_value - min_value) * height


def _write_line_svg(
    *,
    x_values: list[int],
    series: list[list[float | None]],
    series_labels: list[str],
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    log_scale: bool = False,
) -> None:
    width = 1120
    height = 560
    margin_left = 90
    margin_right = 40
    margin_top = 70
    margin_bottom = 108
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    colors = ["#1b5e20", "#0d47a1", "#b71c1c", "#6a1b9a"]

    valid_values = [
        float(value)
        for current_series in series
        for value in current_series
        if value is not None and math.isfinite(value) and (value > 0.0 if log_scale else True)
    ]
    min_value = min(valid_values) if valid_values else 0.0
    max_value = max(valid_values) if valid_values else 1.0
    if not log_scale and min_value == max_value:
        min_value = 0.0
        max_value = max(1.0, max_value)

    x_min = min(x_values)
    x_max = max(x_values)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="20" font-family="monospace">{_svg_escape(title)}</text>',
        f'<text x="24" y="{margin_top + plot_height / 2}" transform="rotate(-90 24 {margin_top + plot_height / 2})" text-anchor="middle" font-size="13" font-family="monospace">{_svg_escape(y_label)}{" (log scale)" if log_scale else ""}</text>',
        f'<text x="{margin_left + plot_width / 2}" y="{margin_top + plot_height + 48}" text-anchor="middle" font-size="13" font-family="monospace">{_svg_escape(x_label)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="1.5"/>',
    ]

    for tick_index in range(5):
        ratio = tick_index / 4 if 4 > 0 else 0.0
        y = margin_top + plot_height - ratio * plot_height
        if log_scale:
            tick_value = 10 ** (math.log10(min_value) + ratio * (math.log10(max_value) - math.log10(min_value)))
            tick_text = f"{tick_value:.3g}"
        else:
            tick_value = min_value + ratio * (max_value - min_value)
            tick_text = f"{tick_value:.3g}"
        lines.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#ddd" stroke-width="1"/>')
        lines.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="monospace">{tick_text}</text>')

    for dimension in x_values:
        if x_max == x_min:
            x = margin_left + plot_width / 2
        else:
            x = margin_left + (dimension - x_min) / (x_max - x_min) * plot_width
        lines.append(f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" y2="{margin_top + plot_height}" stroke="#f0f0f0" stroke-width="1"/>')
        lines.append(f'<text x="{x:.2f}" y="{margin_top + plot_height + 20}" text-anchor="middle" font-size="11" font-family="monospace">{dimension}</text>')

    for series_index, current_series in enumerate(series):
        points: list[str] = []
        for dimension, value in zip(x_values, current_series, strict=True):
            if value is None or (log_scale and value <= 0.0) or not math.isfinite(value):
                continue
            if x_max == x_min:
                x = margin_left + plot_width / 2
            else:
                x = margin_left + (dimension - x_min) / (x_max - x_min) * plot_width
            y = margin_top + plot_height - _scale_y(float(value), min_value, max_value, plot_height, log_scale)
            points.append(f"{x:.2f},{y:.2f}")
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{colors[series_index % len(colors)]}"/>')
        if points:
            lines.append(
                f'<polyline fill="none" stroke="{colors[series_index % len(colors)]}" stroke-width="2.5" points="{" ".join(points)}"/>'
            )

    legend_y = height - 24
    for series_index, label in enumerate(series_labels):
        x = margin_left + series_index * 220
        lines.append(f'<rect x="{x}" y="{legend_y - 12}" width="14" height="14" fill="{colors[series_index % len(colors)]}"/>')
        lines.append(f'<text x="{x + 22}" y="{legend_y}" font-size="12" font-family="monospace">{_svg_escape(label)}</text>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown_report(results: list[dict[str, Any]], output_path: Path, figure_paths: dict[str, Path]) -> None:
    best_batch = max(results, key=lambda item: float(item["batch"]["throughput_integrals_per_second"]))
    best_speedup = max(results, key=lambda item: float(item["batch"]["throughput_speedup_vs_single"]))
    p2_dims = [
        int(item["dimension"])
        for item in results
        if item["batch"]["monitor"] is not None and item["batch"]["monitor"]["dominant_pstate"] in {"P0", "P1", "P2"}
    ]
    lines = [
        "# Smolyak GPU Sweep Report",
        "",
        "## Method",
        "",
        "- Dimensions are swept across the requested range at a fixed Smolyak level.",
        "- `single` measures one compiled `integrate(f)` call.",
        "- `batch` measures `vmap(integrate(f))` over a batch of same-condition integrals with different Gaussian coefficients.",
        "- GPU monitoring samples `nvidia-smi` over additional compiled warm calls until a minimum monitor duration is reached.",
        "",
        "## Summary",
        "",
        f"- Cases run: {len(results)}",
        f"- Best batch throughput dimension: d={best_batch['dimension']} with {best_batch['batch']['throughput_integrals_per_second']:.2f} integrals/s",
        f"- Best vmap speedup dimension: d={best_speedup['dimension']} with {best_speedup['batch']['throughput_speedup_vs_single']:.2f}x",
        f"- Dimensions reaching P2 or better during batch warm runs: {p2_dims if p2_dims else 'none'}",
        f"- Materialization modes observed: {sorted({str(item['materialization_mode']) for item in results})}",
        "",
        "## Figures",
        "",
        f"![Runtime]({figure_paths['runtime'].name})",
        "",
        "- Read: x-axis is dimension `d`; y-axis is warm runtime in milliseconds on a log scale. Lower is better.",
        "",
        f"![Throughput]({figure_paths['throughput'].name})",
        "",
        "- Read: x-axis is dimension `d`; y-axis is integrals per second on a log scale. Higher is better.",
        "",
        f"![vmap speedup]({figure_paths['speedup'].name})",
        "",
        "- Read: x-axis is dimension `d`; y-axis is batch/single throughput speedup on a log scale. Values above 1 mean batching helps.",
        "",
        f"![GPU utilization]({figure_paths['gpu_util'].name})",
        "",
        "- Read: x-axis is dimension `d`; y-axis is average GPU utilization in percent. Higher usually means the workload is finally large enough for the GPU.",
        "",
        f"![Pstate]({figure_paths['pstate'].name})",
        "",
        "- Read: x-axis is dimension `d`; y-axis is the minimum observed Pstate index. Lower is better for performance states.",
        "",
        "## Case Table",
        "",
        "| Dimension | Materialize mode | Single warm ms | Batch warm ms | Single ints/s | Batch ints/s | vmap speedup | Batch dominant Pstate | Batch avg GPU util | Batch peak GPU util | Batch monitor samples |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]

    for item in results:
        monitor = item["batch"]["monitor"] or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["dimension"]),
                    str(item["materialization_mode"]),
                    f"{item['single']['warm_runtime_ms']:.6f}",
                    f"{item['batch']['warm_runtime_ms']:.6f}",
                    f"{item['single']['throughput_integrals_per_second']:.2f}",
                    f"{item['batch']['throughput_integrals_per_second']:.2f}",
                    f"{item['batch']['throughput_speedup_vs_single']:.3f}",
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
                    str(monitor.get("sample_count", 0)),
                ]
            )
            + " |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_results_csv(results: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "dimension",
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
    results: list[dict[str, Any]] = []

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
        eval_one, alpha_single, alpha_batch_seed = _family_eval_factory(
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
        del alpha_batch_seed
        batch_alphas = jnp.linspace(
            0.75,
            1.25,
            args.vmap_batch_size,
            dtype=dtype,
        )

        compiled_single = eqx.filter_jit(eval_one)
        compiled_batch = eqx.filter_jit(jax.vmap(eval_one))

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
        batch = _benchmark_compiled(
            compiled=compiled_batch,
            arg=batch_alphas,
            warm_repeats=args.warm_repeats,
            monitor_factory=monitor_factory,
            monitor_min_duration_ms=args.monitor_min_duration_ms,
        )

        single_throughput = 1000.0 / single["warm_runtime_ms"] if single["warm_runtime_ms"] > 0.0 else 0.0
        batch_throughput = (
            args.vmap_batch_size * 1000.0 / batch["warm_runtime_ms"]
            if batch["warm_runtime_ms"] > 0.0
            else 0.0
        )
        batch["throughput_integrals_per_second"] = batch_throughput
        batch["throughput_speedup_vs_single"] = (
            batch_throughput / single_throughput if single_throughput > 0.0 else 0.0
        )
        single["throughput_integrals_per_second"] = single_throughput

        results.append(
            {
                "dimension": dimension,
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

    dims = [int(item["dimension"]) for item in results]
    runtime_path = report_dir / "runtime.svg"
    throughput_path = report_dir / "throughput.svg"
    speedup_path = report_dir / "vmap_speedup.svg"
    gpu_util_path = report_dir / "gpu_util.svg"
    pstate_path = report_dir / "pstate.svg"

    _write_line_svg(
        x_values=dims,
        series=[
            [float(item["single"]["warm_runtime_ms"]) for item in results],
            [float(item["batch"]["warm_runtime_ms"]) for item in results],
        ],
        series_labels=["single", "vmap(batch)"],
        title=f"Warm Runtime by Dimension (level={args.level})",
        x_label="Dimension d",
        y_label="Warm runtime [ms]",
        output_path=runtime_path,
        log_scale=True,
    )
    _write_line_svg(
        x_values=dims,
        series=[
            [float(item["single"]["throughput_integrals_per_second"]) for item in results],
            [float(item["batch"]["throughput_integrals_per_second"]) for item in results],
        ],
        series_labels=["single throughput", "batch throughput"],
        title=f"Throughput by Dimension (level={args.level})",
        x_label="Dimension d",
        y_label="Integrals per second",
        output_path=throughput_path,
        log_scale=True,
    )
    _write_line_svg(
        x_values=dims,
        series=[
            [float(item["batch"]["throughput_speedup_vs_single"]) for item in results],
        ],
        series_labels=["vmap speedup"],
        title=f"vmap(integrate(f)) Speedup (level={args.level})",
        x_label="Dimension d",
        y_label="Speedup",
        output_path=speedup_path,
        log_scale=True,
    )
    _write_line_svg(
        x_values=dims,
        series=[
            [
                None if item["single"]["monitor"] is None else item["single"]["monitor"]["avg_gpu_util"]
                for item in results
            ],
            [
                None if item["batch"]["monitor"] is None else item["batch"]["monitor"]["avg_gpu_util"]
                for item in results
            ],
        ],
        series_labels=["single avg GPU util", "batch avg GPU util"],
        title=f"Average GPU Utilization (level={args.level})",
        x_label="Dimension d",
        y_label="GPU utilization [%]",
        output_path=gpu_util_path,
        log_scale=False,
    )
    _write_line_svg(
        x_values=dims,
        series=[
            [
                None if item["single"]["monitor"] is None else item["single"]["monitor"]["min_pstate_numeric"]
                for item in results
            ],
            [
                None if item["batch"]["monitor"] is None else item["batch"]["monitor"]["min_pstate_numeric"]
                for item in results
            ],
        ],
        series_labels=["single min Pstate", "batch min Pstate"],
        title=f"Observed Pstate by Dimension (lower is better)",
        x_label="Dimension d",
        y_label="Pstate numeric",
        output_path=pstate_path,
        log_scale=False,
    )

    summary = {
        "experiment": "smolyak_gpu_dimension_sweep",
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
        "vmap_batch_size": args.vmap_batch_size,
        "warm_repeats": args.warm_repeats,
        "monitor_interval_ms": args.monitor_interval_ms,
        "monitor_min_duration_ms": args.monitor_min_duration_ms,
        "dimensions": dimensions,
        "results": results,
        "runtime_plot": str(runtime_path),
        "throughput_plot": str(throughput_path),
        "speedup_plot": str(speedup_path),
        "gpu_util_plot": str(gpu_util_path),
        "pstate_plot": str(pstate_path),
    }
    raw_csv_path = report_dir / "raw_results.csv"
    _write_results_csv(results, raw_csv_path)
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")

    report_path = report_dir / "report.md"
    _write_markdown_report(
        results,
        report_path,
        {
            "runtime": runtime_path,
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
                "runtime_plot": str(runtime_path),
                "throughput_plot": str(throughput_path),
                "speedup_plot": str(speedup_path),
                "gpu_util_plot": str(gpu_util_path),
                "pstate_plot": str(pstate_path),
                "num_cases": len(results),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
