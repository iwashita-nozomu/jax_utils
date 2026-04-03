#!/usr/bin/env python3
"""Run Smolyak-vs-Monte-Carlo sweeps and render a compact report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def _parse_csv_ints(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def _parse_csv_strings(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("At least one string value is required.")
    return values


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep Smolyak-vs-Monte-Carlo comparisons and render a report.",
    )
    parser.add_argument(
        "--platform",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Execution platform.",
    )
    parser.add_argument(
        "--dimensions",
        default="2,3,4",
        help="Comma-separated dimensions to test.",
    )
    parser.add_argument(
        "--levels",
        default="2,3",
        help="Comma-separated Smolyak levels to test.",
    )
    parser.add_argument(
        "--families",
        default="quadratic,exponential",
        help="Comma-separated integrand families to test.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float32",
        help="Floating-point dtype.",
    )
    parser.add_argument(
        "--coeff-start",
        type=float,
        default=0.2,
        help="First coefficient for the exponential family.",
    )
    parser.add_argument(
        "--coeff-stop",
        type=float,
        default=0.8,
        help="Last coefficient for the exponential family.",
    )
    parser.add_argument(
        "--gaussian-alpha",
        type=float,
        default=0.8,
        help="Gaussian coefficient alpha for exp(-alpha ||x||^2).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16384,
        help="Smolyak chunk size forwarded to compare_smolyak_vs_mc.",
    )
    parser.add_argument(
        "--start-samples",
        type=int,
        default=1,
        help="Initial Monte Carlo sample count.",
    )
    parser.add_argument(
        "--growth-factor",
        type=float,
        default=4.0,
        help="Monte Carlo sample growth factor.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1 << 20,
        help="Upper bound for Monte Carlo sample search.",
    )
    parser.add_argument(
        "--warm-repeats",
        type=int,
        default=3,
        help="Number of warm runtime repeats.",
    )
    parser.add_argument(
        "--mc-seeds",
        type=int,
        default=8,
        help="Number of Monte Carlo seeds to average for each sample count.",
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=240.0,
        help="Per-case timeout. Failed cases are retained in the aggregate report.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "results" / "smolyak_vs_mc_report"),
        help="Directory for aggregate report assets.",
    )
    return parser


def _case_label(report: dict[str, Any]) -> str:
    family = str(report["family"]["family"])
    return f"{family}-d{report['dimension']}-l{report['level']}"


def _maybe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return numerator / denominator


def _matched_accuracy_speedup(report: dict[str, Any]) -> float | None:
    return _safe_ratio(
        _maybe_float(report["monte_carlo"]["warm_runtime_ms"]),
        _maybe_float(report["smolyak"]["warm_runtime_ms"]),
    )


def _same_budget_error_ratio(report: dict[str, Any]) -> float | None:
    return _safe_ratio(
        _maybe_float(report["monte_carlo_same_budget"]["absolute_error_mean"]),
        _maybe_float(report["smolyak"]["absolute_error"]),
    )


def _same_budget_runtime_ratio(report: dict[str, Any]) -> float | None:
    return _safe_ratio(
        _maybe_float(report["monte_carlo_same_budget"]["warm_runtime_ms"]),
        _maybe_float(report["smolyak"]["warm_runtime_ms"]),
    )


def _median(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not filtered:
        return None
    return float(statistics.median(filtered))


def _summarize_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    matched_speedups = [
        (report, speedup)
        for report in reports
        for speedup in [_matched_accuracy_speedup(report)]
        if speedup is not None
    ]
    same_budget_error_ratios = [
        (report, ratio)
        for report in reports
        for ratio in [_same_budget_error_ratio(report)]
        if ratio is not None
    ]
    same_budget_runtime_ratios = [
        (report, ratio)
        for report in reports
        for ratio in [_same_budget_runtime_ratio(report)]
        if ratio is not None
    ]

    matched_best = max(matched_speedups, key=lambda item: item[1], default=None)
    same_budget_best_error = max(same_budget_error_ratios, key=lambda item: item[1], default=None)
    same_budget_best_runtime = max(same_budget_runtime_ratios, key=lambda item: item[1], default=None)

    return {
        "cases_run": len(reports),
        "matched_accuracy": {
            "smolyak_faster_cases": sum(
                1
                for report in reports
                if bool(report["summary"]["smolyak_faster_on_warm_runtime"])
            ),
            "monte_carlo_faster_or_unmatched_cases": sum(
                1
                for report in reports
                if not bool(report["summary"]["smolyak_faster_on_warm_runtime"])
            ),
            "comparable_cases": len(matched_speedups),
            "median_runtime_ratio_mc_over_smolyak": _median(
                [item[1] for item in matched_speedups]
            ),
            "best_speedup_case": None
            if matched_best is None
            else {
                "case": _case_label(matched_best[0]),
                "runtime_ratio_mc_over_smolyak": matched_best[1],
            },
        },
        "same_budget": {
            "smolyak_more_accurate_cases": sum(
                1
                for report in reports
                if bool(report["summary"]["smolyak_more_accurate_same_budget"])
            ),
            "monte_carlo_more_accurate_or_tied_cases": sum(
                1
                for report in reports
                if not bool(report["summary"]["smolyak_more_accurate_same_budget"])
            ),
            "smolyak_faster_cases": sum(
                1
                for report in reports
                if bool(report["summary"]["smolyak_faster_same_budget"])
            ),
            "monte_carlo_faster_or_tied_cases": sum(
                1
                for report in reports
                if not bool(report["summary"]["smolyak_faster_same_budget"])
            ),
            "comparable_error_cases": len(same_budget_error_ratios),
            "comparable_runtime_cases": len(same_budget_runtime_ratios),
            "median_error_ratio_mc_over_smolyak": _median(
                [item[1] for item in same_budget_error_ratios]
            ),
            "median_runtime_ratio_mc_over_smolyak": _median(
                [item[1] for item in same_budget_runtime_ratios]
            ),
            "best_accuracy_case": None
            if same_budget_best_error is None
            else {
                "case": _case_label(same_budget_best_error[0]),
                "error_ratio_mc_over_smolyak": same_budget_best_error[1],
            },
            "best_runtime_case": None
            if same_budget_best_runtime is None
            else {
                "case": _case_label(same_budget_best_runtime[0]),
                "runtime_ratio_mc_over_smolyak": same_budget_best_runtime[1],
            },
        },
    }


def _failure_report(
    *,
    platform: str,
    dimension: int,
    level: int,
    family: str,
    dtype: str,
    gaussian_alpha: float,
    chunk_size: int,
    warm_repeats: int,
    mc_seeds: int,
    failure_message: str,
) -> dict[str, Any]:
    failure_summary = failure_message.strip().splitlines()[-1] if failure_message.strip() else "failure"
    family_metadata: dict[str, Any] = {"family": family}
    if family == "gaussian":
        family_metadata["alpha"] = gaussian_alpha

    return {
        "analytic_value": None,
        "chunk_size": chunk_size,
        "dimension": dimension,
        "dtype": dtype,
        "experiment": "smolyak_vs_monte_carlo_failed_case",
        "failed": True,
        "failure_message": failure_message,
        "failure_summary": failure_summary,
        "family": family_metadata,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "mc_seeds": mc_seeds,
        "monte_carlo": {
            "search_skipped": False,
            "chosen_num_samples": None,
            "value_mean": None,
            "value_std": None,
            "absolute_error": None,
            "absolute_error_std": None,
            "relative_error": None,
            "relative_error_std": None,
            "first_call_ms": None,
            "warm_runtime_ms": None,
            "compile_ms": None,
            "history": [],
        },
        "monte_carlo_same_budget": {
            "num_samples": None,
            "num_seeds": mc_seeds,
            "value_mean": None,
            "value_std": None,
            "absolute_error_mean": None,
            "absolute_error_std": None,
            "relative_error_mean": None,
            "relative_error_std": None,
            "first_call_ms": None,
            "warm_runtime_ms": None,
            "compile_ms": None,
        },
        "output_json": None,
        "platform": platform,
        "smolyak": {
            "num_terms": None,
            "num_evaluation_points": None,
            "vectorized_ndim": None,
            "max_vectorized_points": None,
            "storage_bytes": None,
            "uses_materialized_plan": None,
            "materialized_point_count": None,
            "value": None,
            "absolute_error": None,
            "relative_error": None,
            "first_call_ms": None,
            "warm_runtime_ms": None,
            "compile_ms": None,
        },
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "smolyak_faster_on_warm_runtime": False,
            "smolyak_more_accurate_same_budget": False,
            "smolyak_faster_same_budget": False,
            "smolyak_error": None,
            "monte_carlo_error": None,
            "monte_carlo_same_budget_error": None,
        },
        "warm_repeats": warm_repeats,
        "workspace_root": str(Path.cwd().resolve()),
    }


def _run_case(
    *,
    platform: str,
    dimension: int,
    level: int,
    family: str,
    dtype: str,
    coeff_start: float,
    coeff_stop: float,
    gaussian_alpha: float,
    chunk_size: int,
    start_samples: int,
    growth_factor: float,
    max_samples: int,
    warm_repeats: int,
    mc_seeds: int,
    case_timeout_seconds: float,
    output_dir: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "experiments.smolyak_experiment.compare_smolyak_vs_mc",
        "--platform",
        platform,
        "--dimension",
        str(dimension),
        "--level",
        str(level),
        "--family",
        family,
        "--dtype",
        dtype,
        "--coeff-start",
        str(coeff_start),
        "--coeff-stop",
        str(coeff_stop),
        "--gaussian-alpha",
        str(gaussian_alpha),
        "--chunk-size",
        str(chunk_size),
        "--start-samples",
        str(start_samples),
        "--growth-factor",
        str(growth_factor),
        "--max-samples",
        str(max_samples),
        "--warm-repeats",
        str(warm_repeats),
        "--mc-seeds",
        str(mc_seeds),
        "--output-dir",
        str(output_dir),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=case_timeout_seconds,
        )
        report = json.loads(completed.stdout)
        report["failed"] = False
        return report
    except subprocess.CalledProcessError as exc:
        failure_message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        return _failure_report(
            platform=platform,
            dimension=dimension,
            level=level,
            family=family,
            dtype=dtype,
            gaussian_alpha=gaussian_alpha,
            chunk_size=chunk_size,
            warm_repeats=warm_repeats,
            mc_seeds=mc_seeds,
            failure_message=failure_message,
        )
    except subprocess.TimeoutExpired:
        return _failure_report(
            platform=platform,
            dimension=dimension,
            level=level,
            family=family,
            dtype=dtype,
            gaussian_alpha=gaussian_alpha,
            chunk_size=chunk_size,
            warm_repeats=warm_repeats,
            mc_seeds=mc_seeds,
            failure_message=f"timeout after {case_timeout_seconds:.1f}s",
        )


def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _log_scaled_height(value: float | None, min_positive: float, max_positive: float, height: float) -> float:
    if value is None or not math.isfinite(value) or value <= 0.0:
        return 0.0
    if max_positive <= min_positive:
        return height
    log_min = math.log10(min_positive)
    log_max = math.log10(max_positive)
    log_value = math.log10(value)
    return (log_value - log_min) / (log_max - log_min) * height


def _write_grouped_bar_svg(
    *,
    labels: list[str],
    series: list[list[float | None]],
    series_labels: list[str],
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    width = max(960, 160 * len(labels))
    height = 520
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 150
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    positive_values = [
        float(value)
        for current_series in series
        for value in current_series
        if value is not None and math.isfinite(value) and value > 0.0
    ]
    min_positive = min(positive_values) if positive_values else 1.0
    max_positive = max(positive_values) if positive_values else 1.0

    group_width = plot_width / max(1, len(labels))
    bar_width = group_width / max(2, len(series) + 1)
    colors = ["#1b5e20", "#0d47a1", "#b71c1c", "#6a1b9a"]

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-size="20" font-family="monospace">{_svg_escape(title)}</text>',
        f'<text x="24" y="{margin_top + plot_height / 2}" transform="rotate(-90 24 {margin_top + plot_height / 2})" text-anchor="middle" font-size="13" font-family="monospace">{_svg_escape(y_label)} (log scale)</text>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="1.5"/>',
    ]

    for tick_index in range(5):
        tick_ratio = tick_index / 4 if 4 > 0 else 0.0
        y = margin_top + plot_height - tick_ratio * plot_height
        value = 10 ** (math.log10(min_positive) + tick_ratio * (math.log10(max_positive) - math.log10(min_positive)))
        lines.append(f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#ddd" stroke-width="1"/>')
        lines.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="monospace">{value:.3g}</text>')

    for group_index, label in enumerate(labels):
        base_x = margin_left + group_index * group_width
        for series_index, current_series in enumerate(series):
            value = current_series[group_index]
            bar_height = _log_scaled_height(value, min_positive, max_positive, plot_height)
            x = base_x + (series_index + 0.2) * bar_width
            y = margin_top + plot_height - bar_height
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width * 0.8:.2f}" height="{bar_height:.2f}" fill="{colors[series_index % len(colors)]}" opacity="0.85"/>'
            )
        label_x = base_x + group_width / 2
        label_y = margin_top + plot_height + 20
        lines.append(
            f'<text x="{label_x:.2f}" y="{label_y:.2f}" transform="rotate(30 {label_x:.2f} {label_y:.2f})" text-anchor="start" font-size="11" font-family="monospace">{_svg_escape(label)}</text>'
        )

    legend_x = margin_left
    legend_y = height - 28
    for series_index, label in enumerate(series_labels):
        x = legend_x + series_index * 220
        lines.append(f'<rect x="{x}" y="{legend_y - 12}" width="14" height="14" fill="{colors[series_index % len(colors)]}" opacity="0.85"/>')
        lines.append(f'<text x="{x + 22}" y="{legend_y}" font-size="12" font-family="monospace">{_svg_escape(label)}</text>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_runtime_plot(reports: list[dict[str, Any]], output_path: Path) -> None:
    labels = [_case_label(report) for report in reports]
    smolyak_times = [float(report["smolyak"]["warm_runtime_ms"]) for report in reports]
    mc_times = [
        float(report["monte_carlo"]["warm_runtime_ms"])
        if report["monte_carlo"]["warm_runtime_ms"] is not None
        else None
        for report in reports
    ]
    _write_grouped_bar_svg(
        labels=labels,
        series=[smolyak_times, mc_times],
        series_labels=["Smolyak", "Monte Carlo"],
        title="Warm Runtime at Matched Accuracy",
        y_label="Warm runtime [ms]",
        output_path=output_path,
    )


def _write_speedup_plot(reports: list[dict[str, Any]], output_path: Path) -> None:
    labels = [_case_label(report) for report in reports]
    speedups: list[float | None] = []
    for report in reports:
        smolyak_time = float(report["smolyak"]["warm_runtime_ms"])
        mc_time = report["monte_carlo"]["warm_runtime_ms"]
        if mc_time is None or smolyak_time <= 0.0:
            speedups.append(None)
        else:
            speedups.append(float(mc_time) / smolyak_time)
    _write_grouped_bar_svg(
        labels=labels,
        series=[speedups],
        series_labels=["MC / Smolyak"],
        title="Warm Runtime Speedup at Matched Accuracy",
        y_label="Speedup",
        output_path=output_path,
    )


def _write_sample_plot(reports: list[dict[str, Any]], output_path: Path) -> None:
    labels = [_case_label(report) for report in reports]
    smolyak_points = [float(report["smolyak"]["num_evaluation_points"]) for report in reports]
    mc_samples = [
        float(report["monte_carlo"]["chosen_num_samples"])
        if report["monte_carlo"]["chosen_num_samples"] is not None
        else None
        for report in reports
    ]
    _write_grouped_bar_svg(
        labels=labels,
        series=[smolyak_points, mc_samples],
        series_labels=["Smolyak points", "MC matched samples"],
        title="Evaluation Budget at Matched Accuracy",
        y_label="Points / samples",
        output_path=output_path,
    )


def _write_markdown_report(
    reports: list[dict[str, Any]],
    output_path: Path,
    runtime_plot: Path,
    speedup_plot: Path,
    sample_plot: Path,
    aggregates: dict[str, Any],
) -> None:
    successful_reports = [report for report in reports if not bool(report.get("failed"))]
    failed_reports = [report for report in reports if bool(report.get("failed"))]
    matched = aggregates["matched_accuracy"]
    same_budget = aggregates["same_budget"]
    lines = [
        "# Smolyak vs Monte Carlo Report",
        "",
        "## Method",
        "",
        "- Smolyak integrates by scanning the prefix dimensions and vectorizing the last 1-3 dimensions together, with the suffix width chosen from the realized term plan so it stays below the OOM budget.",
        "- Same-budget view compares Smolyak against Monte Carlo with exactly `SmolyakIntegrator.num_evaluation_points` samples.",
        "- Monte Carlo increases the sample count geometrically until its mean absolute error across multiple seeds is no larger than the Smolyak absolute error for the same integrand.",
        "- Runtime comparison uses warm JIT runtime, while compile and first-call timings are also preserved in the raw JSON files.",
        "",
        "## Summary",
        "",
        f"- Successful cases: {aggregates['cases_run']} / {len(reports)}",
        f"- Failed cases: {len(failed_reports)}",
        f"- Matched accuracy: Smolyak faster in {matched['smolyak_faster_cases']} cases, Monte Carlo faster or unmatched in {matched['monte_carlo_faster_or_unmatched_cases']} cases.",
        f"- Same budget: Smolyak more accurate in {same_budget['smolyak_more_accurate_cases']} cases and faster in {same_budget['smolyak_faster_cases']} cases.",
        "",
        "## Figures",
        "",
        f"![Warm runtime]({runtime_plot.name})",
        "",
        f"![Speedup]({speedup_plot.name})",
        "",
        f"![Points vs samples]({sample_plot.name})",
        "",
        "## Observations",
        "",
    ]
    if matched["median_runtime_ratio_mc_over_smolyak"] is not None:
        lines.append(
            "- Matched-accuracy での Monte Carlo / Smolyak warm runtime 比中央値は "
            f"{float(matched['median_runtime_ratio_mc_over_smolyak']):.3f} 倍でした。"
        )
    if matched["best_speedup_case"] is not None:
        lines.append(
            "- Matched-accuracy で最も Smolyak が速かったのは "
            f"{matched['best_speedup_case']['case']} で、Monte Carlo / Smolyak warm runtime 比は "
            f"{float(matched['best_speedup_case']['runtime_ratio_mc_over_smolyak']):.3f} 倍でした。"
        )
    if same_budget["median_error_ratio_mc_over_smolyak"] is not None:
        lines.append(
            "- Same-budget での Monte Carlo / Smolyak absolute-error 比中央値は "
            f"{float(same_budget['median_error_ratio_mc_over_smolyak']):.3f} 倍でした。"
        )
    if same_budget["median_runtime_ratio_mc_over_smolyak"] is not None:
        lines.append(
            "- Same-budget での Monte Carlo / Smolyak warm runtime 比中央値は "
            f"{float(same_budget['median_runtime_ratio_mc_over_smolyak']):.3f} 倍でした。"
        )
    if same_budget["best_accuracy_case"] is not None:
        lines.append(
            "- Same-budget で最も Smolyak が精度優位だったのは "
            f"{same_budget['best_accuracy_case']['case']} で、Monte Carlo / Smolyak error 比は "
            f"{float(same_budget['best_accuracy_case']['error_ratio_mc_over_smolyak']):.3f} 倍でした。"
        )
    if same_budget["best_runtime_case"] is not None:
        lines.append(
            "- Same-budget で最も Smolyak が速度優位だったのは "
            f"{same_budget['best_runtime_case']['case']} で、Monte Carlo / Smolyak runtime 比は "
            f"{float(same_budget['best_runtime_case']['runtime_ratio_mc_over_smolyak']):.3f} 倍でした。"
        )
    lines.extend(
        [
            "",
            "## Matched-Accuracy Table",
            "",
            "| Case | Smolyak warm ms | MC warm ms | MC/Smolyak runtime ratio | Smolyak points | MC matched samples | Smolyak error | MC error |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for report in reports:
        if report.get("failed"):
            continue
        smolyak_time = float(report["smolyak"]["warm_runtime_ms"])
        mc_time = _maybe_float(report["monte_carlo"]["warm_runtime_ms"])
        speedup = _matched_accuracy_speedup(report)
        mc_error = _maybe_float(report["monte_carlo"]["absolute_error"])
        lines.append(
            "| "
            + " | ".join(
                [
                    _case_label(report),
                    f"{smolyak_time:.6f}",
                    f"{float(mc_time):.6f}" if mc_time is not None else "n/a",
                    f"{float(speedup):.3f}" if speedup is not None else "n/a",
                    str(int(report["smolyak"]["num_evaluation_points"])),
                    str(report["monte_carlo"]["chosen_num_samples"]),
                    f"{float(report['smolyak']['absolute_error']):.6e}",
                    f"{float(mc_error):.6e}" if mc_error is not None else "n/a",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Same-Budget Table",
            "",
            "| Case | Smolyak points | Smolyak error | MC same-budget error | MC/Smolyak error ratio | Smolyak warm ms | MC same-budget warm ms | MC/Smolyak runtime ratio |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for report in reports:
        if report.get("failed"):
            continue
        same_budget_error = _maybe_float(report["monte_carlo_same_budget"]["absolute_error_mean"])
        same_budget_time = _maybe_float(report["monte_carlo_same_budget"]["warm_runtime_ms"])
        same_budget_error_ratio = _same_budget_error_ratio(report)
        same_budget_runtime_ratio = _same_budget_runtime_ratio(report)
        lines.append(
            "| "
            + " | ".join(
                [
                    _case_label(report),
                    str(int(report["smolyak"]["num_evaluation_points"])),
                    f"{float(report['smolyak']['absolute_error']):.6e}",
                    f"{float(same_budget_error):.6e}" if same_budget_error is not None else "n/a",
                    f"{float(same_budget_error_ratio):.3f}" if same_budget_error_ratio is not None else "n/a",
                    f"{float(report['smolyak']['warm_runtime_ms']):.6f}",
                    f"{float(same_budget_time):.6f}" if same_budget_time is not None else "n/a",
                    f"{float(same_budget_runtime_ratio):.3f}" if same_budget_runtime_ratio is not None else "n/a",
                ]
            )
            + " |"
        )

    if failed_reports:
        lines.extend(
            [
                "",
                "## Failed Cases",
                "",
                "| Case | Failure |",
                "| --- | --- |",
            ]
        )
        for report in failed_reports:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _case_label(report),
                        str(report.get("failure_summary", report.get("failure_message", "failure"))).replace("|", "/"),
                    ]
                )
                + " |"
            )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_root = Path(args.output_dir)
    started_at = time.time()
    run_label = datetime.fromtimestamp(started_at, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = output_root / f"report_{run_label}"
    case_output_dir = report_dir / "cases"
    report_dir.mkdir(parents=True, exist_ok=True)
    case_output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, Any]] = []
    for family in _parse_csv_strings(args.families):
        for dimension in _parse_csv_ints(args.dimensions):
            for level in _parse_csv_ints(args.levels):
                reports.append(
                    _run_case(
                        platform=args.platform,
                        dimension=dimension,
                        level=level,
                        family=family,
                        dtype=args.dtype,
                        coeff_start=args.coeff_start,
                        coeff_stop=args.coeff_stop,
                        gaussian_alpha=args.gaussian_alpha,
                        chunk_size=args.chunk_size,
                        start_samples=args.start_samples,
                        growth_factor=args.growth_factor,
                        max_samples=args.max_samples,
                        warm_repeats=args.warm_repeats,
                        mc_seeds=args.mc_seeds,
                        case_timeout_seconds=args.case_timeout_seconds,
                        output_dir=case_output_dir,
                    )
                )

    successful_reports = [report for report in reports if not bool(report.get("failed"))]
    runtime_plot = report_dir / "warm_runtime.svg"
    speedup_plot = report_dir / "speedup.svg"
    sample_plot = report_dir / "points_vs_samples.svg"
    _write_runtime_plot(successful_reports, runtime_plot)
    _write_speedup_plot(successful_reports, speedup_plot)
    _write_sample_plot(successful_reports, sample_plot)
    aggregates = _summarize_reports(successful_reports)

    summary = {
        "experiment": "smolyak_vs_monte_carlo_report",
        "started_at_utc": datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(Path.cwd().resolve()),
        "platform": args.platform,
        "dtype": args.dtype,
        "dimensions": _parse_csv_ints(args.dimensions),
        "levels": _parse_csv_ints(args.levels),
        "families": _parse_csv_strings(args.families),
        "gaussian_alpha": args.gaussian_alpha,
        "chunk_size": args.chunk_size,
        "mc_seeds": args.mc_seeds,
        "case_timeout_seconds": args.case_timeout_seconds,
        "cases": reports,
        "aggregates": aggregates,
        "runtime_plot": str(runtime_plot),
        "speedup_plot": str(speedup_plot),
        "points_vs_samples_plot": str(sample_plot),
    }
    summary_json = report_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")

    markdown_path = report_dir / "report.md"
    _write_markdown_report(reports, markdown_path, runtime_plot, speedup_plot, sample_plot, aggregates)

    print(
        json.dumps(
            {
                "report_dir": str(report_dir),
                "summary_json": str(summary_json),
                "report_md": str(markdown_path),
                "runtime_plot": str(runtime_plot),
                "speedup_plot": str(speedup_plot),
                "points_vs_samples_plot": str(sample_plot),
                "num_cases": len(reports),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
