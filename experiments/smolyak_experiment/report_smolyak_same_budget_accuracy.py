#!/usr/bin/env python3
"""Run same-budget Smolyak-vs-Monte-Carlo sweeps across dimensions, levels, and dtypes."""

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
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent


def _default_dimensions() -> str:
    return ",".join(str(value) for value in range(1, 21))


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run same-budget Smolyak-vs-Monte-Carlo sweeps across dimensions, levels, and dtypes."
        ),
    )
    parser.add_argument(
        "--platform",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Execution platform.",
    )
    parser.add_argument(
        "--dimensions",
        default=_default_dimensions(),
        help="Comma-separated dimensions to test.",
    )
    parser.add_argument(
        "--levels",
        default="2,3,4",
        help="Comma-separated Smolyak levels to test.",
    )
    parser.add_argument(
        "--dtypes",
        default="float32,float64",
        help="Comma-separated dtypes to test.",
    )
    parser.add_argument(
        "--family",
        choices=["gaussian", "quadratic", "exponential"],
        default="gaussian",
        help="Analytic integrand family.",
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
        "--warm-repeats",
        type=int,
        default=3,
        help="Number of warm runtime repeats.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16384,
        help="Smolyak chunk size forwarded to compare_smolyak_vs_mc.",
    )
    parser.add_argument(
        "--mc-seeds",
        type=int,
        default=8,
        help="Number of Monte Carlo seeds to average at the same sample budget.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base PRNG seed for Monte Carlo.",
    )
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=120.0,
        help="Per-case timeout. Timed out cases are recorded as missing results.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "results" / "smolyak_same_budget_accuracy"),
        help="Directory for aggregate report assets.",
    )
    parser.add_argument(
        "--resume-report-dir",
        default=None,
        help=(
            "Existing report directory to resume in-place. If set, cached case JSONs under "
            "`<report-dir>/cases` are reused and only missing cases are executed."
        ),
    )
    return parser


def _case_key(dtype: str, level: int, dimension: int) -> tuple[str, int, int]:
    return (str(dtype), int(level), int(dimension))


def _case_storage_path(case_output_dir: Path, dtype: str, level: int, dimension: int) -> Path:
    return case_output_dir / f"case_{dtype}_l{int(level):02d}_d{int(dimension):03d}.json"


def _load_existing_case_reports(case_output_dir: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    reports_with_priority: dict[tuple[str, int, int], tuple[int, float, dict[str, Any]]] = {}
    if not case_output_dir.exists():
        return {}

    for path in sorted(case_output_dir.glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict):
            continue
        dtype = report.get("dtype")
        level = report.get("level")
        dimension = report.get("dimension")
        if dtype is None or level is None or dimension is None:
            continue
        key = _case_key(str(dtype), int(level), int(dimension))
        normalized = dict(report)
        normalized["failed"] = bool(normalized.get("failed", False))
        normalized["output_json"] = str(path)
        priority = 1 if path.name.startswith("case_") else 0
        rank = (priority, path.stat().st_mtime, normalized)
        current = reports_with_priority.get(key)
        if current is None or (rank[0], rank[1]) >= (current[0], current[1]):
            reports_with_priority[key] = rank

    return {key: item[2] for key, item in reports_with_priority.items()}


def _write_case_report(case_output_dir: Path, report: dict[str, Any]) -> Path:
    case_path = _case_storage_path(
        case_output_dir,
        str(report["dtype"]),
        int(report["level"]),
        int(report["dimension"]),
    )
    payload = dict(report)
    payload["output_json"] = str(case_path)
    case_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")
    report["output_json"] = str(case_path)
    return case_path


def _failure_report(
    *,
    platform: str,
    dimension: int,
    level: int,
    dtype: str,
    family: str,
    chunk_size: int,
    mc_seeds: int,
    warm_repeats: int,
    failure_message: str,
) -> dict[str, Any]:
    failure_summary = failure_message.strip().splitlines()[-1] if failure_message.strip() else "failure"
    return {
        "analytic_value": None,
        "chunk_size": chunk_size,
        "dimension": dimension,
        "dtype": dtype,
        "experiment": "smolyak_same_budget_failed_case",
        "failed": True,
        "failure_message": failure_message,
        "failure_summary": failure_summary,
        "family": {"family": family},
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "mc_seeds": mc_seeds,
        "monte_carlo": {
            "absolute_error": None,
            "absolute_error_std": None,
            "chosen_num_samples": None,
            "compile_ms": None,
            "first_call_ms": None,
            "history": [],
            "relative_error": None,
            "relative_error_std": None,
            "search_skipped": True,
            "value_mean": None,
            "value_std": None,
            "warm_runtime_ms": None,
        },
        "monte_carlo_same_budget": {
            "absolute_error_mean": None,
            "absolute_error_std": None,
            "compile_ms": None,
            "first_call_ms": None,
            "num_samples": None,
            "num_seeds": mc_seeds,
            "relative_error_mean": None,
            "relative_error_std": None,
            "value_mean": None,
            "value_std": None,
            "warm_runtime_ms": None,
        },
        "output_json": None,
        "platform": platform,
        "smolyak": {
            "absolute_error": None,
            "compile_ms": None,
            "first_call_ms": None,
            "max_vectorized_points": None,
            "num_evaluation_points": None,
            "num_terms": None,
            "relative_error": None,
            "value": None,
            "vectorized_ndim": None,
            "warm_runtime_ms": None,
        },
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "monte_carlo_error": None,
            "monte_carlo_same_budget_error": None,
            "smolyak_error": None,
            "smolyak_faster_on_warm_runtime": False,
            "smolyak_faster_same_budget": False,
            "smolyak_more_accurate_same_budget": False,
        },
        "warm_repeats": warm_repeats,
        "workspace_root": str(Path.cwd().resolve()),
    }


def _run_case(
    *,
    platform: str,
    dimension: int,
    level: int,
    dtype: str,
    family: str,
    coeff_start: float,
    coeff_stop: float,
    gaussian_alpha: float,
    chunk_size: int,
    warm_repeats: int,
    mc_seeds: int,
    seed: int,
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
        "--dtype",
        dtype,
        "--family",
        family,
        "--coeff-start",
        str(coeff_start),
        "--coeff-stop",
        str(coeff_stop),
        "--gaussian-alpha",
        str(gaussian_alpha),
        "--chunk-size",
        str(chunk_size),
        "--warm-repeats",
        str(warm_repeats),
        "--mc-seeds",
        str(mc_seeds),
        "--seed",
        str(seed),
        "--skip-matched-accuracy",
        "--no-write",
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
            dtype=dtype,
            family=family,
            chunk_size=chunk_size,
            mc_seeds=mc_seeds,
            warm_repeats=warm_repeats,
            failure_message=failure_message,
        )
    except subprocess.TimeoutExpired:
        return _failure_report(
            platform=platform,
            dimension=dimension,
            level=level,
            dtype=dtype,
            family=family,
            chunk_size=chunk_size,
            mc_seeds=mc_seeds,
            warm_repeats=warm_repeats,
            failure_message=f"timeout after {case_timeout_seconds:.1f}s",
        )


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
    width = 1180
    height = 620
    margin_left = 90
    margin_right = 40
    margin_top = 70
    margin_bottom = 140
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    colors = [
        "#1b5e20",
        "#0d47a1",
        "#b71c1c",
        "#6a1b9a",
        "#ef6c00",
        "#00838f",
        "#5d4037",
        "#455a64",
    ]

    valid_values = [
        float(value)
        for current_series in series
        for value in current_series
        if value is not None and math.isfinite(value) and (value > 0.0 if log_scale else True)
    ]
    if log_scale and not valid_values:
        min_value = 1e-12
        max_value = 1.0
    else:
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

    legend_y = height - 68
    for series_index, label in enumerate(series_labels):
        row = series_index // 3
        col = series_index % 3
        x = margin_left + col * 310
        y = legend_y + row * 22
        lines.append(f'<rect x="{x}" y="{y - 12}" width="14" height="14" fill="{colors[series_index % len(colors)]}"/>')
        lines.append(f'<text x="{x + 22}" y="{y}" font-size="12" font-family="monospace">{_svg_escape(label)}</text>')

    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _index_reports(reports: list[dict[str, Any]]) -> dict[tuple[str, int, int], dict[str, Any]]:
    return {
        (str(report["dtype"]), int(report["level"]), int(report["dimension"])): report
        for report in reports
    }


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _series_for_dtype_level(
    *,
    reports_by_key: dict[tuple[str, int, int], dict[str, Any]],
    dimensions: list[int],
    dtypes: list[str],
    levels: list[int],
    value_fn: Callable[[dict[str, Any]], float | None],
) -> tuple[list[list[float | None]], list[str]]:
    series: list[list[float | None]] = []
    labels: list[str] = []
    for dtype in dtypes:
        for level in levels:
            labels.append(f"{dtype}-l{level}")
            series.append(
                [
                    value_fn(reports_by_key[(dtype, level, dimension)])
                    for dimension in dimensions
                ]
            )
    return series, labels


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _case_id(report: dict[str, Any]) -> str:
    return f"{report['dtype']}-d{report['dimension']}-l{report['level']}"


def _generate_observations(
    *,
    reports: list[dict[str, Any]],
    dimensions: list[int],
    levels: list[int],
    dtypes: list[str],
) -> list[str]:
    reports_by_key = _index_reports(reports)
    successful_reports = [report for report in reports if not bool(report.get("failed"))]
    failed_reports = [report for report in reports if bool(report.get("failed"))]
    if not successful_reports:
        return ["全ケースが失敗したため、数値的な考察は生成できませんでした。"]

    overall_more_accurate = sum(
        1 for report in successful_reports if bool(report["summary"]["smolyak_more_accurate_same_budget"])
    )
    overall_faster = sum(
        1 for report in successful_reports if bool(report["summary"]["smolyak_faster_same_budget"])
    )
    error_ratios = [
        float(report["monte_carlo_same_budget"]["absolute_error_mean"]) / float(report["smolyak"]["absolute_error"])
        for report in successful_reports
        if float(report["smolyak"]["absolute_error"]) > 0.0
    ]
    runtime_ratios = [
        float(report["monte_carlo_same_budget"]["warm_runtime_ms"]) / float(report["smolyak"]["warm_runtime_ms"])
        for report in successful_reports
        if float(report["smolyak"]["warm_runtime_ms"]) > 0.0
    ]
    best_error_advantage = max(
        successful_reports,
        key=lambda item: float(item["monte_carlo_same_budget"]["absolute_error_mean"]) / max(float(item["smolyak"]["absolute_error"]), 1e-30),
    )
    strongest_mc_speed_advantage = min(
        successful_reports,
        key=lambda item: float(item["monte_carlo_same_budget"]["warm_runtime_ms"]) / max(float(item["smolyak"]["warm_runtime_ms"]), 1e-30),
    )
    mc_accuracy_wins = [
        report
        for report in successful_reports
        if float(report["monte_carlo_same_budget"]["absolute_error_mean"]) < float(report["smolyak"]["absolute_error"])
    ]
    mc_speed_wins = [
        report
        for report in successful_reports
        if float(report["monte_carlo_same_budget"]["warm_runtime_ms"]) < float(report["smolyak"]["warm_runtime_ms"])
    ]

    median_error_ratio = _median(error_ratios)
    median_runtime_ratio = _median(runtime_ratios)
    error_direction = (
        "典型的には同点数なら Smolyak の方が高精度です。"
        if median_error_ratio > 1.0
        else "典型的には同点数なら Monte Carlo の方が高精度です。"
    )
    runtime_direction = (
        "典型的には warm runtime では Smolyak の方が遅いです。"
        if median_runtime_ratio < 1.0
        else "典型的には warm runtime では Monte Carlo の方が遅いです。"
    )

    observations = [
        f"成功ケースは {len(successful_reports)}/{len(reports)} 件で、失敗ケースは {len(failed_reports)} 件でした。",
        f"同点数比較では Smolyak が {overall_more_accurate}/{len(successful_reports)} ケースで Monte Carlo より高精度でした。",
        f"同点数比較では Smolyak が {overall_faster}/{len(successful_reports)} ケースで Monte Carlo より高速でした。",
        f"Monte Carlo / Smolyak の誤差比の中央値は {median_error_ratio:.2f} 倍で、{error_direction}",
        f"Monte Carlo / Smolyak の runtime 比の中央値は {median_runtime_ratio:.2f} 倍で、{runtime_direction}",
        (
            "最大の精度優位は "
            f"{_case_id(best_error_advantage)} で、同点数 MC の絶対誤差は "
            f"Smolyak の {float(best_error_advantage['monte_carlo_same_budget']['absolute_error_mean']) / max(float(best_error_advantage['smolyak']['absolute_error']), 1e-30):.2f} 倍でした。"
        ),
        (
            "Monte Carlo の速度優位が最大なのは "
            f"{_case_id(strongest_mc_speed_advantage)} で、同点数 MC の warm runtime は "
            f"Smolyak の {float(strongest_mc_speed_advantage['monte_carlo_same_budget']['warm_runtime_ms']) / max(float(strongest_mc_speed_advantage['smolyak']['warm_runtime_ms']), 1e-30):.4f} 倍でした。"
        ),
        f"Monte Carlo が同点数で精度勝ちしたケースは {len(mc_accuracy_wins)} 件、速度勝ちしたケースは {len(mc_speed_wins)} 件でした。",
    ]
    if failed_reports:
        failed_labels = ", ".join(_case_id(report) for report in failed_reports[:3])
        observations.append(
            f"失敗ケースは {failed_labels} などで、今回の run では GPU 側の OOM や timeout を `no result` として扱っています。"
        )

    for dtype in dtypes:
        dtype_reports = [report for report in successful_reports if str(report["dtype"]) == dtype]
        if not dtype_reports:
            continue
        dtype_error_ratios = [
            float(report["monte_carlo_same_budget"]["absolute_error_mean"]) / max(float(report["smolyak"]["absolute_error"]), 1e-30)
            for report in dtype_reports
        ]
        dtype_runtime_ratios = [
            float(report["monte_carlo_same_budget"]["warm_runtime_ms"]) / max(float(report["smolyak"]["warm_runtime_ms"]), 1e-30)
            for report in dtype_reports
        ]
        observations.append(
            f"{dtype} では Monte Carlo / Smolyak の誤差比中央値が {_median(dtype_error_ratios):.2f} 倍、runtime 比中央値が {_median(dtype_runtime_ratios):.2f} 倍でした。"
        )

    for level in levels:
        level_reports = [report for report in successful_reports if int(report["level"]) == level]
        if not level_reports:
            continue
        level_error_ratios = [
            float(report["monte_carlo_same_budget"]["absolute_error_mean"]) / max(float(report["smolyak"]["absolute_error"]), 1e-30)
            for report in level_reports
        ]
        observations.append(
            f"レベル {level} の Monte Carlo / Smolyak 誤差比中央値は {_median(level_error_ratios):.2f} 倍でした。"
        )

    if "float32" in dtypes and "float64" in dtypes:
        smolyak_error_ratios_64_over_32: list[float] = []
        mc_error_ratios_64_over_32: list[float] = []
        smolyak_runtime_ratios_64_over_32: list[float] = []
        mc_runtime_ratios_64_over_32: list[float] = []
        improved_smolyak_cases = 0
        improved_mc_cases = 0
        comparable_pairs = 0
        for dimension in dimensions:
            for level in levels:
                report32 = reports_by_key[("float32", level, dimension)]
                report64 = reports_by_key[("float64", level, dimension)]
                if report32.get("failed") or report64.get("failed"):
                    continue
                comparable_pairs += 1
                smolyak_error32 = float(report32["smolyak"]["absolute_error"])
                smolyak_error64 = float(report64["smolyak"]["absolute_error"])
                mc_error32 = float(report32["monte_carlo_same_budget"]["absolute_error_mean"])
                mc_error64 = float(report64["monte_carlo_same_budget"]["absolute_error_mean"])
                smolyak_time32 = float(report32["smolyak"]["warm_runtime_ms"])
                smolyak_time64 = float(report64["smolyak"]["warm_runtime_ms"])
                mc_time32 = float(report32["monte_carlo_same_budget"]["warm_runtime_ms"])
                mc_time64 = float(report64["monte_carlo_same_budget"]["warm_runtime_ms"])
                smolyak_error_ratios_64_over_32.append(smolyak_error64 / max(smolyak_error32, 1e-30))
                mc_error_ratios_64_over_32.append(mc_error64 / max(mc_error32, 1e-30))
                smolyak_runtime_ratios_64_over_32.append(smolyak_time64 / max(smolyak_time32, 1e-30))
                mc_runtime_ratios_64_over_32.append(mc_time64 / max(mc_time32, 1e-30))
                if smolyak_error64 < smolyak_error32:
                    improved_smolyak_cases += 1
                if mc_error64 < mc_error32:
                    improved_mc_cases += 1
        if comparable_pairs > 0:
            observations.extend(
                [
                    f"float64 にすると Smolyak の絶対誤差は {improved_smolyak_cases}/{comparable_pairs} 比較可能ケースで改善し、誤差比 float64/float32 の中央値は {_median(smolyak_error_ratios_64_over_32):.3f} でした。",
                    f"float64 にすると Monte Carlo の絶対誤差は {improved_mc_cases}/{comparable_pairs} 比較可能ケースで改善し、誤差比 float64/float32 の中央値は {_median(mc_error_ratios_64_over_32):.3f} でした。",
                    f"float64 の Smolyak runtime 罰則は中央値で {_median(smolyak_runtime_ratios_64_over_32):.2f} 倍、Monte Carlo は {_median(mc_runtime_ratios_64_over_32):.2f} 倍でした。",
                ]
            )

    monotonic_smolyak_dimensions = 0
    monotonic_mc_dimensions = 0
    eligible_smolyak_dimensions = 0
    eligible_mc_dimensions = 0
    for dtype in dtypes:
        for dimension in dimensions:
            smolyak_errors = [
                float(reports_by_key[(dtype, current_level, dimension)]["smolyak"]["absolute_error"])
                for current_level in levels
                if not reports_by_key[(dtype, current_level, dimension)].get("failed")
            ]
            mc_errors = [
                float(reports_by_key[(dtype, current_level, dimension)]["monte_carlo_same_budget"]["absolute_error_mean"])
                for current_level in levels
                if not reports_by_key[(dtype, current_level, dimension)].get("failed")
            ]
            if len(smolyak_errors) < 2 or len(mc_errors) < 2:
                continue
            eligible_smolyak_dimensions += 1
            eligible_mc_dimensions += 1
            if all(later <= earlier for earlier, later in zip(smolyak_errors, smolyak_errors[1:])):
                monotonic_smolyak_dimensions += 1
            if all(later <= earlier for earlier, later in zip(mc_errors, mc_errors[1:])):
                monotonic_mc_dimensions += 1
    observations.extend(
        [
            f"レベル上昇で Smolyak 誤差が単調に下がったのは {monotonic_smolyak_dimensions}/{max(eligible_smolyak_dimensions, 1)} の比較可能な次元-dtype 組でした。",
            f"同じ点数規則で見ると Monte Carlo 誤差が単調に下がったのは {monotonic_mc_dimensions}/{max(eligible_mc_dimensions, 1)} の比較可能な次元-dtype 組でした。",
        ]
    )

    return observations


def _write_markdown_report(
    *,
    reports: list[dict[str, Any]],
    dimensions: list[int],
    levels: list[int],
    dtypes: list[str],
    output_path: Path,
    figure_paths: dict[str, Path],
) -> None:
    reports_sorted = sorted(
        reports,
        key=lambda item: (str(item["dtype"]), int(item["level"]), int(item["dimension"])),
    )
    observations = _generate_observations(
        reports=reports_sorted,
        dimensions=dimensions,
        levels=levels,
        dtypes=dtypes,
    )

    lines = [
        "# Smolyak Same-Budget Accuracy Report",
        "",
        "## Method",
        "",
        "- 各ケースで解析解が既知の integrand を使い、Smolyak と Monte Carlo の両方の誤差を直接計算しました。",
        "- Monte Carlo のサンプル数は常に `SmolyakIntegrator.num_evaluation_points` と一致させ、同じ評価予算で比較しました。",
        "- 次元、レベル、dtype を振り、warm runtime と絶対誤差・相対誤差を保存しました。",
        "- OOM や timeout は `failed` ケースとして保存し、今回の高レベル sweep では `no result` とみなしています。",
        "",
        "## Figures",
        "",
        f"![Points (log)]({figure_paths['points'].name})",
        "",
        "- Read: x-axis is dimension `d`; y-axis is Smolyak evaluation points on a log scale. Up means more quadrature work.",
        "",
        f"![Smolyak relative error (log)]({figure_paths['smolyak_error'].name})",
        "",
        "- Read: x-axis is dimension `d`; y-axis is Smolyak relative error on a log scale. Lower is better.",
        "",
        f"![Monte Carlo relative error (log)]({figure_paths['mc_error'].name})",
        "",
        "- Read: x-axis is dimension `d`; y-axis is Monte Carlo relative error at the same budget, on a log scale. Lower is better.",
        "",
        f"![Error ratio (log)]({figure_paths['error_ratio'].name})",
        "",
        "- Read: x-axis is dimension `d`; y-axis is `MC error / Smolyak error` on a log scale. Values above 1 mean Smolyak is more accurate.",
        "",
        f"![Runtime ratio (linear)]({figure_paths['runtime_ratio_linear'].name})",
        "",
        "- Read: x-axis is dimension `d`; y-axis is `MC warm runtime / Smolyak warm runtime` on a linear scale. Values above 1 mean Smolyak is faster.",
        "",
        f"![Runtime ratio (log)]({figure_paths['runtime_ratio_log'].name})",
        "",
        "- Read: x-axis is dimension `d`; y-axis is the same runtime ratio on a log scale, which makes large wins and losses easier to compare.",
        "",
        "## Observations",
        "",
    ]
    for item in observations:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Case Table",
            "",
            "| Case | Points | Smolyak rel err | MC rel err | MC/Smolyak err ratio | Smolyak warm ms | MC warm ms | MC/Smolyak runtime ratio / status |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for report in reports_sorted:
        if report.get("failed"):
            failure_message = str(report.get("failure_summary", report.get("failure_message", "failure"))).replace("|", "/")
            lines.append(
                "| "
                + " | ".join(
                    [
                        _case_id(report),
                        "failed",
                        "failed",
                        "failed",
                        "failed",
                        "failed",
                        "failed",
                        failure_message,
                    ]
                )
                + " |"
            )
            continue
        smolyak_rel_error = float(report["smolyak"]["relative_error"])
        mc_rel_error = float(report["monte_carlo_same_budget"]["relative_error_mean"])
        error_ratio = mc_rel_error / max(smolyak_rel_error, 1e-30)
        smolyak_time = float(report["smolyak"]["warm_runtime_ms"])
        mc_time = float(report["monte_carlo_same_budget"]["warm_runtime_ms"])
        runtime_ratio = mc_time / max(smolyak_time, 1e-30)
        lines.append(
            "| "
            + " | ".join(
                [
                    _case_id(report),
                    str(int(report["smolyak"]["num_evaluation_points"])),
                    f"{smolyak_rel_error:.6e}",
                    f"{mc_rel_error:.6e}",
                    f"{error_ratio:.3f}",
                    f"{smolyak_time:.6f}",
                    f"{mc_time:.6f}",
                    f"{runtime_ratio:.3f}",
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
    report_dir = Path(args.resume_report_dir).resolve() if args.resume_report_dir else output_root / f"report_{run_label}"
    case_output_dir = report_dir / "cases"
    report_dir.mkdir(parents=True, exist_ok=True)
    case_output_dir.mkdir(parents=True, exist_ok=True)

    dimensions = _parse_csv_ints(args.dimensions)
    levels = _parse_csv_ints(args.levels)
    dtypes = _parse_csv_strings(args.dtypes)

    existing_reports = _load_existing_case_reports(case_output_dir)
    reports: list[dict[str, Any]] = []
    case_specs = [
        (dtype, level, dimension)
        for dtype in dtypes
        for level in levels
        for dimension in dimensions
    ]
    interrupted = False

    try:
        for index, (dtype, level, dimension) in enumerate(case_specs, start=1):
            key = _case_key(dtype, level, dimension)
            label = f"{dtype}-d{dimension}-l{level}"
            if key in existing_reports:
                report = existing_reports[key]
                _write_case_report(case_output_dir, report)
                reports.append(report)
                print(f"[{index}/{len(case_specs)}] cached {label}", file=sys.stderr, flush=True)
                continue

            print(f"[{index}/{len(case_specs)}] running {label}", file=sys.stderr, flush=True)
            report = _run_case(
                platform=args.platform,
                dimension=dimension,
                level=level,
                dtype=dtype,
                family=args.family,
                coeff_start=args.coeff_start,
                coeff_stop=args.coeff_stop,
                gaussian_alpha=args.gaussian_alpha,
                chunk_size=args.chunk_size,
                warm_repeats=args.warm_repeats,
                mc_seeds=args.mc_seeds,
                seed=args.seed,
                case_timeout_seconds=args.case_timeout_seconds,
                output_dir=case_output_dir,
            )
            _write_case_report(case_output_dir, report)
            reports.append(report)
            if report.get("failed"):
                print(
                    f"[{index}/{len(case_specs)}] finished {label} -> failed: {report.get('failure_summary', 'failure')}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"[{index}/{len(case_specs)}] finished {label} -> ok: {int(report['smolyak']['num_evaluation_points'])} points",
                    file=sys.stderr,
                    flush=True,
                )
    except KeyboardInterrupt:
        interrupted = True
        print("Interrupted while sweeping; filling remaining cases as `no result` and writing a partial aggregate report.", file=sys.stderr, flush=True)

    reports_by_key_partial = _index_reports(reports)
    for dtype, level, dimension in case_specs:
        key = _case_key(dtype, level, dimension)
        if key in reports_by_key_partial:
            continue
        report = _failure_report(
            platform=args.platform,
            dimension=dimension,
            level=level,
            dtype=dtype,
            family=args.family,
            chunk_size=args.chunk_size,
            mc_seeds=args.mc_seeds,
            warm_repeats=args.warm_repeats,
            failure_message="interrupted before execution",
        )
        _write_case_report(case_output_dir, report)
        reports.append(report)

    reports_by_key = _index_reports(reports)

    points_plot = report_dir / "points.svg"
    smolyak_error_plot = report_dir / "smolyak_relative_error.svg"
    mc_error_plot = report_dir / "mc_relative_error.svg"
    error_ratio_plot = report_dir / "same_budget_error_ratio.svg"
    runtime_ratio_linear_plot = report_dir / "same_budget_runtime_ratio_linear.svg"
    runtime_ratio_log_plot = report_dir / "same_budget_runtime_ratio_log.svg"

    point_series = [
        [
            _maybe_float(reports_by_key[(dtypes[0], level, dimension)]["smolyak"]["num_evaluation_points"])
            for dimension in dimensions
        ]
        for level in levels
    ]
    point_labels = [f"level {level}" for level in levels]
    _write_line_svg(
        x_values=dimensions,
        series=point_series,
        series_labels=point_labels,
        title="Smolyak Evaluation Points by Dimension and Level",
        x_label="Dimension d",
        y_label="Points",
        output_path=points_plot,
        log_scale=True,
    )

    smolyak_error_series, smolyak_error_labels = _series_for_dtype_level(
        reports_by_key=reports_by_key,
        dimensions=dimensions,
        dtypes=dtypes,
        levels=levels,
        value_fn=lambda report: _maybe_float(report["smolyak"]["relative_error"]),
    )
    _write_line_svg(
        x_values=dimensions,
        series=smolyak_error_series,
        series_labels=smolyak_error_labels,
        title="Smolyak Relative Error by Dimension, Level, and Dtype",
        x_label="Dimension d",
        y_label="Relative error",
        output_path=smolyak_error_plot,
        log_scale=True,
    )

    mc_error_series, mc_error_labels = _series_for_dtype_level(
        reports_by_key=reports_by_key,
        dimensions=dimensions,
        dtypes=dtypes,
        levels=levels,
        value_fn=lambda report: _maybe_float(report["monte_carlo_same_budget"]["relative_error_mean"]),
    )
    _write_line_svg(
        x_values=dimensions,
        series=mc_error_series,
        series_labels=mc_error_labels,
        title="Same-Budget Monte Carlo Relative Error by Dimension, Level, and Dtype",
        x_label="Dimension d",
        y_label="Relative error",
        output_path=mc_error_plot,
        log_scale=True,
    )

    error_ratio_series, error_ratio_labels = _series_for_dtype_level(
        reports_by_key=reports_by_key,
        dimensions=dimensions,
        dtypes=dtypes,
        levels=levels,
        value_fn=lambda report: (
            None
            if report.get("failed")
            else float(report["monte_carlo_same_budget"]["absolute_error_mean"]) / max(float(report["smolyak"]["absolute_error"]), 1e-30)
        ),
    )
    _write_line_svg(
        x_values=dimensions,
        series=error_ratio_series,
        series_labels=error_ratio_labels,
        title="Same-Budget Error Ratio (Monte Carlo / Smolyak)",
        x_label="Dimension d",
        y_label="Error ratio",
        output_path=error_ratio_plot,
        log_scale=True,
    )

    runtime_ratio_series, runtime_ratio_labels = _series_for_dtype_level(
        reports_by_key=reports_by_key,
        dimensions=dimensions,
        dtypes=dtypes,
        levels=levels,
        value_fn=lambda report: (
            None
            if report.get("failed")
            else float(report["monte_carlo_same_budget"]["warm_runtime_ms"]) / max(float(report["smolyak"]["warm_runtime_ms"]), 1e-30)
        ),
    )
    _write_line_svg(
        x_values=dimensions,
        series=runtime_ratio_series,
        series_labels=runtime_ratio_labels,
        title="Same-Budget Runtime Ratio (Monte Carlo / Smolyak)",
        x_label="Dimension d",
        y_label="Runtime ratio",
        output_path=runtime_ratio_linear_plot,
        log_scale=False,
    )
    _write_line_svg(
        x_values=dimensions,
        series=runtime_ratio_series,
        series_labels=runtime_ratio_labels,
        title="Same-Budget Runtime Ratio (Monte Carlo / Smolyak)",
        x_label="Dimension d",
        y_label="Runtime ratio",
        output_path=runtime_ratio_log_plot,
        log_scale=True,
    )

    summary = {
        "experiment": "smolyak_same_budget_accuracy_report",
        "started_at_utc": datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "workspace_root": str(Path.cwd().resolve()),
        "platform": args.platform,
        "dimensions": dimensions,
        "levels": levels,
        "dtypes": dtypes,
        "family": args.family,
        "gaussian_alpha": args.gaussian_alpha,
        "chunk_size": args.chunk_size,
        "mc_seeds": args.mc_seeds,
        "warm_repeats": args.warm_repeats,
        "case_timeout_seconds": args.case_timeout_seconds,
        "interrupted": interrupted,
        "cases": reports,
        "points_plot": str(points_plot),
        "smolyak_relative_error_plot": str(smolyak_error_plot),
        "mc_relative_error_plot": str(mc_error_plot),
        "same_budget_error_ratio_plot": str(error_ratio_plot),
        "same_budget_runtime_ratio_linear_plot": str(runtime_ratio_linear_plot),
        "same_budget_runtime_ratio_log_plot": str(runtime_ratio_log_plot),
    }
    summary_json = report_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")

    markdown_path = report_dir / "report.md"
    _write_markdown_report(
        reports=reports,
        dimensions=dimensions,
        levels=levels,
        dtypes=dtypes,
        output_path=markdown_path,
        figure_paths={
            "points": points_plot,
            "smolyak_error": smolyak_error_plot,
            "mc_error": mc_error_plot,
            "error_ratio": error_ratio_plot,
            "runtime_ratio_linear": runtime_ratio_linear_plot,
            "runtime_ratio_log": runtime_ratio_log_plot,
        },
    )

    print(
        json.dumps(
            {
                "report_dir": str(report_dir),
                "summary_json": str(summary_json),
                "report_md": str(markdown_path),
                "points_plot": str(points_plot),
                "smolyak_relative_error_plot": str(smolyak_error_plot),
                "mc_relative_error_plot": str(mc_error_plot),
                "same_budget_error_ratio_plot": str(error_ratio_plot),
                "same_budget_runtime_ratio_linear_plot": str(runtime_ratio_linear_plot),
                "same_budget_runtime_ratio_log_plot": str(runtime_ratio_log_plot),
                "num_cases": len(reports),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
