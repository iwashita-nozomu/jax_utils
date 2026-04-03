#!/usr/bin/env python3
"""Convert Smolyak mode-matrix JSONL into CSV tables and a Markdown report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from experiments.smolyak_experiment.report_smolyak_gpu_sweep import _write_line_svg


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate CSV tables and SVG/Markdown report from a mode-matrix JSONL file.",
    )
    parser.add_argument("--jsonl-path", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _preview_int_list(values: Any, /, *, limit: int = 12) -> str | None:
    if not isinstance(values, list):
        return None
    preview = ",".join(str(int(value)) for value in values[:limit])
    if len(values) > limit:
        preview += ",..."
    return preview


def _flatten_record(record: dict[str, Any]) -> dict[str, object]:
    case_params = record.get("case_params", {})
    smolyak = record.get("smolyak", {})
    single = smolyak.get("single") if isinstance(smolyak, dict) else None
    batch = smolyak.get("batch") if isinstance(smolyak, dict) else None
    batch_monitor = batch.get("monitor") if isinstance(batch, dict) else None
    return {
        "case_id": record.get("case_id"),
        "status": record.get("status"),
        "failure_kind": record.get("failure_kind"),
        "platform": case_params.get("platform"),
        "family": case_params.get("family"),
        "dimension": case_params.get("dimension"),
        "level": case_params.get("level"),
        "dtype": case_params.get("dtype"),
        "requested_mode": case_params.get("requested_mode"),
        "chunk_size": case_params.get("chunk_size"),
        "max_vectorized_suffix_ndim_requested": case_params.get("max_vectorized_suffix_ndim"),
        "batched_axis_order_strategy_requested": case_params.get("batched_axis_order_strategy"),
        "warm_repeats": case_params.get("warm_repeats"),
        "batch_size": case_params.get("batch_size"),
        "actual_mode": smolyak.get("actual_mode") if isinstance(smolyak, dict) else None,
        "active_axis_count": smolyak.get("active_axis_count") if isinstance(smolyak, dict) else None,
        "inactive_axis_count": smolyak.get("inactive_axis_count") if isinstance(smolyak, dict) else None,
        "axis_level_ceilings_label": (
            _preview_int_list(smolyak.get("axis_level_ceilings")) if isinstance(smolyak, dict) else None
        ),
        "requested_matches_actual": (
            str(case_params.get("requested_mode")) == str(smolyak.get("actual_mode"))
            if isinstance(smolyak, dict) and smolyak.get("actual_mode") is not None
            else None
        ),
        "uses_materialized_plan": (
            str(smolyak.get("actual_mode")) in {"points", "indexed"}
            if isinstance(smolyak, dict) and smolyak.get("actual_mode") is not None
            else None
        ),
        "num_terms": smolyak.get("num_terms") if isinstance(smolyak, dict) else None,
        "num_evaluation_points": smolyak.get("num_evaluation_points") if isinstance(smolyak, dict) else None,
        "storage_bytes": smolyak.get("storage_bytes") if isinstance(smolyak, dict) else None,
        "bytes_per_point": (
            float(smolyak.get("storage_bytes")) / float(smolyak.get("num_evaluation_points"))
            if isinstance(smolyak, dict)
            and smolyak.get("storage_bytes") is not None
            and smolyak.get("num_evaluation_points") not in (None, 0)
            else None
        ),
        "terms_per_point": (
            float(smolyak.get("num_terms")) / float(smolyak.get("num_evaluation_points"))
            if isinstance(smolyak, dict)
            and smolyak.get("num_terms") is not None
            and smolyak.get("num_evaluation_points") not in (None, 0)
            else None
        ),
        "vectorized_ndim": smolyak.get("vectorized_ndim") if isinstance(smolyak, dict) else None,
        "max_vectorized_points": smolyak.get("max_vectorized_points") if isinstance(smolyak, dict) else None,
        "max_vectorized_suffix_ndim": smolyak.get("max_vectorized_suffix_ndim") if isinstance(smolyak, dict) else None,
        "batched_axis_order_strategy": smolyak.get("batched_axis_order_strategy") if isinstance(smolyak, dict) else None,
        "value": smolyak.get("value") if isinstance(smolyak, dict) else None,
        "analytic_value": smolyak.get("analytic_value") if isinstance(smolyak, dict) else None,
        "absolute_error": smolyak.get("absolute_error") if isinstance(smolyak, dict) else None,
        "relative_error": smolyak.get("relative_error") if isinstance(smolyak, dict) else None,
        "init_ms": smolyak.get("init_ms") if isinstance(smolyak, dict) else None,
        "maxrss_mb": smolyak.get("maxrss_mb") if isinstance(smolyak, dict) else None,
        "single_first_call_ms": single.get("first_call_ms") if isinstance(single, dict) else None,
        "single_warm_runtime_ms": single.get("warm_runtime_ms") if isinstance(single, dict) else None,
        "single_compile_ms": single.get("compile_ms") if isinstance(single, dict) else None,
        "single_compile_share": (
            float(single.get("compile_ms")) / float(single.get("first_call_ms"))
            if isinstance(single, dict)
            and single.get("compile_ms") is not None
            and single.get("first_call_ms") not in (None, 0)
            else None
        ),
        "single_throughput_integrals_per_second": (
            single.get("throughput_integrals_per_second") if isinstance(single, dict) else None
        ),
        "batch_first_call_ms": batch.get("first_call_ms") if isinstance(batch, dict) else None,
        "batch_warm_runtime_ms": batch.get("warm_runtime_ms") if isinstance(batch, dict) else None,
        "batch_compile_ms": batch.get("compile_ms") if isinstance(batch, dict) else None,
        "batch_compile_share": (
            float(batch.get("compile_ms")) / float(batch.get("first_call_ms"))
            if isinstance(batch, dict)
            and batch.get("compile_ms") is not None
            and batch.get("first_call_ms") not in (None, 0)
            else None
        ),
        "batch_throughput_integrals_per_second": (
            batch.get("throughput_integrals_per_second") if isinstance(batch, dict) else None
        ),
        "batch_throughput_speedup_vs_single": (
            batch.get("throughput_speedup_vs_single") if isinstance(batch, dict) else None
        ),
        "batch_runtime_ms_per_integral": (
            float(batch.get("warm_runtime_ms")) / float(case_params.get("batch_size"))
            if isinstance(batch, dict)
            and batch.get("warm_runtime_ms") is not None
            and case_params.get("batch_size") not in (None, 0)
            else None
        ),
        "batch_avg_gpu_util": batch_monitor.get("avg_gpu_util") if isinstance(batch_monitor, dict) else None,
        "batch_peak_gpu_util": batch_monitor.get("peak_gpu_util") if isinstance(batch_monitor, dict) else None,
        "batch_peak_mem_used_mb": batch_monitor.get("peak_mem_used_mb") if isinstance(batch_monitor, dict) else None,
        "batch_dominant_pstate": batch_monitor.get("dominant_pstate") if isinstance(batch_monitor, dict) else None,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["family"],
        row["dtype"],
        row["requested_mode"],
        row["chunk_size"],
        row["level"],
        row["dimension"],
    )


def _median(values: Iterable[float]) -> float | None:
    values_list = [float(value) for value in values]
    return float(statistics.median(values_list)) if values_list else None


def _aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(row)

    output: list[dict[str, object]] = []
    for key, items in sorted(grouped.items()):
        success_items = [item for item in items if item["status"] == "success"]
        failure_counts = Counter(str(item["failure_kind"]) for item in items if item["status"] != "success")
        output.append(
            {
                "family": key[0],
                "dtype": key[1],
                "requested_mode": key[2],
                "chunk_size": key[3],
                "level": key[4],
                "dimension": key[5],
                "attempt_count": len(items),
                "success_count": len(success_items),
                "failure_count": len(items) - len(success_items),
                "dominant_failure_kind": failure_counts.most_common(1)[0][0] if failure_counts else None,
                "actual_modes": ",".join(sorted({str(item["actual_mode"]) for item in success_items})),
                "mode_mismatch_count": sum(
                    1 for item in success_items if item["requested_matches_actual"] is False
                ),
                "median_num_points": _median(
                    float(item["num_evaluation_points"]) for item in success_items if item["num_evaluation_points"] is not None
                ),
                "median_storage_bytes": _median(
                    float(item["storage_bytes"]) for item in success_items if item["storage_bytes"] is not None
                ),
                "median_bytes_per_point": _median(
                    float(item["bytes_per_point"]) for item in success_items if item["bytes_per_point"] is not None
                ),
                "median_terms_per_point": _median(
                    float(item["terms_per_point"]) for item in success_items if item["terms_per_point"] is not None
                ),
                "median_absolute_error": _median(
                    float(item["absolute_error"]) for item in success_items if item["absolute_error"] is not None
                ),
                "median_init_ms": _median(
                    float(item["init_ms"]) for item in success_items if item["init_ms"] is not None
                ),
                "median_single_warm_runtime_ms": _median(
                    float(item["single_warm_runtime_ms"]) for item in success_items if item["single_warm_runtime_ms"] is not None
                ),
                "median_batch_warm_runtime_ms": _median(
                    float(item["batch_warm_runtime_ms"]) for item in success_items if item["batch_warm_runtime_ms"] is not None
                ),
                "median_batch_runtime_ms_per_integral": _median(
                    float(item["batch_runtime_ms_per_integral"])
                    for item in success_items
                    if item["batch_runtime_ms_per_integral"] is not None
                ),
                "median_batch_throughput": _median(
                    float(item["batch_throughput_integrals_per_second"])
                    for item in success_items
                    if item["batch_throughput_integrals_per_second"] is not None
                ),
                "median_batch_speedup": _median(
                    float(item["batch_throughput_speedup_vs_single"])
                    for item in success_items
                    if item["batch_throughput_speedup_vs_single"] is not None
                ),
                "median_batch_avg_gpu_util": _median(
                    float(item["batch_avg_gpu_util"]) for item in success_items if item["batch_avg_gpu_util"] is not None
                ),
            }
        )
    return output


def _frontier_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["family"]),
                str(row["dtype"]),
                str(row["requested_mode"]),
                int(row["chunk_size"]),
            )
        ].append(row)

    output: list[dict[str, object]] = []
    for key, items in sorted(grouped.items()):
        levels = sorted({int(item["level"]) for item in items})
        for level in levels:
            level_items = [item for item in items if int(item["level"]) == level]
            success_items = [item for item in level_items if item["status"] == "success"]
            max_success_dimension = max((int(item["dimension"]) for item in success_items), default=None)
            last_success_item = (
                max(success_items, key=lambda item: int(item["dimension"])) if success_items else None
            )
            failure_dimensions = sorted(
                int(item["dimension"]) for item in level_items if item["status"] != "success"
            )
            first_failure_dimension = failure_dimensions[0] if failure_dimensions else None
            first_oom_dimension = min(
                (
                    int(item["dimension"])
                    for item in level_items
                    if item["status"] != "success" and str(item["failure_kind"]) == "oom"
                ),
                default=None,
            )
            first_timeout_dimension = min(
                (
                    int(item["dimension"])
                    for item in level_items
                    if item["status"] != "success" and str(item["failure_kind"]) == "timeout"
                ),
                default=None,
            )
            first_numerical_dimension = min(
                (
                    int(item["dimension"])
                    for item in level_items
                    if item["status"] != "success" and str(item["failure_kind"]) == "numerical"
                ),
                default=None,
            )
            output.append(
                {
                    "family": key[0],
                    "dtype": key[1],
                    "requested_mode": key[2],
                    "chunk_size": key[3],
                    "level": level,
                    "max_success_dimension": max_success_dimension,
                    "success_count": len(success_items),
                    "first_failure_dimension": first_failure_dimension,
                    "first_oom_dimension": first_oom_dimension,
                    "first_timeout_dimension": first_timeout_dimension,
                    "first_numerical_dimension": first_numerical_dimension,
                    "last_success_storage_bytes": None if last_success_item is None else last_success_item["storage_bytes"],
                    "last_success_maxrss_mb": None if last_success_item is None else last_success_item["maxrss_mb"],
                    "last_success_batch_peak_mem_used_mb": (
                        None if last_success_item is None else last_success_item["batch_peak_mem_used_mb"]
                    ),
                    "last_success_batch_warm_runtime_ms": (
                        None if last_success_item is None else last_success_item["batch_warm_runtime_ms"]
                    ),
                }
            )
    return output


def _failure_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    counter: Counter[tuple[str, str, str, int, str]] = Counter()
    for row in rows:
        if row["status"] == "success":
            continue
        counter[
            (
                str(row["family"]),
                str(row["dtype"]),
                str(row["requested_mode"]),
                int(row["chunk_size"]),
                str(row["failure_kind"]),
            )
        ] += 1
    return [
        {
            "family": key[0],
            "dtype": key[1],
            "requested_mode": key[2],
            "chunk_size": key[3],
            "failure_kind": key[4],
            "count": count,
        }
        for key, count in sorted(counter.items())
    ]


def _cross_mode_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["family"]),
                str(row["dtype"]),
                int(row["chunk_size"]),
                int(row["level"]),
                int(row["dimension"]),
            )
        ].append(row)

    output: list[dict[str, object]] = []
    for key, items in sorted(grouped.items()):
        success_by_mode = {
            str(item["requested_mode"]): item for item in items if item["status"] == "success"
        }
        surviving_modes = sorted(success_by_mode)
        runtime_candidates = [
            (mode, float(item["batch_warm_runtime_ms"]))
            for mode, item in success_by_mode.items()
            if item["batch_warm_runtime_ms"] is not None
        ]
        storage_candidates = [
            (mode, float(item["storage_bytes"]))
            for mode, item in success_by_mode.items()
            if item["storage_bytes"] is not None
        ]
        error_candidates = [
            (mode, float(item["absolute_error"]))
            for mode, item in success_by_mode.items()
            if item["absolute_error"] is not None
        ]
        fastest_success_mode = min(runtime_candidates, key=lambda item: item[1])[0] if runtime_candidates else None
        lowest_storage_success_mode = (
            min(storage_candidates, key=lambda item: item[1])[0] if storage_candidates else None
        )
        lowest_error_success_mode = min(error_candidates, key=lambda item: item[1])[0] if error_candidates else None
        auto_item = success_by_mode.get("auto")
        best_runtime = min((value for _, value in runtime_candidates), default=None)
        best_storage = min((value for _, value in storage_candidates), default=None)
        best_error = min((value for _, value in error_candidates), default=None)
        output.append(
            {
                "family": key[0],
                "dtype": key[1],
                "chunk_size": key[2],
                "level": key[3],
                "dimension": key[4],
                "surviving_modes": ",".join(surviving_modes),
                "fastest_success_mode": fastest_success_mode,
                "lowest_storage_success_mode": lowest_storage_success_mode,
                "lowest_error_success_mode": lowest_error_success_mode,
                "auto_success": "auto" in success_by_mode,
                "auto_runtime_regret_vs_best": (
                    float(auto_item["batch_warm_runtime_ms"]) / best_runtime
                    if auto_item is not None
                    and auto_item["batch_warm_runtime_ms"] is not None
                    and best_runtime not in (None, 0.0)
                    else None
                ),
                "auto_storage_regret_vs_best": (
                    float(auto_item["storage_bytes"]) / best_storage
                    if auto_item is not None
                    and auto_item["storage_bytes"] is not None
                    and best_storage not in (None, 0.0)
                    else None
                ),
                "auto_error_regret_vs_best": (
                    float(auto_item["absolute_error"]) / best_error
                    if auto_item is not None
                    and auto_item["absolute_error"] is not None
                    and best_error not in (None, 0.0)
                    else None
                ),
                "best_mode_if_auto_failed": (
                    fastest_success_mode if auto_item is None and fastest_success_mode is not None else None
                ),
            }
        )
    return output


def _status_color(status: str, actual_mode: str | None, failure_kind: str | None) -> str:
    if status == "success":
        if actual_mode == "points":
            return "#1b5e20"
        if actual_mode == "indexed":
            return "#1565c0"
        if actual_mode == "lazy-indexed":
            return "#00838f"
        if actual_mode in {"none", "batched"}:
            return "#6a1b9a"
        return "#2e7d32"
    if failure_kind == "oom":
        return "#c62828"
    if failure_kind == "timeout":
        return "#ef6c00"
    if failure_kind == "numerical":
        return "#ad1457"
    return "#757575"


def _write_status_heatmap(
    *,
    rows: list[dict[str, object]],
    family: str,
    dtype: str,
    requested_mode: str,
    chunk_size: int,
    output_path: Path,
) -> None:
    subset = [
        row
        for row in rows
        if str(row["family"]) == family
        and str(row["dtype"]) == dtype
        and str(row["requested_mode"]) == requested_mode
        and int(row["chunk_size"]) == chunk_size
    ]
    if not subset:
        return
    dimensions = sorted({int(row["dimension"]) for row in subset})
    levels = sorted({int(row["level"]) for row in subset})
    width = 1000
    height = 60 + 36 * len(levels) + 100
    margin_left = 100
    margin_top = 60
    cell_width = max(14, int((width - margin_left - 50) / max(len(dimensions), 1)))
    cell_height = 28
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18" font-family="monospace">{family} {dtype} {requested_mode} chunk={chunk_size}</text>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="12" font-family="monospace">Dimension d</text>',
    ]
    for col, dimension in enumerate(dimensions):
        x = margin_left + col * cell_width
        lines.append(
            f'<text x="{x + cell_width / 2}" y="{margin_top - 10}" text-anchor="middle" font-size="10" font-family="monospace">{dimension}</text>'
        )
    for row_index, level in enumerate(levels):
        y = margin_top + row_index * cell_height
        lines.append(
            f'<text x="{margin_left - 12}" y="{y + 18}" text-anchor="end" font-size="11" font-family="monospace">l={level}</text>'
        )
        for col, dimension in enumerate(dimensions):
            x = margin_left + col * cell_width
            cell = next(
                (
                    item
                    for item in subset
                    if int(item["dimension"]) == dimension and int(item["level"]) == level
                ),
                None,
            )
            color = "#eeeeee"
            if cell is not None:
                color = _status_color(
                    str(cell["status"]),
                    None if cell["actual_mode"] is None else str(cell["actual_mode"]),
                    None if cell["failure_kind"] is None else str(cell["failure_kind"]),
                )
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_width - 1}" height="{cell_height - 1}" fill="{color}" stroke="#ffffff" stroke-width="1"/>'
            )
    legend_y = margin_top + len(levels) * cell_height + 24
    legend = [
        ("points success", "#1b5e20"),
        ("indexed success", "#1565c0"),
        ("lazy-indexed success", "#00838f"),
        ("batched success", "#6a1b9a"),
        ("oom", "#c62828"),
        ("timeout", "#ef6c00"),
        ("error", "#757575"),
    ]
    for index, (label, color) in enumerate(legend):
        x = margin_left + index * 135
        lines.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{x + 20}" y="{legend_y + 12}" font-size="11" font-family="monospace">{label}</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mode_color(mode: str | None) -> str:
    if mode == "points":
        return "#1b5e20"
    if mode == "indexed":
        return "#1565c0"
    if mode == "lazy-indexed":
        return "#00838f"
    if mode == "batched":
        return "#6a1b9a"
    if mode == "auto":
        return "#2e7d32"
    return "#bdbdbd"


def _write_fastest_mode_heatmap(
    *,
    rows: list[dict[str, object]],
    family: str,
    dtype: str,
    chunk_size: int,
    output_path: Path,
) -> None:
    subset = [
        row
        for row in rows
        if str(row["family"]) == family
        and str(row["dtype"]) == dtype
        and int(row["chunk_size"]) == chunk_size
    ]
    if not subset:
        return
    dimensions = sorted({int(row["dimension"]) for row in subset})
    levels = sorted({int(row["level"]) for row in subset})
    width = 1000
    height = 60 + 36 * len(levels) + 100
    margin_left = 100
    margin_top = 60
    cell_width = max(14, int((width - margin_left - 50) / max(len(dimensions), 1)))
    cell_height = 28
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18" font-family="monospace">{family} {dtype} chunk={chunk_size} fastest success mode</text>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="12" font-family="monospace">Dimension d</text>',
    ]
    for col, dimension in enumerate(dimensions):
        x = margin_left + col * cell_width
        lines.append(
            f'<text x="{x + cell_width / 2}" y="{margin_top - 10}" text-anchor="middle" font-size="10" font-family="monospace">{dimension}</text>'
        )
    for row_index, level in enumerate(levels):
        y = margin_top + row_index * cell_height
        lines.append(
            f'<text x="{margin_left - 12}" y="{y + 18}" text-anchor="end" font-size="11" font-family="monospace">l={level}</text>'
        )
        for col, dimension in enumerate(dimensions):
            x = margin_left + col * cell_width
            cell = next(
                (
                    item
                    for item in subset
                    if int(item["dimension"]) == dimension and int(item["level"]) == level
                ),
                None,
            )
            color = _mode_color(None if cell is None else str(cell["fastest_success_mode"]))
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_width - 1}" height="{cell_height - 1}" fill="{color}" stroke="#ffffff" stroke-width="1"/>'
            )
        legend_y = margin_top + len(levels) * cell_height + 24
    legend = [
        ("points", "#1b5e20"),
        ("indexed", "#1565c0"),
        ("lazy-indexed", "#00838f"),
        ("batched", "#6a1b9a"),
        ("no success", "#bdbdbd"),
    ]
    for index, (label, color) in enumerate(legend):
        x = margin_left + index * 155
        lines.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{color}"/>')
        lines.append(f'<text x="{x + 20}" y="{legend_y + 12}" font-size="11" font-family="monospace">{label}</text>')
    lines.append("</svg>")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _auto_frontier_gap_rows(frontier_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in frontier_rows:
        grouped[
            (
                str(row["family"]),
                str(row["dtype"]),
                int(row["chunk_size"]),
                int(row["level"]),
            )
        ].append(row)
    output: list[dict[str, object]] = []
    for key, items in sorted(grouped.items()):
        max_any = max(
            (int(row["max_success_dimension"]) for row in items if row["max_success_dimension"] is not None),
            default=None,
        )
        auto_row = next((row for row in items if str(row["requested_mode"]) == "auto"), None)
        auto_max = None if auto_row is None else auto_row["max_success_dimension"]
        output.append(
            {
                "family": key[0],
                "dtype": key[1],
                "chunk_size": key[2],
                "level": key[3],
                "best_mode_max_success_dimension": max_any,
                "auto_max_success_dimension": auto_max,
                "auto_frontier_gap": (
                    None
                    if max_any is None or auto_max is None
                    else int(max_any) - int(auto_max)
                ),
            }
        )
    return output


def _write_markdown_report(
    *,
    records: list[dict[str, Any]],
    raw_csv_path: Path,
    aggregate_csv_path: Path,
    frontier_csv_path: Path,
    failure_csv_path: Path,
    figure_paths: dict[str, Path],
    output_path: Path,
) -> None:
    success_records = [item for item in records if item.get("status") == "success"]
    failure_records = [item for item in records if item.get("status") != "success"]
    actual_mode_counts = Counter(
        str(item.get("smolyak", {}).get("actual_mode"))
        for item in success_records
        if isinstance(item.get("smolyak"), dict)
    )
    failure_counts = Counter(str(item.get("failure_kind", "unknown")) for item in failure_records)
    frontier_snapshot_rows = _frontier_snapshot_rows(records)
    lines = [
        "# Smolyak Mode Matrix Report",
        "",
        "## Method",
        "",
        "- Every requested case is launched in its own child process; no pre-filter skips cases based on predicted feasibility.",
        "- Requested modes are `auto`, `points`, `indexed`, `lazy-indexed`, and `batched`, and the realized mode is recorded separately.",
        "- Each successful case benchmarks a single compiled integral and a batched `vmap` integral, and GPU monitor data is collected when running on GPU.",
        "- Raw JSONL is converted into CSV tables so later loops can reuse the data without re-running the full matrix.",
        "",
        "## Summary",
        "",
        f"- Cases recorded: {len(records)}",
        f"- Successful cases: {len(success_records)}",
        f"- Failed cases: {len(failure_records)}",
        f"- Realized mode counts: {dict(sorted(actual_mode_counts.items()))}",
        f"- Failure counts: {dict(sorted(failure_counts.items()))}",
        "",
        "## Frontier Snapshot",
        "",
        "| family | dtype | requested mode | chunk | max dimension (any level) | max dimension at highest recorded level | highest level with any success |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in frontier_snapshot_rows:
        lines.append(
            f"| {row['family']} | {row['dtype']} | {row['requested_mode']} | {row['chunk_size']} | {row['max_dimension_any_level']} | {row['max_dimension_at_highest_level']} | {row['highest_level_with_success']} |"
        )
    lines.extend(
        [
            "",
        "## Tables",
        "",
        f"- Raw per-case table: `{raw_csv_path.name}`",
        f"- Aggregated per-case table: `{aggregate_csv_path.name}`",
        f"- Frontier table: `{frontier_csv_path.name}`",
        f"- Failure table: `{failure_csv_path.name}`",
        f"- Cross-mode table: `mode_matrix_cross_mode.csv`",
        "",
        ]
    )
    for label, path in figure_paths.items():
        lines.extend(_figure_section(label=label, path=path))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _frontier_snapshot_rows(records: list[dict[str, Any]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        case_params = record.get("case_params", {})
        grouped[
            (
                str(case_params.get("family")),
                str(case_params.get("dtype")),
                str(case_params.get("requested_mode")),
                int(case_params.get("chunk_size", 0)),
            )
        ].append(record)

    output: list[dict[str, object]] = []
    for key, items in sorted(grouped.items()):
        success_items = [item for item in items if item.get("status") == "success"]
        if not success_items:
            output.append(
                {
                    "family": key[0],
                    "dtype": key[1],
                    "requested_mode": key[2],
                    "chunk_size": key[3],
                    "max_dimension_any_level": None,
                    "max_dimension_at_highest_level": None,
                    "highest_level_with_success": None,
                }
            )
            continue
        highest_level = max(int(item.get("case_params", {}).get("level", 0)) for item in success_items)
        highest_level_items = [
            item for item in success_items if int(item.get("case_params", {}).get("level", 0)) == highest_level
        ]
        output.append(
            {
                "family": key[0],
                "dtype": key[1],
                "requested_mode": key[2],
                "chunk_size": key[3],
                "max_dimension_any_level": max(
                    int(item.get("case_params", {}).get("dimension", 0)) for item in success_items
                ),
                "max_dimension_at_highest_level": max(
                    int(item.get("case_params", {}).get("dimension", 0)) for item in highest_level_items
                ),
                "highest_level_with_success": highest_level,
            }
        )
    return output


def _figure_section(*, label: str, path: Path) -> list[str]:
    note = "Read this figure together with the CSV tables."
    if label.startswith("Frontier "):
        note = (
            "Reading guide: x-axis is Smolyak level, y-axis is the largest dimension that finished successfully. "
            "Each line is a requested mode. Higher is better."
        )
    elif label.startswith("Status "):
        note = (
            "Reading guide: x-axis is dimension and y-axis is level. "
            "Green/blue/teal/purple cells are successful runs colored by realized execution mode; warm colors indicate failures."
        )
    elif label.startswith("Points "):
        note = (
            "Reading guide: x-axis is dimension and y-axis is median evaluation-point count on a log scale. "
            "Steeper growth means the combinatorial burden is becoming the dominant issue."
        )
    elif label.startswith("Storage "):
        note = (
            "Reading guide: x-axis is dimension and y-axis is median storage bytes on a log scale. "
            "This helps separate memory-layout limits from pure arithmetic limits."
        )
    elif label.startswith("Batch Runtime "):
        note = (
            "Reading guide: x-axis is dimension and y-axis is median batched warm runtime in milliseconds on a log scale. "
            "This is the steady-state execution cost after compilation."
        )
    elif label.startswith("GPU Util "):
        note = (
            "Reading guide: x-axis is dimension and y-axis is median average GPU utilization in percent. "
            "Low values on successful cases usually indicate under-filled kernels or host-side overhead."
        )
    elif label.startswith("Fastest Success Mode "):
        note = (
            "Reading guide: x-axis is dimension and y-axis is level. "
            "Each cell shows which requested mode achieved the lowest batched warm runtime among the successful modes."
        )
    elif label.startswith("Auto Frontier Gap "):
        note = (
            "Reading guide: x-axis is level and y-axis is the difference between the best empirical frontier and the auto-mode frontier. "
            "A positive gap means the current auto policy is leaving reachable cases on the table."
        )
    return [
        f"## {label}",
        "",
        note,
        "",
        f"![{label}]({path.name})",
        "",
    ]


def _plot_metric_by_dimension(
    *,
    aggregate_rows: list[dict[str, object]],
    family: str,
    dtype: str,
    requested_mode: str,
    chunk_size: int,
    metric_key: str,
    title_prefix: str,
    y_label: str,
    output_path: Path,
    log_scale: bool,
) -> bool:
    subset = [
        row
        for row in aggregate_rows
        if str(row["family"]) == family
        and str(row["dtype"]) == dtype
        and str(row["requested_mode"]) == requested_mode
        and int(row["chunk_size"]) == chunk_size
    ]
    if not subset:
        return False
    dimensions = sorted({int(row["dimension"]) for row in subset})
    levels = sorted({int(row["level"]) for row in subset})
    series: list[list[float | None]] = []
    labels: list[str] = []
    for level in levels:
        values = [
            _maybe_float(
                next(
                    (
                        row[metric_key]
                        for row in subset
                        if int(row["level"]) == level and int(row["dimension"]) == dimension
                    ),
                    None,
                )
            )
            for dimension in dimensions
        ]
        if any(value is not None for value in values):
            series.append(values)
            labels.append(f"l={level}")
    if not series:
        return False
    _write_line_svg(
        x_values=dimensions,
        series=series,
        series_labels=labels,
        title=f"{title_prefix} ({family}, {dtype}, mode={requested_mode}, chunk={chunk_size})",
        x_label="Dimension d",
        y_label=y_label,
        output_path=output_path,
        log_scale=log_scale,
    )
    return True


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = _read_jsonl(jsonl_path)
    flat_rows = [_flatten_record(record) for record in records]
    aggregate_rows = _aggregate_rows(flat_rows)
    frontier_rows = _frontier_rows(flat_rows)
    failure_rows = _failure_rows(flat_rows)
    cross_mode_rows = _cross_mode_rows(flat_rows)
    auto_gap_rows = _auto_frontier_gap_rows(frontier_rows)

    raw_csv_path = output_dir / "mode_matrix_raw_cases.csv"
    aggregate_csv_path = output_dir / "mode_matrix_aggregate.csv"
    frontier_csv_path = output_dir / "mode_matrix_frontier.csv"
    failure_csv_path = output_dir / "mode_matrix_failures.csv"
    cross_mode_csv_path = output_dir / "mode_matrix_cross_mode.csv"
    _write_csv(raw_csv_path, flat_rows)
    _write_csv(aggregate_csv_path, aggregate_rows)
    _write_csv(frontier_csv_path, frontier_rows)
    _write_csv(failure_csv_path, failure_rows)
    _write_csv(cross_mode_csv_path, cross_mode_rows)

    figure_paths: dict[str, Path] = {}
    if frontier_rows:
        frontier_groups = sorted(
            {
                (
                    str(row["family"]),
                    str(row["dtype"]),
                    int(row["chunk_size"]),
                )
                for row in frontier_rows
            }
        )
        for family, dtype, chunk_size in frontier_groups:
            subset = [
                row
                for row in frontier_rows
                if str(row["family"]) == family and str(row["dtype"]) == dtype and int(row["chunk_size"]) == chunk_size
            ]
            levels = sorted({int(row["level"]) for row in subset})
            modes = sorted({str(row["requested_mode"]) for row in subset})
            frontier_plot = output_dir / f"frontier_{family}_{dtype}_c{chunk_size}.svg"
            _write_line_svg(
                x_values=levels,
                series=[
                    [
                        _maybe_float(
                            next(
                                (
                                    row["max_success_dimension"]
                                    for row in subset
                                    if str(row["requested_mode"]) == mode and int(row["level"]) == level
                                ),
                                None,
                            )
                        )
                        for level in levels
                    ]
                    for mode in modes
                ],
                series_labels=[f"mode={mode}" for mode in modes],
                title=f"Success Frontier by Level ({family}, {dtype}, chunk={chunk_size})",
                x_label="Level l",
                y_label="Max successful dimension",
                output_path=frontier_plot,
                log_scale=False,
            )
            figure_paths[f"Frontier {family} {dtype} chunk={chunk_size}"] = frontier_plot

            auto_gap_subset = [
                row
                for row in auto_gap_rows
                if str(row["family"]) == family and str(row["dtype"]) == dtype and int(row["chunk_size"]) == chunk_size
            ]
            if auto_gap_subset:
                auto_gap_plot = output_dir / f"auto_frontier_gap_{family}_{dtype}_c{chunk_size}.svg"
                auto_gap_levels = sorted(int(row["level"]) for row in auto_gap_subset)
                _write_line_svg(
                    x_values=auto_gap_levels,
                    series=[
                        [
                            _maybe_float(
                                next(
                                    (
                                        row["auto_frontier_gap"]
                                        for row in auto_gap_subset
                                        if int(row["level"]) == level
                                    ),
                                    None,
                                )
                            )
                            for level in auto_gap_levels
                        ]
                    ],
                    series_labels=["best-mode frontier - auto frontier"],
                    title=f"Auto Frontier Gap ({family}, {dtype}, chunk={chunk_size})",
                    x_label="Level l",
                    y_label="Dimension gap",
                    output_path=auto_gap_plot,
                    log_scale=False,
                )
                figure_paths[f"Auto Frontier Gap {family} {dtype} chunk={chunk_size}"] = auto_gap_plot

    metric_groups = sorted(
        {
            (
                str(row["family"]),
                str(row["dtype"]),
                str(row["requested_mode"]),
                int(row["chunk_size"]),
            )
            for row in aggregate_rows
        }
    )
    metric_specs = [
        ("median_num_points", "Points by Dimension", "Median evaluation points", True, "Points"),
        ("median_storage_bytes", "Storage by Dimension", "Median storage bytes", True, "Storage"),
        ("median_batch_warm_runtime_ms", "Batch Runtime by Dimension", "Median batched warm runtime (ms)", True, "Batch Runtime"),
        ("median_batch_avg_gpu_util", "GPU Util by Dimension", "Median average GPU utilization (%)", False, "GPU Util"),
    ]
    for family, dtype, requested_mode, chunk_size in metric_groups:
        for metric_key, title_prefix, y_label, log_scale, label_prefix in metric_specs:
            output_path = output_dir / f"{metric_key}_{family}_{dtype}_{requested_mode}_c{chunk_size}.svg"
            wrote = _plot_metric_by_dimension(
                aggregate_rows=aggregate_rows,
                family=family,
                dtype=dtype,
                requested_mode=requested_mode,
                chunk_size=chunk_size,
                metric_key=metric_key,
                title_prefix=title_prefix,
                y_label=y_label,
                output_path=output_path,
                log_scale=log_scale,
            )
            if wrote:
                figure_paths[f"{label_prefix} {family} {dtype} mode={requested_mode} chunk={chunk_size}"] = output_path

    fastest_mode_groups = sorted(
        {
            (
                str(row["family"]),
                str(row["dtype"]),
                int(row["chunk_size"]),
            )
            for row in cross_mode_rows
        }
    )
    for family, dtype, chunk_size in fastest_mode_groups:
        fastest_mode_path = output_dir / f"fastest_success_mode_{family}_{dtype}_c{chunk_size}.svg"
        _write_fastest_mode_heatmap(
            rows=cross_mode_rows,
            family=family,
            dtype=dtype,
            chunk_size=chunk_size,
            output_path=fastest_mode_path,
        )
        figure_paths[f"Fastest Success Mode {family} {dtype} chunk={chunk_size}"] = fastest_mode_path

    heatmap_groups = sorted(
        {
            (
                str(row["family"]),
                str(row["dtype"]),
                str(row["requested_mode"]),
                int(row["chunk_size"]),
            )
            for row in flat_rows
        }
    )
    for family, dtype, requested_mode, chunk_size in heatmap_groups:
        heatmap_path = output_dir / f"status_{family}_{dtype}_{requested_mode}_c{chunk_size}.svg"
        _write_status_heatmap(
            rows=flat_rows,
            family=family,
            dtype=dtype,
            requested_mode=requested_mode,
            chunk_size=chunk_size,
            output_path=heatmap_path,
        )
        figure_paths[f"Status {family} {dtype} mode={requested_mode} chunk={chunk_size}"] = heatmap_path

    report_md = output_dir / "report.md"
    _write_markdown_report(
        records=records,
        raw_csv_path=raw_csv_path,
        aggregate_csv_path=aggregate_csv_path,
        frontier_csv_path=frontier_csv_path,
        failure_csv_path=failure_csv_path,
        figure_paths=figure_paths,
        output_path=report_md,
    )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_csv": str(raw_csv_path),
        "aggregate_csv": str(aggregate_csv_path),
        "frontier_csv": str(frontier_csv_path),
        "failure_csv": str(failure_csv_path),
        "cross_mode_csv": str(cross_mode_csv_path),
        "report_md": str(report_md),
        "figures": {label: str(path) for label, path in figure_paths.items()},
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
