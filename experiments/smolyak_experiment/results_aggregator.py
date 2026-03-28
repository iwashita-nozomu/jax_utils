"""Helpers for aggregating Smolyak experiment results."""

from __future__ import annotations

from collections import defaultdict
import statistics
from typing import Any

__all__ = [
    "aggregate_by_dimension",
    "aggregate_by_dtype",
    "aggregate_by_level",
    "compute_statistics",
    "filter_failures",
]


def _case_params(result: dict[str, Any], /) -> dict[str, Any]:
    case_params = result.get("case_params", {})
    return case_params if isinstance(case_params, dict) else {}


def _smolyak_metrics(result: dict[str, Any], /) -> dict[str, Any]:
    smolyak = result.get("smolyak", {})
    return smolyak if isinstance(smolyak, dict) else {}


def aggregate_by_dtype(results: list[dict[str, Any]], /) -> dict[str, list[dict[str, Any]]]:
    aggregated: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        aggregated[str(_case_params(result).get("dtype", "unknown"))].append(result)
    return dict(aggregated)


def aggregate_by_dimension(
    results: list[dict[str, Any]],
    /,
) -> dict[int, list[dict[str, Any]]]:
    aggregated: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        aggregated[int(_case_params(result).get("dimension", -1))].append(result)
    return dict(sorted(aggregated.items()))


def aggregate_by_level(results: list[dict[str, Any]], /) -> dict[int, list[dict[str, Any]]]:
    aggregated: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        aggregated[int(_case_params(result).get("level", -1))].append(result)
    return dict(sorted(aggregated.items()))


def filter_failures(results: list[dict[str, Any]], /) -> dict[str, list[dict[str, Any]]]:
    failures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        smolyak = _smolyak_metrics(result)
        status = str(smolyak.get("status", "OTHER"))
        if status == "SUCCESS":
            failures["SUCCESS"].append(result)
            continue

        error_text = str(smolyak.get("error", "")).lower()
        if "timeout" in error_text:
            failures["TIMEOUT"].append(result)
        elif "out of memory" in error_text or "resource_exhausted" in error_text or "oom" in error_text:
            failures["OOM"].append(result)
        elif "nan" in error_text or "inf" in error_text or "diverg" in error_text:
            failures["NUMERICAL"].append(result)
        else:
            failures[status].append(result)
    return dict(failures)


def compute_statistics(
    results: list[dict[str, Any]],
    field_name: str,
    /,
    *,
    namespace: str = "smolyak",
) -> dict[str, Any]:
    values: list[float] = []
    for result in results:
        payload = result.get(namespace, result)
        if not isinstance(payload, dict):
            continue
        value = payload.get(field_name)
        if isinstance(value, (int, float)):
            values.append(float(value))

    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}

    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }
