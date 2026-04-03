from __future__ import annotations

from pathlib import Path
import sys


WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))

from experiments.smolyak_experiment.report_smolyak_vs_mc import _summarize_reports


def _make_report(
    *,
    family: str,
    dimension: int,
    level: int,
    smolyak_runtime_ms: float,
    smolyak_error: float,
    num_points: int,
    mc_runtime_ms: float | None,
    mc_error: float | None,
    mc_samples: int | None,
    mc_same_budget_runtime_ms: float,
    mc_same_budget_error: float,
) -> dict[str, object]:
    return {
        "family": {"family": family},
        "dimension": dimension,
        "level": level,
        "smolyak": {
            "warm_runtime_ms": smolyak_runtime_ms,
            "absolute_error": smolyak_error,
            "num_evaluation_points": num_points,
        },
        "monte_carlo": {
            "warm_runtime_ms": mc_runtime_ms,
            "absolute_error": mc_error,
            "chosen_num_samples": mc_samples,
        },
        "monte_carlo_same_budget": {
            "warm_runtime_ms": mc_same_budget_runtime_ms,
            "absolute_error_mean": mc_same_budget_error,
        },
        "summary": {
            "smolyak_faster_on_warm_runtime": (
                mc_runtime_ms is not None and smolyak_runtime_ms < mc_runtime_ms
            ),
            "smolyak_more_accurate_same_budget": smolyak_error < mc_same_budget_error,
            "smolyak_faster_same_budget": smolyak_runtime_ms < mc_same_budget_runtime_ms,
        },
    }


def test_summarize_reports_tracks_both_comparison_axes() -> None:
    reports = [
        _make_report(
            family="gaussian",
            dimension=2,
            level=3,
            smolyak_runtime_ms=2.0,
            smolyak_error=0.10,
            num_points=8,
            mc_runtime_ms=4.0,
            mc_error=0.09,
            mc_samples=64,
            mc_same_budget_runtime_ms=1.0,
            mc_same_budget_error=0.25,
        ),
        _make_report(
            family="gaussian",
            dimension=4,
            level=3,
            smolyak_runtime_ms=5.0,
            smolyak_error=0.20,
            num_points=32,
            mc_runtime_ms=5.0,
            mc_error=0.19,
            mc_samples=256,
            mc_same_budget_runtime_ms=10.0,
            mc_same_budget_error=0.10,
        ),
    ]

    summary = _summarize_reports(reports)

    assert summary["cases_run"] == 2

    matched = summary["matched_accuracy"]
    assert matched["smolyak_faster_cases"] == 1
    assert matched["monte_carlo_faster_or_unmatched_cases"] == 1
    assert matched["comparable_cases"] == 2
    assert matched["median_runtime_ratio_mc_over_smolyak"] == 1.5
    assert matched["best_speedup_case"] == {
        "case": "gaussian-d2-l3",
        "runtime_ratio_mc_over_smolyak": 2.0,
    }

    same_budget = summary["same_budget"]
    assert same_budget["smolyak_more_accurate_cases"] == 1
    assert same_budget["monte_carlo_more_accurate_or_tied_cases"] == 1
    assert same_budget["smolyak_faster_cases"] == 1
    assert same_budget["monte_carlo_faster_or_tied_cases"] == 1
    assert same_budget["comparable_error_cases"] == 2
    assert same_budget["comparable_runtime_cases"] == 2
    assert same_budget["median_error_ratio_mc_over_smolyak"] == 1.5
    assert same_budget["median_runtime_ratio_mc_over_smolyak"] == 1.25
    assert same_budget["best_accuracy_case"] == {
        "case": "gaussian-d2-l3",
        "error_ratio_mc_over_smolyak": 2.5,
    }
    assert same_budget["best_runtime_case"] == {
        "case": "gaussian-d4-l3",
        "runtime_ratio_mc_over_smolyak": 2.0,
    }


def test_summarize_reports_handles_unmatched_cases() -> None:
    reports = [
        _make_report(
            family="quadratic",
            dimension=3,
            level=2,
            smolyak_runtime_ms=1.0,
            smolyak_error=0.05,
            num_points=10,
            mc_runtime_ms=None,
            mc_error=None,
            mc_samples=None,
            mc_same_budget_runtime_ms=0.5,
            mc_same_budget_error=0.07,
        )
    ]

    summary = _summarize_reports(reports)

    matched = summary["matched_accuracy"]
    assert matched["comparable_cases"] == 0
    assert matched["median_runtime_ratio_mc_over_smolyak"] is None
    assert matched["best_speedup_case"] is None
