#!/usr/bin/env python3
"""Compare same-budget experiment results against literature and analytic theory."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from .report_smolyak_same_budget_accuracy import _maybe_float, _write_line_svg


SCRIPT_DIR = Path(__file__).resolve().parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a literature-and-theory note for Smolyak same-budget experiments.",
    )
    parser.add_argument(
        "--summary-json",
        required=True,
        help="Path to summary.json produced by report_smolyak_same_budget_accuracy.py",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for theory outputs. Defaults to the summary directory.",
    )
    return parser


def _gaussian_integral(alpha: float, dimension: int) -> float:
    factor = math.sqrt(math.pi / alpha) * math.erf(0.5 * math.sqrt(alpha))
    return factor**dimension


def _gaussian_second_moment(alpha: float, dimension: int) -> float:
    factor = math.sqrt(math.pi / (2.0 * alpha)) * math.erf(0.5 * math.sqrt(2.0 * alpha))
    return factor**dimension


def _case_key(case: dict[str, Any]) -> tuple[str, int, int]:
    return str(case["dtype"]), int(case["level"]), int(case["dimension"])


def _case_label(case: dict[str, Any]) -> str:
    return f"{case['dtype']}-d{case['dimension']}-l{case['level']}"


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else float("nan")


def _build_theory_case(case: dict[str, Any], alpha: float) -> dict[str, Any]:
    dimension = int(case["dimension"])
    num_points = int(case["smolyak"]["num_evaluation_points"])
    integral = _gaussian_integral(alpha, dimension)
    second_moment = _gaussian_second_moment(alpha, dimension)
    variance = max(0.0, second_moment - integral * integral)
    mc_theory_rmse = math.sqrt(variance / num_points)
    mc_theory_mae = math.sqrt(2.0 / math.pi) * mc_theory_rmse
    observed_mc_mae = float(case["monte_carlo_same_budget"]["absolute_error_mean"])
    observed_smolyak_error = float(case["smolyak"]["absolute_error"])
    return {
        "label": _case_label(case),
        "dtype": case["dtype"],
        "dimension": dimension,
        "level": int(case["level"]),
        "num_points": num_points,
        "analytic_integral": integral,
        "gaussian_second_moment": second_moment,
        "mc_theory_variance": variance,
        "mc_theory_rmse": mc_theory_rmse,
        "mc_theory_mae_clt": mc_theory_mae,
        "mc_observed_mae": observed_mc_mae,
        "mc_observed_to_theory_mae_ratio": observed_mc_mae / max(mc_theory_mae, 1e-30),
        "smolyak_absolute_error": observed_smolyak_error,
        "smolyak_to_mc_theory_mae_ratio": observed_smolyak_error / max(mc_theory_mae, 1e-30),
    }


def _write_markdown(
    *,
    summary: dict[str, Any],
    theory_cases: list[dict[str, Any]],
    output_path: Path,
    figure_paths: dict[str, Path],
) -> None:
    successful = theory_cases
    by_dtype = {
        dtype: [case for case in successful if case["dtype"] == dtype]
        for dtype in summary["dtypes"]
    }
    by_level = {
        level: [case for case in successful if case["level"] == level]
        for level in summary["levels"]
    }

    mc_ratio_median = _median([case["mc_observed_to_theory_mae_ratio"] for case in successful])
    smolyak_ratio_median = _median([case["smolyak_to_mc_theory_mae_ratio"] for case in successful])
    best_smolyak = min(successful, key=lambda case: case["smolyak_to_mc_theory_mae_ratio"])
    worst_smolyak = max(successful, key=lambda case: case["smolyak_to_mc_theory_mae_ratio"])

    lines = [
        "# Smolyak Literature and Theory Note",
        "",
        "## Literature",
        "",
        "- Smolyak quadrature on cubes with mixed smoothness is classically justified by Smolyak-type sparse constructions and later analyses such as Novak and Ritter (1996), who describe a Clenshaw-Curtis/Smolyak method with near-optimal error bounds for bounded mixed derivatives and report strong performance for smooth integrands in higher dimension.",
        "- Gerstner and Griebel (1998) emphasize that sparse-grid quadrature combines tensor-product rules so that cost becomes only weakly dimension-dependent for functions with bounded mixed derivatives, and compare trapezoidal, Clenshaw-Curtis, Gauss, and Gauss-Patterson variants.",
        "- Bungartz and Griebel (2004) summarize the core sparse-grid complexity picture: sparse grids replace the full-grid scaling by polylogarithmic corrections in the number of 1D points while preserving higher-order convergence for mixed-regular functions.",
        "- Hinrichs, Novak, and Ullrich (2013) show that the Clenshaw-Curtis Smolyak algorithm is weakly tractable for analytic functions, which is directly relevant because the Gaussian test family used here is analytic.",
        "- For plain Monte Carlo, the absolute-error exponent remains the classical `N^{-1/2}` in the sample count, but Tang (2022) stresses that the dimension enters through constants and relative-error requirements, so behavior across dimensions is not uniform even when the exponent is dimension-free.",
        "",
        "## Theory Used Here",
        "",
        "- The current implementation uses nested Clenshaw-Curtis difference rules in each axis and combines them with Smolyak's construction; see [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py).",
        "- For the Gaussian integrand `f(x)=exp(-alpha ||x||^2)` on `[-1/2,1/2]^d`, the exact integral is `I_d(alpha) = [sqrt(pi/alpha) * erf(sqrt(alpha)/2)]^d`.",
        "- The exact second moment under the uniform cube measure is `M_{2,d}(alpha) = [sqrt(pi/(2 alpha)) * erf(sqrt(2 alpha)/2)]^d`.",
        "- Therefore the same-budget Monte Carlo estimator with `N` points has exact variance `Var[f(U)] / N = (M_{2,d}(alpha) - I_d(alpha)^2) / N`.",
        "- Using the CLT, the corresponding theoretical mean absolute error is approximated by `sqrt(2/pi) * sqrt(Var[f(U)] / N)`.",
        "",
        "## Comparison",
        "",
        f"- Successful Gaussian cases compared to theory: {len(successful)}",
        f"- Median observed-MC / theoretical-CLT-MAE ratio: {mc_ratio_median:.3f}",
        f"- Median Smolyak error / theoretical-MC-MAE ratio: {smolyak_ratio_median:.3f}",
        (
            "- Best Smolyak-vs-theory case: "
            f"{best_smolyak['label']} with Smolyak / MC-theory ratio "
            f"{best_smolyak['smolyak_to_mc_theory_mae_ratio']:.3e}"
        ),
        (
            "- Worst Smolyak-vs-theory case: "
            f"{worst_smolyak['label']} with Smolyak / MC-theory ratio "
            f"{worst_smolyak['smolyak_to_mc_theory_mae_ratio']:.3e}"
        ),
        "",
        "## Figures",
        "",
        f"![Observed vs theory MC (linear)]({figure_paths['mc_theory_ratio_linear'].name})",
        "",
        f"![Observed vs theory MC (log)]({figure_paths['mc_theory_ratio_log'].name})",
        "",
        f"![Smolyak vs MC theory (log)]({figure_paths['smolyak_vs_mc_theory'].name})",
        "",
        "## Per-Dtype Summary",
        "",
        "| Dtype | Median observed/theory MC ratio | Median Smolyak / MC-theory ratio |",
        "| --- | ---: | ---: |",
    ]

    for dtype, items in by_dtype.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    dtype,
                    f"{_median([case['mc_observed_to_theory_mae_ratio'] for case in items]):.3f}",
                    f"{_median([case['smolyak_to_mc_theory_mae_ratio'] for case in items]):.3f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Per-Level Summary",
            "",
            "| Level | Median observed/theory MC ratio | Median Smolyak / MC-theory ratio |",
            "| ---: | ---: | ---: |",
        ]
    )
    for level, items in by_level.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(level),
                    f"{_median([case['mc_observed_to_theory_mae_ratio'] for case in items]):.3f}",
                    f"{_median([case['smolyak_to_mc_theory_mae_ratio'] for case in items]):.3f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## References",
            "",
            "- E. Novak and K. Ritter, *High Dimensional Integration of Smooth Functions over Cubes* (1996): https://users.fmi.uni-jena.de/~novak/numer_math.html",
            "- T. Gerstner and M. Griebel, *Numerical integration using sparse grids* (1998), DOI: https://doi.org/10.1023/A:1019129717644",
            "- H.-J. Bungartz and M. Griebel, *Sparse grids* (2004), DOI: https://doi.org/10.1017/S0962492904000182",
            "- A. Hinrichs, E. Novak, and M. Ullrich, *On Weak Tractability of the Clenshaw-Curtis Smolyak Algorithm* (2013): https://arxiv.org/abs/1309.0360",
            "- Y. Tang, *A Note on Monte Carlo Integration in High Dimensions* (2022/2023): https://arxiv.org/abs/2206.09036",
            "- W. J. Morokoff and R. E. Caflisch, *Quasi-Monte Carlo integration* (1995), DOI: https://doi.org/10.1006/jcph.1995.1209",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    summary_path = Path(args.summary_json).resolve()
    summary = json.loads(summary_path.read_text())
    output_dir = Path(args.output_dir).resolve() if args.output_dir else summary_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if summary.get("family") != "gaussian":
        raise ValueError("This theory comparison currently supports only the gaussian family.")

    alpha = float(summary["gaussian_alpha"])
    successful_cases = [case for case in summary["cases"] if not case.get("failed")]
    theory_cases = [_build_theory_case(case, alpha) for case in successful_cases]

    dimensions = list(summary["dimensions"])
    levels = list(summary["levels"])
    dtypes = list(summary["dtypes"])
    theory_by_key = {(_case["dtype"], _case["level"], _case["dimension"]): _case for _case in theory_cases}

    mc_theory_ratio_linear_plot = output_dir / "mc_theory_ratio_linear.svg"
    mc_theory_ratio_log_plot = output_dir / "mc_theory_ratio_log.svg"
    smolyak_vs_mc_theory_plot = output_dir / "smolyak_vs_mc_theory.svg"

    mc_ratio_series = []
    mc_ratio_labels = []
    smolyak_ratio_series = []
    smolyak_ratio_labels = []
    for dtype in dtypes:
        for level in levels:
            label = f"{dtype}-l{level}"
            mc_ratio_labels.append(label)
            smolyak_ratio_labels.append(label)
            mc_ratio_series.append([
                _maybe_float(theory_by_key.get((dtype, level, dimension), {}).get("mc_observed_to_theory_mae_ratio"))
                for dimension in dimensions
            ])
            smolyak_ratio_series.append([
                _maybe_float(theory_by_key.get((dtype, level, dimension), {}).get("smolyak_to_mc_theory_mae_ratio"))
                for dimension in dimensions
            ])

    _write_line_svg(
        x_values=dimensions,
        series=mc_ratio_series,
        series_labels=mc_ratio_labels,
        title="Observed MC MAE / CLT-Theory MAE",
        y_label="Observed / theory",
        output_path=mc_theory_ratio_linear_plot,
        log_scale=False,
    )
    _write_line_svg(
        x_values=dimensions,
        series=mc_ratio_series,
        series_labels=mc_ratio_labels,
        title="Observed MC MAE / CLT-Theory MAE",
        y_label="Observed / theory",
        output_path=mc_theory_ratio_log_plot,
        log_scale=True,
    )
    _write_line_svg(
        x_values=dimensions,
        series=smolyak_ratio_series,
        series_labels=smolyak_ratio_labels,
        title="Smolyak Error / CLT-Theory MC MAE",
        y_label="Smolyak / theory-MC",
        output_path=smolyak_vs_mc_theory_plot,
        log_scale=True,
    )

    theory_summary = {
        "summary_json": str(summary_path),
        "output_dir": str(output_dir),
        "gaussian_alpha": alpha,
        "num_successful_cases": len(theory_cases),
        "cases": theory_cases,
        "mc_theory_ratio_linear_plot": str(mc_theory_ratio_linear_plot),
        "mc_theory_ratio_log_plot": str(mc_theory_ratio_log_plot),
        "smolyak_vs_mc_theory_plot": str(smolyak_vs_mc_theory_plot),
    }
    theory_summary_path = output_dir / "theory_summary.json"
    theory_summary_path.write_text(
        json.dumps(theory_summary, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    literature_md = output_dir / "literature_theory.md"
    _write_markdown(
        summary=summary,
        theory_cases=theory_cases,
        output_path=literature_md,
        figure_paths={
            "mc_theory_ratio_linear": mc_theory_ratio_linear_plot,
            "mc_theory_ratio_log": mc_theory_ratio_log_plot,
            "smolyak_vs_mc_theory": smolyak_vs_mc_theory_plot,
        },
    )

    print(
        json.dumps(
            {
                "literature_md": str(literature_md),
                "theory_summary_json": str(theory_summary_path),
                "mc_theory_ratio_linear_plot": str(mc_theory_ratio_linear_plot),
                "mc_theory_ratio_log_plot": str(mc_theory_ratio_log_plot),
                "smolyak_vs_mc_theory_plot": str(smolyak_vs_mc_theory_plot),
                "num_successful_cases": len(theory_cases),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
