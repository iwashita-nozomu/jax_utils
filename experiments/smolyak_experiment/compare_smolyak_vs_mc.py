#!/usr/bin/env python3
"""Compare Smolyak and Monte Carlo on the same analytic integrand family.

The script always records a same-budget Monte Carlo baseline that uses the same
number of samples as the realized Smolyak evaluation points. Optionally it also
searches for the Monte Carlo sample count whose mean absolute error is no
larger than the Smolyak absolute error. It records compile-vs-warm runtime for
both methods and writes a compact JSON report.
"""

from __future__ import annotations

import argparse
import json
import math
import time
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Smolyak and Monte Carlo on the same integrand family.",
    )
    parser.add_argument(
        "--platform",
        choices=["cpu", "gpu"],
        default="gpu",
        help="Execution platform. GPU is the default.",
    )
    parser.add_argument(
        "--dimension",
        type=int,
        default=2,
        help="Problem dimension.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=3,
        help="Smolyak level.",
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
        default="exponential",
        help="Analytic integrand family shared by both methods.",
    )
    parser.add_argument(
        "--coeff-start",
        type=float,
        default=-1.5,
        help="First coefficient for the exponential family.",
    )
    parser.add_argument(
        "--coeff-stop",
        type=float,
        default=1.5,
        help="Last coefficient for the exponential family.",
    )
    parser.add_argument(
        "--gaussian-alpha",
        type=float,
        default=0.8,
        help="Gaussian coefficient alpha for exp(-alpha ||x||^2).",
    )
    parser.add_argument(
        "--anisotropic-alpha-start",
        type=float,
        default=0.2,
        help="First per-axis alpha for anisotropic Gaussian families.",
    )
    parser.add_argument(
        "--anisotropic-alpha-stop",
        type=float,
        default=1.4,
        help="Last per-axis alpha for anisotropic Gaussian families.",
    )
    parser.add_argument(
        "--shift-start",
        type=float,
        default=-0.25,
        help="First per-axis shift for shifted families.",
    )
    parser.add_argument(
        "--shift-stop",
        type=float,
        default=0.25,
        help="Last per-axis shift for shifted families.",
    )
    parser.add_argument(
        "--laplace-beta-start",
        type=float,
        default=1.0,
        help="First per-axis beta for shifted Laplace-product families.",
    )
    parser.add_argument(
        "--laplace-beta-stop",
        type=float,
        default=6.0,
        help="Last per-axis beta for shifted Laplace-product families.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=16384,
        help="Smolyak chunk size.",
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
        "--start-samples",
        type=int,
        default=1,
        help="Initial Monte Carlo sample count.",
    )
    parser.add_argument(
        "--growth-factor",
        type=float,
        default=4.0,
        help="How fast to grow the Monte Carlo sample count.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1 << 20,
        help="Upper bound on the Monte Carlo sample search.",
    )
    parser.add_argument(
        "--warm-repeats",
        type=int,
        default=3,
        help="Number of warm runtime repeats to average after compilation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base PRNG seed for Monte Carlo.",
    )
    parser.add_argument(
        "--mc-seeds",
        type=int,
        default=8,
        help="Number of Monte Carlo seeds to average for same-budget and matched-accuracy comparisons.",
    )
    parser.add_argument(
        "--skip-matched-accuracy",
        action="store_true",
        help="Skip the geometric Monte Carlo search and only evaluate the same-budget baseline.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "results" / "smolyak_vs_mc"),
        help="Directory for the JSON report.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Print the JSON report without writing a file.",
    )
    return parser


def _scalar(value: Any) -> float:
    array = value
    if hasattr(array, "block_until_ready"):
        array.block_until_ready()
    return float(np.asarray(array).reshape(-1)[0])


def _call_and_time(fn: Callable[[], Any]) -> tuple[float, float]:
    start = time.perf_counter()
    value = fn()
    if hasattr(value, "block_until_ready"):
        value.block_until_ready()
    stop = time.perf_counter()
    return (stop - start) * 1000.0, _scalar(value)


def _sample_schedule(start: int, growth_factor: float, max_samples: int) -> list[int]:
    if start < 1:
        raise ValueError("start-samples must be positive.")
    if growth_factor <= 1.0:
        raise ValueError("growth-factor must be greater than 1.0.")
    if max_samples < start:
        raise ValueError("max-samples must be at least start-samples.")

    schedule = []
    sample_count = start
    while sample_count <= max_samples:
        schedule.append(sample_count)
        next_count = max(sample_count + 1, int(math.ceil(sample_count * growth_factor)))
        if next_count <= sample_count:
            next_count = sample_count + 1
        sample_count = next_count
    return schedule


def _benchmark_integrator(
    eqx_module: Any,
    integrator: Any,
    integrand: Callable[[Any], Any],
    analytic_value: float,
    warm_repeats: int,
) -> dict[str, Any]:
    compiled = eqx_module.filter_jit(integrator.integrate)

    first_ms, first_value = _call_and_time(lambda: compiled(integrand))
    warm_times: list[float] = []
    warm_value = first_value
    for _ in range(warm_repeats):
        warm_ms, warm_value = _call_and_time(lambda: compiled(integrand))
        warm_times.append(warm_ms)

    warm_runtime_ms = float(sum(warm_times) / len(warm_times)) if warm_times else 0.0
    compile_ms = max(0.0, first_ms - warm_runtime_ms)
    absolute_error = abs(warm_value - analytic_value)

    return {
        "value": warm_value,
        "absolute_error": absolute_error,
        "relative_error": absolute_error / abs(analytic_value) if analytic_value != 0.0 else absolute_error,
        "first_call_ms": first_ms,
        "warm_runtime_ms": warm_runtime_ms,
        "compile_ms": compile_ms,
    }


def _aggregate_seed_benchmarks(seed_benchmarks: list[dict[str, Any]]) -> dict[str, Any]:
    if not seed_benchmarks:
        raise ValueError("seed_benchmarks must not be empty.")

    values = [float(item["value"]) for item in seed_benchmarks]
    absolute_errors = [float(item["absolute_error"]) for item in seed_benchmarks]
    relative_errors = [float(item["relative_error"]) for item in seed_benchmarks]
    first_call_ms = [float(item["first_call_ms"]) for item in seed_benchmarks]
    warm_runtime_ms = [float(item["warm_runtime_ms"]) for item in seed_benchmarks]
    compile_ms = [float(item["compile_ms"]) for item in seed_benchmarks]

    return {
        "value_mean": float(np.mean(values)),
        "value_std": float(np.std(values)),
        "absolute_error_mean": float(np.mean(absolute_errors)),
        "absolute_error_std": float(np.std(absolute_errors)),
        "relative_error_mean": float(np.mean(relative_errors)),
        "relative_error_std": float(np.std(relative_errors)),
        "first_call_ms": float(np.mean(first_call_ms)),
        "warm_runtime_ms": float(np.mean(warm_runtime_ms)),
        "compile_ms": float(np.mean(compile_ms)),
    }


def _benchmark_monte_carlo_fixed_samples(
    jax_module: Any,
    eqx_module: Any,
    dimension: int,
    integrand: Callable[[Any], Any],
    analytic_value: float,
    *,
    seed: int,
    num_samples: int,
    warm_repeats: int,
    mc_seeds: int,
) -> dict[str, Any]:
    if mc_seeds < 1:
        raise ValueError("mc-seeds must be positive.")

    from python.jax_util.functional.monte_carlo import MonteCarloIntegrator

    seed_benchmarks: list[dict[str, Any]] = []
    for seed_offset in range(mc_seeds):
        mc = MonteCarloIntegrator(
            dimension=dimension,
            num_samples=num_samples,
            key=jax_module.random.PRNGKey(seed + seed_offset),
        )
        seed_benchmarks.append(
            _benchmark_integrator(
                eqx_module,
                mc,
                integrand,
                analytic_value,
                warm_repeats,
            )
        )

    aggregate = _aggregate_seed_benchmarks(seed_benchmarks)
    aggregate["num_samples"] = num_samples
    aggregate["num_seeds"] = mc_seeds
    return aggregate


def _find_monte_carlo_match(
    jax_module: Any,
    eqx_module: Any,
    dimension: int,
    integrand: Callable[[Any], Any],
    analytic_value: float,
    *,
    seed: int,
    start_samples: int,
    growth_factor: float,
    max_samples: int,
    warm_repeats: int,
    target_error: float,
    mc_seeds: int,
) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    schedule = _sample_schedule(start_samples, growth_factor, max_samples)
    chosen: dict[str, Any] | None = None

    from python.jax_util.functional.monte_carlo import MonteCarloIntegrator

    for sample_count in schedule:
        if mc_seeds < 1:
            raise ValueError("mc-seeds must be positive.")

        seed_benchmarks: list[dict[str, Any]] = []
        for seed_offset in range(mc_seeds):
            mc = MonteCarloIntegrator(
                dimension=dimension,
                num_samples=sample_count,
                key=jax_module.random.PRNGKey(seed + seed_offset),
            )
            seed_benchmarks.append(
                _benchmark_integrator(
                    eqx_module,
                    mc,
                    integrand,
                    analytic_value,
                    warm_repeats,
                )
            )

        item = {
            "num_samples": sample_count,
            "num_seeds": mc_seeds,
            **_aggregate_seed_benchmarks(seed_benchmarks),
        }
        history.append(item)
        if item["absolute_error_mean"] <= target_error:
            chosen = item
            break

    if chosen is None:
        chosen = {
            "num_samples": None,
            "value_mean": None,
            "value_std": None,
            "absolute_error_mean": None,
            "absolute_error_std": None,
            "relative_error_mean": None,
            "relative_error_std": None,
            "first_call_ms": None,
            "warm_runtime_ms": None,
            "compile_ms": None,
        }

    return {
        "chosen": chosen,
        "history": history,
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from python.jax_util.functional.smolyak import SmolyakIntegrator

    observed_platform = jax.devices()[0].platform if jax.devices() else "unknown"
    dtype = getattr(jnp, args.dtype)
    family_bundle = make_family_bundle(
        jnp=jnp,
        family=args.family,
        dimension=args.dimension,
        dtype=dtype,
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
    integrand = family_bundle.integrand
    analytic_value = family_bundle.analytic_value
    family_metadata = family_bundle.metadata
    dimension_weights = resolve_dimension_weights(
        dimension=args.dimension,
        dimension_weights_csv=args.dimension_weights,
        weight_scheme=args.weight_scheme,
        weight_scale=args.weight_scale,
    )

    started_at = time.time()

    smolyak = SmolyakIntegrator(
        dimension=args.dimension,
        level=args.level,
        dimension_weights=dimension_weights,
        requested_materialization_mode=args.requested_mode,
        max_vectorized_suffix_ndim=args.max_vectorized_suffix_ndim,
        batched_axis_order_strategy=args.batched_axis_order_strategy,
        dtype=dtype,
        chunk_size=args.chunk_size,
    )
    smolyak_benchmark = _benchmark_integrator(
        eqx,
        smolyak,
        integrand,
        analytic_value,
        args.warm_repeats,
    )

    mc_same_budget = _benchmark_monte_carlo_fixed_samples(
        jax_module=jax,
        eqx_module=eqx,
        dimension=args.dimension,
        integrand=integrand,
        analytic_value=analytic_value,
        seed=args.seed,
        num_samples=int(smolyak.num_evaluation_points),
        warm_repeats=args.warm_repeats,
        mc_seeds=args.mc_seeds,
    )

    if args.skip_matched_accuracy:
        mc_match = {
            "chosen": {
                "num_samples": None,
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
            "history": [],
            "skipped": True,
        }
    else:
        mc_match = _find_monte_carlo_match(
            jax,
            eqx,
            args.dimension,
            integrand,
            analytic_value,
            seed=args.seed,
            start_samples=args.start_samples,
            growth_factor=args.growth_factor,
            max_samples=args.max_samples,
            warm_repeats=args.warm_repeats,
            target_error=smolyak_benchmark["absolute_error"],
            mc_seeds=args.mc_seeds,
        )
        mc_match["skipped"] = False

    finished_at = time.time()
    started_at_utc = datetime.fromtimestamp(started_at, tz=timezone.utc).isoformat()
    finished_at_utc = datetime.fromtimestamp(finished_at, tz=timezone.utc).isoformat()

    report: dict[str, Any] = {
        "experiment": "smolyak_vs_monte_carlo_speed_match",
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc,
        "workspace_root": str(Path.cwd().resolve()),
        "requested_platform": args.platform,
        "platform": observed_platform,
        "dtype": args.dtype,
        "dimension": args.dimension,
        "level": args.level,
        "chunk_size": args.chunk_size,
        "requested_mode": args.requested_mode,
        "max_vectorized_suffix_ndim": args.max_vectorized_suffix_ndim,
        "batched_axis_order_strategy": args.batched_axis_order_strategy,
        "dimension_weights": None if dimension_weights is None else list(dimension_weights),
        "dimension_weights_label": format_dimension_weights(dimension_weights),
        "weight_scheme": args.weight_scheme,
        "weight_scale": args.weight_scale,
        "warm_repeats": args.warm_repeats,
        "mc_seeds": args.mc_seeds,
        "analytic_value": analytic_value,
        "family": family_metadata,
        "smolyak": {
            "active_axis_count": int(smolyak.active_axis_count),
            "inactive_axis_count": int(smolyak.dimension - smolyak.active_axis_count),
            "axis_level_ceilings": [int(value) for value in np.asarray(smolyak.axis_level_ceilings)],
            "num_terms": int(smolyak.num_terms),
            "num_evaluation_points": int(smolyak.num_evaluation_points),
            "vectorized_ndim": int(smolyak.vectorized_ndim),
            "max_vectorized_points": int(smolyak.max_vectorized_points),
            "max_vectorized_suffix_ndim": int(smolyak.max_vectorized_suffix_ndim),
            "batched_axis_order_strategy": str(smolyak.batched_axis_order_strategy),
            "storage_bytes": int(smolyak.storage_bytes),
            "materialization_mode": str(smolyak.materialization_mode),
            "uses_materialized_plan": str(smolyak.materialization_mode) in {"points", "indexed"},
            "materialized_point_count": int(smolyak.materialized_weights.shape[0]),
            "materialized_index_dtype": str(smolyak.materialized_rule_indices.dtype),
            "value": smolyak_benchmark["value"],
            "absolute_error": smolyak_benchmark["absolute_error"],
            "relative_error": smolyak_benchmark["relative_error"],
            "first_call_ms": smolyak_benchmark["first_call_ms"],
            "warm_runtime_ms": smolyak_benchmark["warm_runtime_ms"],
            "compile_ms": smolyak_benchmark["compile_ms"],
        },
        "monte_carlo_same_budget": {
            "num_samples": mc_same_budget["num_samples"],
            "num_seeds": mc_same_budget["num_seeds"],
            "value_mean": mc_same_budget["value_mean"],
            "value_std": mc_same_budget["value_std"],
            "absolute_error_mean": mc_same_budget["absolute_error_mean"],
            "absolute_error_std": mc_same_budget["absolute_error_std"],
            "relative_error_mean": mc_same_budget["relative_error_mean"],
            "relative_error_std": mc_same_budget["relative_error_std"],
            "first_call_ms": mc_same_budget["first_call_ms"],
            "warm_runtime_ms": mc_same_budget["warm_runtime_ms"],
            "compile_ms": mc_same_budget["compile_ms"],
        },
        "monte_carlo": {
            "search_skipped": bool(mc_match["skipped"]),
            "chosen_num_samples": mc_match["chosen"]["num_samples"],
            "value_mean": mc_match["chosen"]["value_mean"],
            "value_std": mc_match["chosen"]["value_std"],
            "absolute_error": mc_match["chosen"]["absolute_error_mean"],
            "absolute_error_std": mc_match["chosen"]["absolute_error_std"],
            "relative_error": mc_match["chosen"]["relative_error_mean"],
            "relative_error_std": mc_match["chosen"]["relative_error_std"],
            "first_call_ms": mc_match["chosen"]["first_call_ms"],
            "warm_runtime_ms": mc_match["chosen"]["warm_runtime_ms"],
            "compile_ms": mc_match["chosen"]["compile_ms"],
            "history": mc_match["history"],
        },
        "summary": {
            "smolyak_faster_on_warm_runtime": (
                mc_match["chosen"]["warm_runtime_ms"] is not None
                and smolyak_benchmark["warm_runtime_ms"] < mc_match["chosen"]["warm_runtime_ms"]
            ),
            "smolyak_more_accurate_same_budget": (
                smolyak_benchmark["absolute_error"] < mc_same_budget["absolute_error_mean"]
            ),
            "smolyak_faster_same_budget": (
                smolyak_benchmark["warm_runtime_ms"] < mc_same_budget["warm_runtime_ms"]
            ),
            "smolyak_error": smolyak_benchmark["absolute_error"],
            "monte_carlo_error": mc_match["chosen"]["absolute_error_mean"],
            "monte_carlo_same_budget_error": mc_same_budget["absolute_error_mean"],
        },
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"compare_smolyak_vs_mc_{int(started_at)}.json"
    report["output_json"] = str(output_path)
    if not args.no_write:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=True, indent=2, sort_keys=True)

    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
