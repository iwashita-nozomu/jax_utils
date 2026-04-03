"""Case definitions for Smolyak experiments.

This module is the single case-layer entry point for the Smolyak experiment
directory:

- Cartesian case generation for runner-style sweeps
- Conservative resource estimation
- Analytic integrand families with closed-form reference values
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import os
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from python.experiment_runner import FullResourceEstimate

__all__ = [
    "FamilyBundle",
    "SUPPORTED_FAMILIES",
    "SUPPORTED_FLOAT_DTYPES",
    "SUPPORTED_INDEX_DTYPES",
    "estimate_case_resources",
    "generate_cases",
    "make_family_bundle",
]


SUPPORTED_FAMILIES = (
    "gaussian",
    "anisotropic_gaussian",
    "shifted_anisotropic_gaussian",
    "quadratic",
    "absolute_sum",
    "exponential",
    "balanced_exponential",
    "shifted_laplace_product",
)

SUPPORTED_FLOAT_DTYPES = (
    "float16",
    "bfloat16",
    "float32",
    "float64",
)

SUPPORTED_INDEX_DTYPES = (
    "int32",
    "int64",
)


@dataclass(frozen=True)
class FamilyBundle:
    integrand: Callable[[Any], Any]
    analytic_value: float
    metadata: dict[str, object]
    eval_one: Callable[[Any, Any], Any]
    single_scale: Any


def _safe_exponential_box_factors(coefficients_np: np.ndarray) -> np.ndarray:
    factors = np.ones_like(coefficients_np, dtype=np.float64)
    nonzero_mask = np.abs(coefficients_np) > 1e-15
    factors[nonzero_mask] = (
        2.0 * np.sinh(0.5 * coefficients_np[nonzero_mask])
    ) / coefficients_np[nonzero_mask]
    return factors


def estimate_case_resources(case: Mapping[str, Any], /) -> FullResourceEstimate:
    """Return a conservative resource estimate for a Smolyak experiment case."""
    dimension = int(case["dimension"])
    level = int(case["level"])
    dtype_name = str(case["dtype"])
    device = str(case.get("device", "cpu"))

    dtype_bytes = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "float64": 8,
    }
    bytes_per_element = dtype_bytes.get(dtype_name, 4)

    max_points_multiplier = 2 ** min(level, 6)
    estimated_points = max_points_multiplier ** min(dimension, 4)
    host_memory_bytes = int(estimated_points * bytes_per_element * 2.5)

    default_max_memory = 8 * 1024 * 1024 * 1024
    env_max = os.environ.get("ESTIMATE_MAX_MEMORY_BYTES")
    try:
        max_memory = int(env_max) if env_max is not None else default_max_memory
    except ValueError:
        max_memory = default_max_memory
    host_memory_bytes = min(host_memory_bytes, max_memory)

    env_min = os.environ.get("ESTIMATE_MIN_MEMORY_BYTES")
    if env_min is not None:
        try:
            host_memory_bytes = max(host_memory_bytes, int(env_min))
        except ValueError:
            pass

    if device == "gpu":
        gpu_count = 1
        gpu_memory_bytes = host_memory_bytes
    else:
        gpu_count = 0
        gpu_memory_bytes = 0

    return FullResourceEstimate(
        host_memory_bytes=host_memory_bytes,
        gpu_count=gpu_count,
        gpu_memory_bytes=gpu_memory_bytes,
        gpu_slots=1,
    )


def generate_cases(
    *,
    dimensions: Iterable[int],
    levels: Iterable[int],
    float_dtypes: Iterable[str],
    index_dtypes: Iterable[str],
    family: str,
    platform: str,
    batch_size: int,
    warm_repeats: int,
    chunk_size: int,
) -> list[dict[str, Any]]:
    """Generate the Cartesian product of benchmark parameters."""
    cases: list[dict[str, Any]] = []
    case_index = 0
    for dimension in dimensions:
        for level in levels:
            for float_dtype in float_dtypes:
                if float_dtype not in SUPPORTED_FLOAT_DTYPES:
                    raise ValueError(f"Unsupported float dtype: {float_dtype}")
                for index_dtype in index_dtypes:
                    if index_dtype not in SUPPORTED_INDEX_DTYPES:
                        raise ValueError(f"Unsupported index dtype: {index_dtype}")
                    case_id = f"{family}_d{dimension:03d}_l{level:03d}_{float_dtype}_{index_dtype}"
                    cases.append(
                        {
                            "case_id": case_id,
                            "family": family,
                            "dimension": int(dimension),
                            "level": int(level),
                            "dtype": str(float_dtype),
                            "index_dtype": str(index_dtype),
                            "platform": str(platform),
                            "batch_size": int(batch_size),
                            "warm_repeats": int(warm_repeats),
                            "chunk_size": int(chunk_size),
                            "index": case_index,
                        }
                    )
                    case_index += 1
    return cases


def make_family_bundle(
    *,
    jnp: Any,
    family: str,
    dimension: int,
    dtype: Any,
    gaussian_alpha: float,
    anisotropic_alpha_start: float,
    anisotropic_alpha_stop: float,
    shift_start: float,
    shift_stop: float,
    laplace_beta_start: float,
    laplace_beta_stop: float,
    coeff_start: float,
    coeff_stop: float,
) -> FamilyBundle:
    if family == "gaussian":
        alpha = gaussian_alpha

        def integrand(x: Any) -> Any:
            return jnp.asarray([jnp.exp(-alpha * jnp.sum(x**2, axis=-1))], dtype=dtype)

        factor = math.sqrt(math.pi / alpha) * math.erf(0.5 * math.sqrt(alpha))
        analytic = float(factor**dimension)

        def eval_one(scale: Any, integrator: Any) -> Any:
            current_alpha = jnp.asarray(alpha, dtype=integrator.dtype) * scale
            return integrator.integrate(
                lambda x: jnp.asarray(
                    [jnp.exp(-current_alpha * jnp.sum(x**2, axis=-1))],
                    dtype=integrator.dtype,
                )
            )[0]

        return FamilyBundle(
            integrand=integrand,
            analytic_value=analytic,
            metadata={"family": family, "alpha": alpha},
            eval_one=eval_one,
            single_scale=jnp.asarray(1.0, dtype=dtype),
        )

    if family == "anisotropic_gaussian":
        alphas_np = np.linspace(
            anisotropic_alpha_start,
            anisotropic_alpha_stop,
            num=dimension,
            dtype=np.float64,
        )
        alphas = jnp.asarray(alphas_np, dtype=dtype)

        def integrand(x: Any) -> Any:
            return jnp.asarray([jnp.exp(-jnp.dot(alphas, x**2))], dtype=dtype)

        factors = [
            math.sqrt(math.pi / float(alpha)) * math.erf(0.5 * math.sqrt(float(alpha)))
            for alpha in alphas_np
        ]
        analytic = float(np.prod(np.asarray(factors, dtype=np.float64)))

        def eval_one(scale: Any, integrator: Any) -> Any:
            scaled_alphas = jnp.asarray(alphas, dtype=integrator.dtype) * scale
            return integrator.integrate(
                lambda x: jnp.asarray(
                    [jnp.exp(-jnp.dot(scaled_alphas, x**2))],
                    dtype=integrator.dtype,
                )
            )[0]

        return FamilyBundle(
            integrand=integrand,
            analytic_value=analytic,
            metadata={
                "family": family,
                "alpha_start": anisotropic_alpha_start,
                "alpha_stop": anisotropic_alpha_stop,
            },
            eval_one=eval_one,
            single_scale=jnp.asarray(1.0, dtype=dtype),
        )

    if family == "shifted_anisotropic_gaussian":
        alphas_np = np.linspace(
            anisotropic_alpha_start,
            anisotropic_alpha_stop,
            num=dimension,
            dtype=np.float64,
        )
        shifts_np = np.linspace(shift_start, shift_stop, num=dimension, dtype=np.float64)
        alphas = jnp.asarray(alphas_np, dtype=dtype)
        shifts = jnp.asarray(shifts_np, dtype=dtype)

        def integrand(x: Any) -> Any:
            centered = x - shifts
            return jnp.asarray([jnp.exp(-jnp.dot(alphas, centered**2))], dtype=dtype)

        factors = [
            math.sqrt(math.pi) / (2.0 * math.sqrt(float(alpha))) * (
                math.erf(math.sqrt(float(alpha)) * (0.5 - float(shift)))
                + math.erf(math.sqrt(float(alpha)) * (0.5 + float(shift)))
            )
            for alpha, shift in zip(alphas_np, shifts_np, strict=True)
        ]
        analytic = float(np.prod(np.asarray(factors, dtype=np.float64)))

        def eval_one(scale: Any, integrator: Any) -> Any:
            scaled_alphas = jnp.asarray(alphas, dtype=integrator.dtype) * scale
            current_shifts = jnp.asarray(shifts, dtype=integrator.dtype)
            return integrator.integrate(
                lambda x: jnp.asarray(
                    [
                        jnp.exp(
                            -jnp.dot(scaled_alphas, (x - current_shifts) ** 2)
                        )
                    ],
                    dtype=integrator.dtype,
                )
            )[0]

        return FamilyBundle(
            integrand=integrand,
            analytic_value=analytic,
            metadata={
                "family": family,
                "alpha_start": anisotropic_alpha_start,
                "alpha_stop": anisotropic_alpha_stop,
                "shift_start": shift_start,
                "shift_stop": shift_stop,
            },
            eval_one=eval_one,
            single_scale=jnp.asarray(1.0, dtype=dtype),
        )

    if family == "quadratic":
        def integrand(x: Any) -> Any:
            return jnp.asarray([jnp.sum(x**2, axis=-1)], dtype=dtype)

        def eval_one(scale: Any, integrator: Any) -> Any:
            del scale
            return integrator.integrate(
                lambda x: jnp.asarray([jnp.sum(x**2, axis=-1)], dtype=integrator.dtype)
            )[0]

        return FamilyBundle(
            integrand=integrand,
            analytic_value=float(dimension / 12.0),
            metadata={"family": family},
            eval_one=eval_one,
            single_scale=jnp.asarray(1.0, dtype=dtype),
        )

    if family == "absolute_sum":
        def integrand(x: Any) -> Any:
            return jnp.asarray([jnp.sum(jnp.abs(x), axis=-1)], dtype=dtype)

        def eval_one(scale: Any, integrator: Any) -> Any:
            del scale
            return integrator.integrate(
                lambda x: jnp.asarray([jnp.sum(jnp.abs(x), axis=-1)], dtype=integrator.dtype)
            )[0]

        return FamilyBundle(
            integrand=integrand,
            analytic_value=float(dimension / 4.0),
            metadata={"family": family},
            eval_one=eval_one,
            single_scale=jnp.asarray(1.0, dtype=dtype),
        )

    if family == "exponential":
        coefficients_np = np.linspace(coeff_start, coeff_stop, num=dimension, dtype=np.float64)
        coefficients = jnp.asarray(coefficients_np, dtype=dtype)

        def integrand(x: Any) -> Any:
            return jnp.asarray([jnp.exp(jnp.dot(coefficients, x))], dtype=dtype)

        analytic = float(np.prod(_safe_exponential_box_factors(coefficients_np)))

        def eval_one(scale: Any, integrator: Any) -> Any:
            scaled_coeffs = jnp.asarray(coefficients, dtype=integrator.dtype) * scale
            return integrator.integrate(
                lambda x: jnp.asarray(
                    [jnp.exp(jnp.dot(scaled_coeffs, x))],
                    dtype=integrator.dtype,
                )
            )[0]

        return FamilyBundle(
            integrand=integrand,
            analytic_value=analytic,
            metadata={"family": family, "coeff_start": coeff_start, "coeff_stop": coeff_stop},
            eval_one=eval_one,
            single_scale=jnp.asarray(1.0, dtype=dtype),
        )

    if family == "balanced_exponential":
        coefficients_np = np.linspace(coeff_start, coeff_stop, num=dimension, dtype=np.float64)
        coefficients = jnp.asarray(coefficients_np, dtype=dtype)

        def _exp_factor_product(scale_value: float) -> float:
            scaled_coeffs_np = scale_value * coefficients_np
            return float(np.prod(_safe_exponential_box_factors(scaled_coeffs_np)))

        def _exp_factor_product_jax(scaled_coeffs: Any) -> Any:
            factors = jnp.where(
                jnp.abs(scaled_coeffs) > 1e-15,
                (2.0 * jnp.sinh(0.5 * scaled_coeffs)) / scaled_coeffs,
                1.0,
            )
            return jnp.prod(factors)

        baseline = _exp_factor_product(1.0)

        def integrand(x: Any) -> Any:
            return jnp.asarray([jnp.exp(jnp.dot(coefficients, x)) - baseline], dtype=dtype)

        def eval_one(scale: Any, integrator: Any) -> Any:
            scaled_coeffs = jnp.asarray(coefficients, dtype=integrator.dtype) * scale
            current_baseline = _exp_factor_product_jax(scaled_coeffs)
            return integrator.integrate(
                lambda x: jnp.asarray(
                    [jnp.exp(jnp.dot(scaled_coeffs, x)) - current_baseline],
                    dtype=integrator.dtype,
                )
            )[0]

        return FamilyBundle(
            integrand=integrand,
            analytic_value=0.0,
            metadata={"family": family, "coeff_start": coeff_start, "coeff_stop": coeff_stop},
            eval_one=eval_one,
            single_scale=jnp.asarray(1.0, dtype=dtype),
        )

    if family == "shifted_laplace_product":
        betas_np = np.linspace(laplace_beta_start, laplace_beta_stop, num=dimension, dtype=np.float64)
        shifts_np = np.linspace(shift_start, shift_stop, num=dimension, dtype=np.float64)
        betas = jnp.asarray(betas_np, dtype=dtype)
        shifts = jnp.asarray(shifts_np, dtype=dtype)

        def integrand(x: Any) -> Any:
            centered = jnp.abs(x - shifts)
            return jnp.asarray([jnp.exp(-jnp.dot(betas, centered))], dtype=dtype)

        factors = np.where(
            np.abs(betas_np) > 1e-15,
            (
                2.0
                - np.exp(-betas_np * (0.5 + shifts_np))
                - np.exp(-betas_np * (0.5 - shifts_np))
            )
            / betas_np,
            1.0,
        )
        analytic = float(np.prod(factors))

        def eval_one(scale: Any, integrator: Any) -> Any:
            scaled_betas = jnp.asarray(betas, dtype=integrator.dtype) * scale
            current_shifts = jnp.asarray(shifts, dtype=integrator.dtype)
            return integrator.integrate(
                lambda x: jnp.asarray(
                    [
                        jnp.exp(
                            -jnp.dot(scaled_betas, jnp.abs(x - current_shifts))
                        )
                    ],
                    dtype=integrator.dtype,
                )
            )[0]

        return FamilyBundle(
            integrand=integrand,
            analytic_value=analytic,
            metadata={
                "family": family,
                "beta_start": laplace_beta_start,
                "beta_stop": laplace_beta_stop,
                "shift_start": shift_start,
                "shift_stop": shift_stop,
            },
            eval_one=eval_one,
            single_scale=jnp.asarray(1.0, dtype=dtype),
        )

    raise ValueError(f"Unknown family: {family}")
