"""Shared helpers for anisotropic Smolyak dimension-weight schedules."""

from __future__ import annotations

import math


SUPPORTED_WEIGHT_SCHEMES = (
    "none",
    "log2",
    "sqrt",
    "linear",
)


def _parse_csv_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("At least one integer value is required.")
    return values


def resolve_dimension_weights(
    *,
    dimension: int,
    dimension_weights_csv: str | None = None,
    weight_scheme: str = "none",
    weight_scale: float = 1.0,
) -> tuple[int, ...] | None:
    if dimension_weights_csv is not None:
        weights = _parse_csv_ints(dimension_weights_csv)
        if len(weights) != dimension:
            raise ValueError("dimension-weights must have length equal to dimension.")
        if any(weight < 1 for weight in weights):
            raise ValueError("dimension-weights must be positive integers.")
        return weights

    if weight_scheme == "none":
        return None
    if weight_scale <= 0.0:
        raise ValueError("weight-scale must be positive.")

    weights: list[int] = []
    for axis in range(1, dimension + 1):
        if weight_scheme == "linear":
            raw_weight = float(axis)
        elif weight_scheme == "sqrt":
            raw_weight = math.sqrt(float(axis))
        elif weight_scheme == "log2":
            raw_weight = math.log2(float(axis) + 1.0)
        else:
            raise ValueError(f"Unknown weight scheme: {weight_scheme}")
        weights.append(max(1, int(math.ceil(weight_scale * raw_weight))))
    return tuple(weights)


def format_dimension_weights(weights: tuple[int, ...] | None) -> str:
    if weights is None:
        return "isotropic"
    preview = ",".join(str(weight) for weight in weights[:8])
    if len(weights) > 8:
        preview += ",..."
    return preview
