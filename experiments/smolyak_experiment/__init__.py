"""Smolyak scaling experiment package."""

from .cases import estimate_case_resources, generate_cases
from .results_aggregator import (
    aggregate_by_dimension,
    aggregate_by_dtype,
    aggregate_by_level,
    compute_statistics,
    filter_failures,
)
from .runner_config import SmolyakExperimentConfig

__all__ = [
    "SmolyakExperimentConfig",
    "aggregate_by_dimension",
    "aggregate_by_dtype",
    "aggregate_by_level",
    "compute_statistics",
    "estimate_case_resources",
    "filter_failures",
    "generate_cases",
]
