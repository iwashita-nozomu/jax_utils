"""Case generation and resource estimation for Smolyak experiments."""

from __future__ import annotations

import os
from typing import Any, Mapping

from jax_util.experiment_runner import FullResourceEstimate

from .runner_config import SmolyakExperimentConfig

__all__ = [
    "estimate_case_resources",
    "generate_cases",
]


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


def generate_cases(config: SmolyakExperimentConfig, /) -> list[dict[str, Any]]:
    """Generate the Cartesian product of experiment parameters."""
    config.validate()

    cases: list[dict[str, Any]] = []
    case_index = 0
    for dimension in range(config.min_dimension, config.max_dimension + 1):
        for level in range(config.min_level, config.max_level + 1):
            for dtype_name in config.dtypes:
                for trial_index in range(config.num_trials):
                    case_id = f"d{dimension}_l{level}_{dtype_name}_t{trial_index}"
                    cases.append(
                        {
                            "case_id": case_id,
                            "dimension": dimension,
                            "level": level,
                            "dtype": dtype_name,
                            "trial_index": trial_index,
                            "index": case_index,
                            "device": config.device,
                        }
                    )
                    case_index += 1
    return cases
