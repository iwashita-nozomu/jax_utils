"""Configuration objects for Smolyak experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "SmolyakExperimentConfig",
]


@dataclass
class SmolyakExperimentConfig:
    """Configuration for a family of Smolyak scaling experiments."""

    min_dimension: int = 1
    max_dimension: int = 50
    min_level: int = 1
    max_level: int = 50
    dtypes: list[str] = field(
        default_factory=lambda: ["float16", "bfloat16", "float32", "float64"]
    )
    num_trials: int = 3
    timeout_seconds: float = 300.0
    device: Literal["cpu", "gpu"] = "cpu"
    num_accuracy_problems: int = 9
    coeff_start: float = -0.55
    coeff_stop: float = 0.65
    experimental: bool = False

    @property
    def total_cases(self) -> int:
        return (
            (self.max_dimension - self.min_dimension + 1)
            * (self.max_level - self.min_level + 1)
            * len(self.dtypes)
        )

    @property
    def total_tasks(self) -> int:
        return self.total_cases * self.num_trials

    def validate(self) -> None:
        if self.min_dimension < 1:
            raise ValueError("min_dimension must be >= 1")
        if self.max_dimension < self.min_dimension:
            raise ValueError("max_dimension must be >= min_dimension")
        if self.min_level < 1:
            raise ValueError("min_level must be >= 1")
        if self.max_level < self.min_level:
            raise ValueError("max_level must be >= min_level")
        if not self.dtypes:
            raise ValueError("dtypes must not be empty")
        supported_dtypes = {"float16", "bfloat16", "float32", "float64"}
        for dtype_name in self.dtypes:
            if dtype_name not in supported_dtypes:
                raise ValueError(f"Unsupported dtype: {dtype_name}")
        if self.num_trials < 1:
            raise ValueError("num_trials must be >= 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'")

    def to_dict(self) -> dict[str, object]:
        return {
            "min_dimension": self.min_dimension,
            "max_dimension": self.max_dimension,
            "min_level": self.min_level,
            "max_level": self.max_level,
            "dtypes": list(self.dtypes),
            "num_trials": self.num_trials,
            "timeout_seconds": self.timeout_seconds,
            "device": self.device,
            "num_accuracy_problems": self.num_accuracy_problems,
            "coeff_start": self.coeff_start,
            "coeff_stop": self.coeff_stop,
            "total_cases": self.total_cases,
            "total_tasks": self.total_tasks,
            "experimental": self.experimental,
        }
