from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from typing import Callable, Generic, Mapping, TypeVar, cast

from .protocols import TaskContext
from .runner import StandardResourceCapacity, StandardScheduler


T = TypeVar("T")

_GPU_ENV_NAMES = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")


def visible_gpu_ids_from_environment(
    environ: Mapping[str, str] | None = None,
    /,
) -> tuple[int, ...]:
    source = os.environ if environ is None else environ

    for env_name in _GPU_ENV_NAMES:
        raw_value = source.get(env_name)
        if raw_value is None:
            continue

        stripped_value = raw_value.strip()
        if stripped_value in {"", "-1", "none", "void"}:
            return ()

        gpu_ids: list[int] = []
        for token in stripped_value.split(","):
            item = token.strip()
            if not item:
                continue
            if not item.isdigit():
                raise ValueError(
                    f"{env_name} must contain comma-separated integer GPU ids."
                )
            gpu_ids.append(int(item))
        return tuple(gpu_ids)

    raise ValueError(
        "CUDA_VISIBLE_DEVICES or NVIDIA_VISIBLE_DEVICES must be set for GPU scheduling."
    )


@dataclass(frozen=True)
class GPUResourceCapacity(StandardResourceCapacity):
    gpu_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.gpu_ids:
            raise ValueError("gpu_ids must not be empty.")
        if self.max_workers != len(self.gpu_ids):
            raise ValueError("max_workers must match len(gpu_ids).")

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        /,
    ) -> GPUResourceCapacity:
        gpu_ids = visible_gpu_ids_from_environment(environ)
        if not gpu_ids:
            raise ValueError("no visible GPUs found in environment.")
        return cls(
            max_workers=len(gpu_ids),
            gpu_ids=gpu_ids,
        )


class StandardGPUScheduler(StandardScheduler[T], Generic[T]):
    def __init__(
        self,
        resource_capacity: GPUResourceCapacity,
        cases: list[T],
        context_builder: Callable[[T], TaskContext] | None = None,
        disable_gpu_preallocation: bool = False,
    ) -> None:
        super().__init__(
            resource_capacity=resource_capacity,
            cases=cases,
            context_builder=context_builder,
        )
        self._available_gpu_ids = deque(resource_capacity.gpu_ids)
        self._disable_gpu_preallocation = disable_gpu_preallocation

    @property
    def resource_capacity(self) -> GPUResourceCapacity:
        return cast(GPUResourceCapacity, self._resource_capacity)

    def next_case(self) -> tuple[T, TaskContext] | None:
        if not self._pending_cases or not self._available_gpu_ids:
            return None

        case = self._pending_cases.pop(0)
        gpu_id = self._available_gpu_ids.popleft()
        context = self._build_context(case)
        context["gpu_id"] = str(gpu_id)
        context["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        context["NVIDIA_VISIBLE_DEVICES"] = str(gpu_id)
        if self._disable_gpu_preallocation:
            context["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        return case, context

    def on_finish(self, case: T, context: TaskContext, exit_code: int) -> None:
        super().on_finish(case, context, exit_code)

        gpu_id_text = context.get("gpu_id")
        if gpu_id_text is None or not gpu_id_text.isdigit():
            raise ValueError("gpu_id must be present in TaskContext.")

        self._available_gpu_ids.append(int(gpu_id_text))
