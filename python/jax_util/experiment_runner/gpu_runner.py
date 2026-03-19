"""GPU 固有のリソース表現とシンプルな GPU スケジューラ。

- 環境変数 `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES` の解釈を行うユーティリティを提供する。
- 単純な FIFO ベースで GPU ID を割り当てる `StandardGPUScheduler` を実装する。

このモジュールは副作用を持たないように設計されています（インポート時に環境を検査しません）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from typing import Callable, Generic, Mapping, TypeVar, cast

from .protocols import TaskContext
from .runner import StandardResourceCapacity, StandardScheduler


T = TypeVar("T")

_GPU_ENV_NAMES = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")

__all__ = [
    "visible_gpu_ids_from_environment",
    "GPUResourceCapacity",
    "StandardGPUScheduler",
]


def visible_gpu_ids_from_environment(
    environ: Mapping[str, str] | None = None,
    /,
) -> tuple[int, ...]:
    """環境変数から可視 GPU ID のタプルを返す。

    - 空文字列, "-1", "none", "void" は "GPU を使わない" を意味して空タプルを返す。
    - "all" のような特殊語はここでは扱わず、明示的な ID 列を期待する。
    - 不正なトークンが混入していれば ValueError を投げる。
    """
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
    """GPU 使用に特化したリソース容量表現。

    - `gpu_ids` は利用可能な GPU ID のタプルで空であってはならない。
    - `max_workers` は `len(gpu_ids)` と一致する必要がある。
    """
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
        """環境変数から `GPUResourceCapacity` を構築するユーティリティ。"""
        gpu_ids = visible_gpu_ids_from_environment(environ)
        if not gpu_ids:
            raise ValueError("no visible GPUs found in environment.")
        return cls(
            max_workers=len(gpu_ids),
            gpu_ids=gpu_ids,
        )


class StandardGPUScheduler(StandardScheduler[T], Generic[T]):
    """単純な GPU ID FIFO を用いたスケジューラ実装。

    - `next_case()` は利用可能な GPU があればケースを返し、コンテキストに GPU 指定を埋める。
    - `on_finish()` で GPU ID をプールへ戻す。
    """
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
