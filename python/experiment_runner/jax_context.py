"""JAX と multiprocessing を安全に併用するための補助関数。"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing.context import SpawnContext
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = [
    "get_spawn_context",
    "disable_jax_memory_preallocation",
    "check_picklable",
    "create_jax_safe_process_pool",
]


def get_spawn_context() -> SpawnContext:
    """`spawn` context を返す。"""
    try:
        return mp.get_context("spawn")
    except ValueError as exc:
        raise ValueError("Failed to get 'spawn' multiprocessing context.") from exc


def disable_jax_memory_preallocation(gpu_devices: bool = False) -> None:
    """JAX のメモリ先取り設定を安全側へ寄せる。"""
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
    if gpu_devices:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        return
    os.environ.setdefault("JAX_PLATFORMS", "cpu")


def check_picklable(obj: Any, name: str = "object") -> None:
    """ProcessPoolExecutor に送れるオブジェクトか検証する。"""
    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, TypeError, AttributeError) as exc:
        raise ValueError(
            f"{name} '{type(obj).__name__}' is not picklable."
        ) from exc


@contextmanager
def create_jax_safe_process_pool(
    max_workers: int,
    *,
    mp_context: SpawnContext | None = None,
) -> Generator[ProcessPoolExecutor, None, None]:
    """JAX 向けに `spawn` context を使う process pool を返す。"""
    if max_workers < 1:
        raise ValueError(f"max_workers must be positive, got {max_workers}")
    resolved_context = get_spawn_context() if mp_context is None else mp_context
    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=resolved_context,
    )
    try:
        yield executor
    finally:
        executor.shutdown(wait=True)
