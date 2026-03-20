# -*- coding: utf-8 -*-
"""JAX fork() 互換性ユーティリティ。

JAX は fork() ベースの multiprocessing と互換性が不安定です。
/bin/python3 /workspace/python/tests/neuralnetwork/test_neuralnetwork.py

1. **JIT コンパイル後の fork**: JAX がメモリを確保して JIT コンパイルを完了した
   後に fork() されると、child プロセス内で JAX 内部状態（GPU メモリポインタ等）
   が不正になる。

2. **メモリ先取り**: デフォルトでは GPU メモリを先制的に確保してしまい、
   child プロセス間でメモリが共有されない。

/bin/python3 /workspace mitigate するために：/python/tests/neuralnetwork/test_neuralnetwork.py

- **spawn コンテキスト**: ProcessPoolExecutor に mp_context='spawn' を指定。
  各 worker プロセスが独立した Python インタプリタを起動し、fork() ベースの
  共有メモリ問題を回避する。

- **メモリ先取り無効化**: 環境変数の設定で JAX メモリ先取りを無効化。
  例). JAX_PLATFORMS=cpu で CPU-only 実行、XLA_FLAGS でメモリ前割当を制御。

/bin/python3 /workspace/python/tests/neuralnetwork/test_neuralnetwork.py

    from jax_util.experiment_runner.jax_context import (
        create_jax_safe_process_pool,
        disable_jax_memory_preallocation,
    )

    # メモリ先取りを無効化
    disable_jax_memory_preallocation()

    # JAX 対応の ProcessPoolExecutor を作成
    with create_jax_safe_process_pool(max_workers=4) as ex:
        futures = [ex.submit(jax_task, case) for case in cases]
        # ...
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

__all__ = [
    "get_spawn_context",
    "disable_jax_memory_preallocation",
    "create_jax_safe_process_pool",
]


def get_spawn_context() -> mp.context.SpawnContext:
    """'spawn' コンテキストを取得し、JAX fork() 互換性問題を回避する。"""
    try:
        ctx = mp.get_context("spawn")
        return ctx
    except ValueError as e:
        raise ValueError(
            "Failed to get 'spawn' multiprocessing context. "
            "spawn is not available on this platform."
        ) from e


def disable_jax_memory_preallocation(
    gpu_devices: bool = False,
) -> None:
    """JAX メモリ先取りを無効化する環境変数を設定。
    
    Parameters
    ----------
    gpu_devices : bool, optional
        GPU を使用する場合 True を指定。デフォルト False（CPU-only）。
    """
    # ホストの device 数を 1 に制限
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

    if gpu_devices:
        # GPU メモリ先取りを無効化
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    else:
        # CPU-only 実行に強制
        os.environ.setdefault("JAX_PLATFORMS", "cpu")


@contextmanager
def create_jax_safe_process_pool(
    max_workers: int,
    *,
    mp_context: mp.context.SpawnContext | None = None,
) -> Generator[ProcessPoolExecutor, None, None]:
    """JAX fork() 互換性を確保した ProcessPoolExecutor を作成する context manager."""
    if max_workers < 1:
        raise ValueError(f"max_workers must be positive, got {max_workers}")

    if mp_context is None:
        mp_context = get_spawn_context()

    executor = ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_context,
    )
    try:
        yield executor
    finally:
        executor.shutdown(wait=True)
