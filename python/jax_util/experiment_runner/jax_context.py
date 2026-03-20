# -*- coding: utf-8 -*-
"""JAX fork() 互換性ユーティリティ。

JAX は fork() ベースの multiprocessing と互換性が不安定です。
具体的には：

1. **JIT コンパイル後の fork**: JAX がメモリを確保して JIT コンパイルを完了した
   後に fork() されると、child プロセス内で JAX 内部状態（GPU メモリポインタ等）
   が不正になる。

2. **メモリ先取り**: デフォルトでは GPU メモリを先制的に確保してしまい、
   child プロセス間でメモリが共有されない。

3. **CPU-only でも注意**: JAX は import 時に初期化を行うため、CPU-only 実行でも
   fork() ベースの状態共有は不確定。spawn コンテキストで各 worker プロセスが
   independent な Python インタプリタを起動することが最も安全。

本モジュールはこれらの問題を mitigate するために：

- **spawn コンテキスト**: ProcessPoolExecutor に mp_context='spawn' を指定。
  各 worker プロセスが独立した Python インタプリタを起動し、fork() ベースの
  共有メモリ問題を回避する。CPU-only でも使用すべき。

- **メモリ先取り無効化**: 環境変数の設定で JAX メモリ先取りを無効化。
  例). JAX_PLATFORMS=cpu で CPU-only 実行、XLA_FLAGS でメモリ前割当を制御。

- **Pickle 化可能性チェック**: ProcessPoolExecutor に送出する worker 責任任務が
  pickle 化可能であることを事前に検証。不可能な場合は早期エラーを返す。
"""

from __future__ import annotations

import multiprocessing as mp
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


def get_spawn_context() -> mp.context.SpawnContext:
    """'spawn' コンテキストを取得し、JAX fork() 互換性問題を回避する。
    
    Notes
    -----
    spawn コンテキストは CPU-only でも推奨されます。JAX は import 時に
    キャッシュやトレース情報を初期化するため、fork() ベースでは状態が
    不確定になる可能性があります。
    """
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


def check_picklable(obj: Any, name: str = "object") -> None:
    """
    オブジェクトが pickle 化可能であることを確認する。
    
    ProcessPoolExecutor で別プロセスに送出する worker や task は
    pickle 化可能である必要があります。この関数で事前検証し、
    早期エラーを検出します。
    
    Parameters
    ----------
    obj : Any
        pickle 化可能性を確認するオブジェクト
    
    name : str, optional
        エラーメッセージ出力用の名前。デフォルト "object"。
    
    Raises
    ------
    ValueError
        オブジェクトが pickle 化不可能な場合
        
    Examples
    --------
    >>> from jax_util.experiment_runner.jax_context import check_picklable
    >>> 
    >>> class MyWorker:
    ...     def __call__(self, case, context):
    ...         return 0
    >>> 
    >>> worker = MyWorker()
    >>> check_picklable(worker, name="SmolyakWorker")
    >>> # OK - no error
    >>> 
    >>> import threading
    >>> lock = threading.Lock()
    >>> check_picklable(lock, name="threading.Lock")
    Traceback (most recent call last):
        ...
    ValueError: threading.Lock 'lock' is not picklable...
    """
    try:
        pickle.dumps(obj)
    except (pickle.PicklingError, TypeError) as e:
        raise ValueError(
            f"{name} '{type(obj).__name__}' is not picklable. "
            f"ProcessPoolExecutor cannot send it to worker processes. "
            f"Error: {e}"
        ) from e


@contextmanager
def create_jax_safe_process_pool(
    max_workers: int,
    *,
    mp_context: mp.context.SpawnContext | None = None,
) -> Generator[ProcessPoolExecutor, None, None]:
    """JAX fork() 互換性を確保した ProcessPoolExecutor を作成する context manager.
    
    spawn コンテキストを使用して、各 worker プロセスが independent な
    Python インタプリタを起動します。CPU-only でも spawn を使用すべき理由：
    
    - JAX JIT コンパイル状態の独立性確保
    - import 時のキャッシュ・トレース状態の分離
    - fork() ベースの不確定な状態共有を完全に回避
    
    Parameters
    ----------
    max_workers : int
        ワーカープロセス数（1 以上）

    mp_context : mp.context.SpawnContext | None, optional
        マルチプロセッシングコンテキスト。
        デフォルト (None) では get_spawn_context() を自動取得。

    Yields
    ------
    ProcessPoolExecutor
        max_workers 個の spawn process を持つ executor
    """
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
