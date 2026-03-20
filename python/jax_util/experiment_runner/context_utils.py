# -*- coding: utf-8 -*-
"""
TaskContext ユーティリティ。

environment_variables を os.environ に適用するヘルパー関数など、
ワーカー実装で頻繁に使用される便利関数を提供。
"""

from __future__ import annotations

import os
from .protocols import TaskContext


def apply_environment_variables(context: TaskContext) -> None:
    """
    context["environment_variables"] を os.environ に適用する。
    
    resource_scheduler が生成した環境変数情報を、
    実際の os.environ に設定することで、
    ワーカープロセス内の JAX や GPU 処理が正しく GPU ID を参照できるようになる。
    
    Parameters
    ----------
    context : TaskContext
        ランナーから渡された TaskContext。
        context["environment_variables"] が dict として格納されていることを前提。
    
    Example
    -------
    >>> def __call__(self, case, context: TaskContext) -> int:
    ...     # environment_variables を os.environ に適用
    ...     apply_environment_variables(context)
    ...     # 以降、os.environ["CUDA_VISIBLE_DEVICES"] などが正しく設定される
    ...     import jax.numpy as jnp
    ...     # JAX がこのプロセスに割り当てられた GPU を正しく認識
    """
    env_vars = context.get("environment_variables", {})
    for key, value in env_vars.items():
        os.environ[key] = value
