"""TaskContext ユーティリティ。"""

from __future__ import annotations

import os

from .protocols import TaskContext

__all__ = [
    "apply_environment_variables",
]


def apply_environment_variables(context: TaskContext) -> None:
    """`context["environment_variables"]` を `os.environ` に適用する。"""
    env_vars = context.get("environment_variables", {})
    if not isinstance(env_vars, dict):
        raise TypeError("context['environment_variables'] must be a dict.")
    for key, value in env_vars.items():
        os.environ[str(key)] = str(value)
