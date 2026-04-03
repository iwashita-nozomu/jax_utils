"""TaskContext ユーティリティ。"""

from __future__ import annotations

import os
from .protocols import TaskContext

__all__ = [
    "apply_environment_variables",
]


def _cpu_affinity_from_context(context: TaskContext) -> tuple[int, ...]:
    runner_metadata = context.get("runner_metadata")
    if isinstance(runner_metadata, dict):
        cpu_affinity = runner_metadata.get("cpu_affinity")
        if isinstance(cpu_affinity, (list, tuple)):
            values: list[int] = []
            for cpu in cpu_affinity:
                if not isinstance(cpu, (int, str)):
                    raise TypeError("runner_metadata['cpu_affinity'] must contain ints.")
                values.append(int(cpu))
            return tuple(values)

    cpu_affinity = context.get("cpu_affinity")
    if isinstance(cpu_affinity, (list, tuple)):
        values = []
        for cpu in cpu_affinity:
            if not isinstance(cpu, (int, str)):
                raise TypeError("context['cpu_affinity'] must contain ints.")
            values.append(int(cpu))
        return tuple(values)
    return ()


def apply_environment_variables(context: TaskContext) -> None:
    """`TaskContext` の環境変数と process-local CPU affinity を適用する。"""
    env_vars = context.get("environment_variables", {})
    if not isinstance(env_vars, dict):
        raise TypeError("context['environment_variables'] must be a dict.")
    for key, value in env_vars.items():
        os.environ[str(key)] = str(value)

    cpu_affinity = _cpu_affinity_from_context(context)
    if cpu_affinity and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpu_affinity))
