from __future__ import annotations

from ..base import Vector
from .protocols import Function, Integrator


# 責務: 被積分関数を指定した積分器へ委譲して積分値を返す。
def integrate(f: Function, integrator: Integrator, /) -> Vector:
    return integrator.integrate(f)


__all__ = [
    "integrate",
]
