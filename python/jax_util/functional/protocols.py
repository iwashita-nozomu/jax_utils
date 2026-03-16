from __future__ import annotations

from typing import Callable, Protocol

import equinox as eqx

from ..base import Scalar, Vector


class Function(Protocol):
    def __call__(self, x: Vector, /) -> Vector: ...

    def __matmul__(self, other: "Function") -> "Function": ...

    def __mul__(self, other: "Function") -> "Function": ...


class Func(eqx.Module):
    fn: Callable[[Vector], Vector]

    def __init__(self, fn: Callable[[Vector], Vector]):
        self.fn = fn

    def __call__(self, x: Vector, /) -> Vector:
        return self.fn(x)

    def __matmul__(self, other: "Func") -> "Func":
        def matmul_fn(x: Vector):
            return self(other(x))

        return Func(fn=matmul_fn)

    def __mul__(self, other: "Func") -> "Func":
        def mul_fn(x: Vector):
            return self(x) * other(x)

        return Func(fn=mul_fn)


class Functional(Protocol):
    def __call__(self, f: Function, /) -> Scalar: ...


class Integrator(Protocol):
    def __call__(self, f: Function, /) -> Vector: ...


__all__ = [
    "Integrator",
    "Function",
    "Func",
    "Functional",
]
