from typing import Any, Protocol, TypeAlias

import abc

from jaxtyping import PyTree, Array

from ..base import (
    ConstrainedOptimizationProblem,
    ConstrainedOptimizationState,
    Matrix,
    OptimizationProblem,
    OptimizationState,
    # Scalar,
    Vector,
)

# from ..functional import Function


import equinox as eqx

Params: TypeAlias = PyTree[Array]
Static: TypeAlias = PyTree[Any]

class Ctx(Protocol):
    """固定状態を表します。"""
    ...

class Carry(Protocol):
    """更新状態を表します。"""
    z: Matrix
    ...

# Module = TypeVar("Module", bound=eqx.Module)

class NeuralNetworkLayer(eqx.Module):
    @abc.abstractmethod
    def __call__(self, carry: Carry, ctx: Ctx, /) -> Carry: ...


class Aux(Protocol):
    ...
# class State(Protocol):
#     ...

class BackpropState(Protocol):#上位レイヤーからの伝搬情報を格納する
    ...


class PyTreeOptimizationProblem(OptimizationProblem[Params], Protocol):
    ...


class ConstrainedPyTreeOptimizationProblem(
    ConstrainedOptimizationProblem[Params, Vector, Vector],
    PyTreeOptimizationProblem,
    Protocol,
):
    ...


class PyTreeOptimizationState(OptimizationState[Params], Protocol):
    ...


class ConstrainedPyTreeOptimizationState(
    ConstrainedOptimizationState[Params, Vector],
    PyTreeOptimizationState,
    Protocol,
):
    ...



__all__ = [
    "Params",
    "Static",
    "Ctx",
    "Carry",
    "NeuralNetworkLayer",
    "Aux",
    # "State",
    # "UpdateFunc",
    "BackpropState",
    "PyTreeOptimizationProblem",
    "ConstrainedPyTreeOptimizationProblem",
    "PyTreeOptimizationState",
    "ConstrainedPyTreeOptimizationState",
]
