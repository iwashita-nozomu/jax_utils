from typing import Any, Protocol, Tuple, TypeAlias

import abc

from jaxtyping import PyTree, Array

from ..base import (
    ConstrainedOptimizationProblem,
    ConstrainedOptimizationState,
    Matrix,
    OptimizationProblem,
    OptimizationState,
    Scalar,
    Vector,
)


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

class LossFn(Protocol):
    def __call__(self, params: Params, x: Matrix, /) -> Scalar: ...


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

class LayerUpdate(Protocol):
    def __call__(
            self,
            layer_param: Params,
            optim: PyTreeOptimizationProblem,
    ) -> Tuple[Params, PyTreeOptimizationState, Aux]: ...
    ...

class BuildLayerOptim(Protocol):
    def __call__(
            self,
            layer: NeuralNetworkLayer,
            obj: PyTreeOptimizationProblem,
            train_params: Tuple[Carry,Ctx],
    ) -> Tuple[Params,Static, PyTreeOptimizationProblem]:
        ...
    ...

class SingleLayerBackprop(Protocol):
    optstate: PyTreeOptimizationState
    def __call__(
            self,
            layer: NeuralNetworkLayer,
            cache: Tuple[Carry, Ctx],
            obj: PyTreeOptimizationProblem,
    ) -> Tuple["SingleLayerBackprop",NeuralNetworkLayer, PyTreeOptimizationProblem,Aux]:
        ...
    def buildoptim(
            self,
            layer: NeuralNetworkLayer,
            obj: PyTreeOptimizationProblem,
            train_params: Tuple[Carry,Ctx],
    ) -> Tuple[Params,Static, PyTreeOptimizationProblem]:
        ...

    def update(
            self,
            layer_param: Params,
            optim: PyTreeOptimizationProblem,
    ) -> Tuple[Params, PyTreeOptimizationState, Aux]:
        ...
    ...




__all__ = [
    "Params",
    "Static",
    "Ctx",
    "Carry",
    "NeuralNetworkLayer",
    "LossFn",
    "Aux",
    # "State",
    # "UpdateFunc",
    "LayerUpdate",
    "BuildLayerOptim",
    "SingleLayerBackprop",
    "BackpropState",
    "PyTreeOptimizationProblem",
    "ConstrainedPyTreeOptimizationProblem",
    "PyTreeOptimizationState",
    "ConstrainedPyTreeOptimizationState",
]
