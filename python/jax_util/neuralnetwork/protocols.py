from typing import Any,  Protocol, Tuple, TypeAlias

import abc

import equinox as eqx
from jaxtyping import Array, PyTree

from ..base import Matrix, Scalar, Vector,OptimizationProblem,ConstraintedOptimizationProblem

from ..functional import FunctionalOptimizationProblem 

from ..optimizers.protocols import VectorOptimizationState

Params: TypeAlias = PyTree[Array]
Static: TypeAlias = PyTree[Any]


class Ctx(Protocol):
    """固定状態を表します。"""


class Carry(Protocol):
    """更新状態を表します。"""

    z: Matrix


class NeuralNetworkLayer(eqx.Module):
    @abc.abstractmethod
    def __call__(self, carry: Carry, ctx: Ctx, /) -> Carry: ...


class LossFn(Protocol):
    def __call__(self, params: Params, x: Matrix, /) -> Scalar: ...


class Aux(Protocol):
    ...


class BackpropState(Protocol):
    """上位レイヤーからの伝搬情報を格納する。"""


class PyTreeOptimizationProblem(OptimizationProblem[Params], Protocol):
    """パラメータ更新用の PyTree 最適化問題。"""
    ... #note:: 目的関数はパラメータ空間上の関数であることに注意してください。

class ConstrainedPyTreeOptimizationProblem(ConstraintedOptimizationProblem[Params,Vector,Vector],
                                                  PyTreeOptimizationProblem,
                                                    Protocol):
    ... #note:: 目的関数,制約関数はパラメータ空間上の関数であることに注意してください。


class PyTreeOptimizationState(Protocol):
    x: Params


class ConstrainedPyTreeOptimizationState(PyTreeOptimizationState, Protocol):
    lam_eq: Vector
    lam_ineq: Vector
    slack: Vector


class LayerUpdate(Protocol):
    """パラメータを更新して新しい state を返す。"""

    def __call__(
        self,
        layer_param: Params,
        optim: PyTreeOptimizationProblem,
    ) -> Tuple[Params, PyTreeOptimizationState, Aux]: ...


class BuildLayerOptim(Protocol):
    """レイヤーごとの最適化問題を構成する。"""

    def __call__(
        self,
        layer: NeuralNetworkLayer,
        obj: PyTreeOptimizationProblem,
        train_params: Tuple[Carry, Ctx],
    ) -> Tuple[Params, Static, PyTreeOptimizationProblem]: ...


class SingleLayerBackprop(Protocol):
   
    pytreeoptimstate: PyTreeOptimizationState|VectorOptimizationState

    def __call__(
        self,
        layer: NeuralNetworkLayer,
        cache: Tuple[Carry, Ctx],
        optim: FunctionalOptimizationProblem,
        backprop_state: BackpropState,
    ) -> Tuple["SingleLayerBackprop", NeuralNetworkLayer, BackpropState, Aux]:
        # レイヤーの分解と最適化問題の構築
        # レイヤーの再構築と新しい最適化問題の構築
        # 階層への伝搬のための情報の構築
        # 更新したパラメータでレイヤーを再構築
        # 必要に応じてstateを更新して、返す
        ...

    def buildoptim(
        self,
        layer: NeuralNetworkLayer,
        optim: FunctionalOptimizationProblem,
        backprop_state: BackpropState,
        train_params: Tuple[Carry, Ctx],
    ) -> Tuple[Params, Static, PyTreeOptimizationProblem]:
        # レイヤーのパラメータと静的部分の分解
        # レイヤーのパラメータを引数とする最適化問題の構築
        ...

    def update(
        self,
        layer_param: Params,
        optim: PyTreeOptimizationProblem,
    ) -> Tuple[Params, PyTreeOptimizationState, Aux]:
        # 必要に応じて、ベクトル化とかやって最適化問題を解くなりして、レイヤーのパラメータを更新する。
        ...



__all__ = [
    "Params",
    "Static",
    "Ctx",
    "Carry",
    "NeuralNetworkLayer",
    "LossFn",
    "Aux",
    "BackpropState",
    "LayerUpdate",
    "BuildLayerOptim",
    "SingleLayerBackprop",
    "PyTreeOptimizationProblem",
    "ConstrainedPyTreeOptimizationProblem",
    "PyTreeOptimizationState",
    "ConstrainedPyTreeOptimizationState",
]
