from typing import Protocol, TypeAlias,Any, Tuple , Callable

import abc

from jaxtyping import PyTree, Array

from ..base import Matrix, Scalar


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


class OptimizeProblemPytree(Protocol):#パラメータ更新用の最適化問題
    objective: Callable[[Params], Scalar]
    # static: Static

    # variable_dim: int
    # def __call__(self, x: Params)-> Scalar: ...
    ...

class OptimizeProblemStatePytree(Protocol):
    x: Params
    ...


# RebuildState: TypeAlias = Tuple[Callable[[Vector], Params], Static]

class LayerUpdate(Protocol): #パラメータのベクトル化＋新しいTreeを作成
    def __call__(
            self,
            layer_param: Params,
            optim: OptimizeProblemPytree #
    ) -> Tuple[Params, OptimizeProblemStatePytree, Aux]: ...
    ...

class BuildLayerOptim(Protocol):# 当該レイヤーの最適化問題を構成する
    def __call__(
            self,
            layer: NeuralNetworkLayer,
            obj: OptimizeProblemPytree,
            train_params: Tuple[Carry,Ctx],
    ) -> Tuple[Params,Static, OptimizeProblemPytree]:
        ...
    ...

class SingleLayerBackprop(Protocol):
    # buildoptim: BuildLayerOptim
    # update: LayerUpdate
    optstate: OptimizeProblemStatePytree
    def __call__(
            self,
            layer: NeuralNetworkLayer,
            cache: Tuple[Carry, Ctx],
            obj: OptimizeProblemPytree,
    ) -> Tuple["SingleLayerBackprop",NeuralNetworkLayer, OptimizeProblemPytree,Aux]:
        ...
    def buildoptim(
            self,
            layer: NeuralNetworkLayer,
            obj: OptimizeProblemPytree,
            train_params: Tuple[Carry,Ctx],
    ) -> Tuple[Params,Static, OptimizeProblemPytree]:
        ...

    def update(
            self,
            layer_param: Params,
            optim: OptimizeProblemPytree #
    ) -> Tuple[Params, OptimizeProblemStatePytree, Aux]: 
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
    "OptimizeProblemPytree",
    "OptimizeProblemStatePytree",
]
