from typing import Any, Protocol, Tuple, TypeAlias

from jaxtyping import PyTree, Array

from ..base import (
    OptimizationProblem,
)

from .protocols import (
    Params,
    Static,
    Ctx,
    Carry,
    NeuralNetworkLayer,
    Aux,
    BackpropState,
    PyTreeOptimizationProblem,
    PyTreeOptimizationState,
)

from .neuralnetwork import NeuralNetwork





class SingleLayerBackprop(Protocol):
    optstate: PyTreeOptimizationState
    def __call__(
            self,
            layer: NeuralNetworkLayer,
            cache: Tuple[Carry, Ctx],
            optim: OptimizationProblem[NeuralNetwork],
            backprop_state: BackpropState,
    ) -> Tuple["SingleLayerBackprop",NeuralNetworkLayer, PyTreeOptimizationProblem,Aux]:
        ...

    # そのレイヤーのパラメータに関する最適化問題に変換する
    def buildoptim(
            self,
            layer: NeuralNetworkLayer,
            optim: OptimizationProblem[NeuralNetworkLayer],
            train_params: Tuple[Carry,Ctx],
    ) -> Tuple[Params,Static, PyTreeOptimizationProblem]:
        ...

    # そのままやるなり、フラット化するなりして更新
    def update(
            self,
            layer_param: Params,
            optim: PyTreeOptimizationProblem,
    ) -> Tuple[Params, PyTreeOptimizationState, Aux]:
        ...
    ...

class IncrementalTrainer(Protocol):
    layer_trainers: Tuple[SingleLayerBackprop, ...]

    def __call__(
            self,
            model: NeuralNetwork,
            optim: OptimizationProblem[NeuralNetwork], 
            context: Tuple[Any,...],#積分器、学習率スケジューラーなどの追加情報
            )-> Tuple[NeuralNetwork, Aux]:
        ...
    ...


__all__ = [
    "SingleLayerBackprop",
    "IncrementalTrainer",
]
