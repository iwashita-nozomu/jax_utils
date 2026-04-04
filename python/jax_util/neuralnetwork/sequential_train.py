"""カスタム学習（実験用）。

実験中の逐次学習パスです。公開 API にはまだ含めませんが、
構文エラーや読みにくい命名が残ると import / 静的解析 / 依存抽出が壊れるため、
最低限の読みやすさと保守性は保ちます。
"""

from __future__ import annotations

import equinox as eqx
import jax
from typing import Callable, Tuple
from jaxtyping import Array

from ..base import Matrix, Scalar
from .layer_utils import Static
from .neuralnetwork import NeuralNetwork
from .protocols import (
    Aux,
    Carry,
    Ctx,
    NeuralNetworkLayer,
    Params,
    PyTreeOptimizationProblem,
    PyTreeOptimizationState,
    SingleLayerBackprop,
)


class PyTreeOptim(eqx.Module):
    objective: Callable[[Params], Scalar] = eqx.field(static=True)


class GradientBackprop(eqx.Module):
    optstate: PyTreeOptimizationState

    def __call__(
        self,
        layer: NeuralNetworkLayer,
        cache: Tuple[Carry, Ctx],
        obj: PyTreeOptimizationProblem,
    ) -> Tuple["SingleLayerBackprop", NeuralNetworkLayer, PyTreeOptimizationProblem, Aux]:
        param, static, layer_optim = self.buildoptim(layer, obj, cache)
        new_param, new_state, aux = self.update(
            param,
            layer_optim,
        )
        new_layer: NeuralNetworkLayer = eqx.combine(new_param, static)  # pyright: ignore[reportAssignmentType]
        input_carry, ctx = cache

        def new_loss(param: Params) -> Scalar:
            lower_layer: NeuralNetworkLayer = eqx.combine(param, static)  # pyright: ignore[reportAssignmentType]
            return obj.objective(lower_layer(input_carry, ctx).z)

        new_obj = PyTreeOptim(
            objective=new_loss,
        )

        return GradientBackprop(
            optstate=new_state,
        ), new_layer, new_obj, aux

    def buildoptim(
        self,
        layer: NeuralNetworkLayer,
        obj: PyTreeOptimizationProblem,
        train_params: Tuple[Carry, Ctx],
    ) -> Tuple[Params, Static, PyTreeOptimizationProblem]:
        param, static = eqx.partition(layer, eqx.is_inexact_array)

        def layer_objective(param: Params) -> Scalar:
            new_layer: NeuralNetworkLayer = eqx.combine(param, static)  # pyright: ignore[reportAssignmentType]
            carry, ctx = train_params
            new_carry = new_layer(carry, ctx)
            return obj.objective(new_carry.z)

        layer_optim = PyTreeOptim(objective=layer_objective)
        return param, static, layer_optim

    def update(
        self,
        layer_param: Params,
        optim: PyTreeOptimizationProblem,
    ) -> Tuple[Params, PyTreeOptimizationState, Aux]:
        grads = jax.grad(optim.objective)(layer_param)

        def apply_gradient(param: Array, grad: Array) -> Array:
            return param - grad

        new_param = jax.tree_util.tree_map(apply_gradient, layer_param, grads)

        return new_param, self.optstate, None


def sequential_train_step(
    model: NeuralNetwork,
    trainers: Tuple[SingleLayerBackprop, ...],
    x: Matrix,
    optim: PyTreeOptimizationProblem,
) -> Tuple[NeuralNetwork, Tuple[SingleLayerBackprop, ...]]:
    _ = model
    _ = trainers
    _ = x
    _ = optim
    raise NotImplementedError("experimental sequential training path is not implemented")


class GradientTrainer(eqx.Module):
    """層ごとの trainer を束ねる薄い実験用ラッパー。"""

    layer_trainers: Tuple[SingleLayerBackprop, ...]

    def __call__(
        self,
        model: NeuralNetwork,
        x: Matrix,
        optim: PyTreeOptimizationProblem,
    ) -> tuple[NeuralNetwork, Tuple[SingleLayerBackprop, ...]]:
        return sequential_train_step(
            model=model,
            trainers=self.layer_trainers,
            x=x,
            optim=optim,
        )


__all__ = [
    "PyTreeOptim",
    "GradientBackprop",
    "GradientTrainer",
    "sequential_train_step",
]
