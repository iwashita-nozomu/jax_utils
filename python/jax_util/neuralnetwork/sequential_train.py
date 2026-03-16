"""カスタム学習（実験用）。

現状このモジュールは未実装です。
構文エラーが残ると import/静的解析/依存抽出などが壊れるため、
明示的に NotImplemented として残します。
"""

from __future__ import annotations

from typing import Callable, Tuple

import equinox as eqx
import jax

from ..base import Matrix, Scalar
from .neuralnetwork import NeuralNetwork, forward_with_cache
from .protocols import (
    Aux,
    Carry,
    Ctx,
    NeuralNetworkLayer,
    Params,
    PyTreeOptimizationProblem,
    PyTreeOptimizationState,
    SingleLayerBackprop,
    Static,
)


class PyTreeOptim(eqx.Module):
    objective: Callable[[Params], Scalar]


class GradientBackprop(eqx.Module):
    optstate: PyTreeOptimizationState

    def __call__(
        self,
        layer: NeuralNetworkLayer,
        cache: Tuple[Carry, Ctx],
        obj: PyTreeOptimizationProblem,
    ) -> Tuple["SingleLayerBackprop", NeuralNetworkLayer, PyTreeOptimizationProblem, Aux]:
        param, static, layer_optim = self.buildoptim(layer, obj, cache)
        new_param, new_state, aux = self.update(param, layer_optim)
        new_layer: NeuralNetworkLayer = eqx.combine(new_param, static)  # pyright: ignore

        _, ctx = cache

        def new_loss(_param: Params) -> Scalar:
            lower_model = eqx.combine(_param, static)
            return obj.objective(lower_model(cache[0], ctx).z)

        new_obj = PyTreeOptim(objective=new_loss)

        return GradientBackprop(optstate=new_state), new_layer, new_obj, aux

    def buildoptim(
        self,
        layer: NeuralNetworkLayer,
        obj: PyTreeOptimizationProblem,
        train_params: Tuple[Carry, Ctx],
    ) -> Tuple[Params, Static, PyTreeOptimizationProblem]:
        param, static = eqx.partition(layer, eqx.is_inexact_array)

        def layer_objective(param: Params) -> Scalar:
            new_layer: NeuralNetworkLayer = eqx.combine(param, static)
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
        new_param = layer_param - grads
        return new_param, self.optstate, None


def sequential_train_step(
    model: NeuralNetwork,
    trainers: Tuple[SingleLayerBackprop, ...],
    x: Matrix,
    optim: PyTreeOptimizationProblem,
) -> Tuple[NeuralNetwork, Tuple[SingleLayerBackprop, ...]]:
    _, carries, ctx = forward_with_cache(x, model)

    current_optim = optim
    new_layers = []
    new_trainers = []

    for layer_idx in reversed(range(len(model.layers))):
        layer = model.layers[layer_idx]
        trainer = trainers[layer_idx]

        new_trainer, layer, current_optim, _ = trainer(
            layer,
            (carries[layer_idx], ctx),
            current_optim,
        )
        new_layers.append(layer)
        new_trainers.append(new_trainer)

    return NeuralNetwork(
        layers=tuple(reversed(new_layers)),
        network_type=model.network_type,
        layer_sizes=model.layer_sizes,
    ), tuple(reversed(new_trainers))


__all__ = [
    "PyTreeOptim",
    "GradientBackprop",
    "sequential_train_step",
]
