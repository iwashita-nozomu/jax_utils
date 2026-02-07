"""カスタム学習（実験用）。

現状このモジュールは未実装です。
構文エラーが残ると import/静的解析/依存抽出などが壊れるため、
明示的に NotImplemented として残します。
"""

from __future__ import annotations
from ..base import *
from ..optimizers import *
from typing import Tuple,Callable

import equinox as eqx

from optimizers.protocols import *

from .neuralnetwork import *
from .protocols import *
from .layer_utils import *
# from jax import flatten_util

class PytrreeOptim(eqx.Module):
    objective: Callable[[Params], Scalar]
    # static: Static
    

class NormalBP(eqx.Module):
    optstate: OptimizeProblemStatePytree
    def __call__(
            self,
            layer: NeuralNetworkLayer,
            cache: Tuple[Carry, Ctx],
            obj: OptimizeProblem,
            state: OptimizeProblemState,
    ) -> Tuple[NeuralNetworkLayer, BackpropState, OptimizeProblemState, OptimizeProblem,Aux]:
        
        param, static, layer_optim = self.buildoptim(
            layer,
            obj,
            cache,
        )
        new_param, new_state, aux = self.update(
            param,
            # cache,
            state,
            layer_optim,
        )
        new_layer: NeuralNetworkLayer = eqx.combine(new_param, static) #pyright: ignore

        _, ctx = cache

        #下段の目的関数を定義する
        def new_loss(param: Params) -> Scalar:
            return obj.objective(new_layer(param, ctx).z)
        
        new_obj = PytrreeOptim(
            objective=new_loss,
            # variable_dim=new_layer_param.shape[0],
        )

        return new_layer, new_obj, new_state, obj, aux

    def buildoptim(
            self,
            layer: NeuralNetworkLayer,
            obj: OptimizeProblem,# R^(n^k) * b -> R 
            train_params: Tuple[Carry,Ctx],
    ) -> Tuple[Params,Static,OptimizeProblemPytree]:
        # layer_param, rebuild_static = module_to_vector(layer)
        param, static = eqx.partition(layer, eqx.is_inexact_array)

        def layer_objective(param: Params) -> Scalar:
            # new_layer: NeuralNetworkLayer = vector_to_module(param, rebuild_static)
            new_layer: NeuralNetworkLayer = eqx.combine(param, static)
            carry, ctx = train_params
            new_carry = new_layer(carry, ctx)
            loss = obj.objective(new_carry.z)
            return loss
        
        layer_optim = PytrreeOptim(
            objective=layer_objective,
            # variable_dim=param.shape[0],
        )
        return param,static, layer_optim

    def update(
            self,
            layer_param: Params,
            # cache: Tuple[Carry,Ctx],
            state: OptimizeProblemState,
            optim: OptimizeProblemPytree) -> Tuple[Params, OptimizeProblemState, Aux]:
        
        grads = jax.grad(optim.objective)(layer_param)
        new_param = layer_param - grads

        return new_param, state, None


def sequential_train_step(
        model: NeuralNetwork,
        trainers:Tuple[SingleLayerBackprop,...],
        x: Matrix,
        optim: OptimizeProblemPytree,
) -> Tuple[NeuralNetwork,Tuple[SingleLayerBackprop,...]]:

    z,carrys, ctx = forward_with_cache(
        x,
        model,
    )

    _optim = optim

    new_layers = []
    new_trainers = []

    for layer_idx in reversed(range(len(model.layers))):
        layer = model.layers[layer_idx]
        trainer = trainers[layer_idx]

        new_triner,layer, _optim, _, = trainer(
            layer,
            (carrys[layer_idx], ctx),
            _optim,
        )
        new_layers.append(layer)
        new_trainers.append(new_triner)

    return NeuralNetwork(
        layers=tuple(reversed(new_layers)),
        network_type=model.network_type,
        layer_sizes=model.layer_sizes,
    ), tuple(reversed(new_trainers))