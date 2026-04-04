"""Dynamic-programming inspired layer-wise training scaffolding.

旧 nn worktree に残っていた着想を import-safe な形で保存する。
実装本体はまだ未着手なので、呼び出し時は明示的に失敗させる。
"""

from __future__ import annotations

from typing import Any

import equinox as eqx

from ..base import OptimizationProblem
from ..functional import Function
from .neuralnetwork import NeuralNetwork
from .protocols import Aux, Params, PyTreeOptimizationProblem, Static


class SingleLayerDynamicProgramming(eqx.Module):
    """上位層へ再帰する optimizer の実験用プレースホルダ。"""

    upper_layer_optimizer: "SingleLayerDynamicProgramming | None" = None
    layer_static: Static | None = eqx.field(default=None, static=True)

    def __call__(
        self,
        layer_param: Params,
        lower_model: Function,
    ) -> tuple[Params, PyTreeOptimizationProblem, Aux]:
        _ = layer_param
        _ = lower_model
        raise NotImplementedError(
            "dynamic-programming based layer-wise training is not implemented yet"
        )


def initialize_dp_trainer(
    model: NeuralNetwork,
) -> tuple[SingleLayerDynamicProgramming, ...]:
    _ = model
    raise NotImplementedError(
        "dynamic-programming based layer-wise trainer initialization is not implemented yet"
    )


def initiakize_dp_trainer(
    model: NeuralNetwork,
) -> tuple[SingleLayerDynamicProgramming, ...]:
    """旧メモの綴りを残した互換 alias。"""

    return initialize_dp_trainer(model)


def train(
    model: NeuralNetwork,
    optim: OptimizationProblem[NeuralNetwork],
    context: tuple[Any, ...],
) -> tuple[NeuralNetwork, Aux]:
    _ = model
    _ = optim
    _ = context
    raise NotImplementedError(
        "dynamic-programming based neural network training is not implemented yet"
    )


__all__ = [
    "SingleLayerDynamicProgramming",
    "initialize_dp_trainer",
    "initiakize_dp_trainer",
    "train",
]
