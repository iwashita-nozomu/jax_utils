from __future__ import annotations

from typing import Protocol, Tuple

from ..base import Matrix
from .neuralnetwork import NeuralNetwork
from .protocols import PyTreeOptimizationProblem, SingleLayerBackprop


class IncrementalTrainer(Protocol):
    """層ごとの trainer を束ねる実験用 protocol。"""

    layer_trainers: Tuple[SingleLayerBackprop, ...]

    def __call__(
        self,
        model: NeuralNetwork,
        x: Matrix,
        optim: PyTreeOptimizationProblem,
    ) -> tuple[NeuralNetwork, Tuple[SingleLayerBackprop, ...]]:
        ...


__all__ = [
    "SingleLayerBackprop",
    "IncrementalTrainer",
]
