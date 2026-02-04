from typing import Protocol, TypeAlias

from jaxtyping import PyTree, Array
from ..base import *

Params: TypeAlias = PyTree[Array]

class Ctx(Protocol):
    """固定状態を表します。"""
    ...

class Carry(Protocol):
    """更新状態を表します。"""
    z: Matrix
    ...

class NeuralNetworkLayer(Protocol):
    def __call__(self, carry: Carry, ctx: Ctx, /) -> Carry: ...


class LossFn(Protocol):
    def __call__(self, params: Params, batch: Matrix, /) -> Scalar: ...


__all__ = [
    "Params",
    "Ctx",
    "Carry",
    "NeuralNetworkLayer",
    "LossFn",
]