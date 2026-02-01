from __future__ import annotations

from typing import Literal, Protocol

import equinox as eqx
from jaxtyping import jaxtyped
from typeguard import typechecked

from jax_typing.mixin import *
from jax_typing.base_protocol import *
from jax_typing.composite_protocol import LinOpProtocol

from _ensure import *

class LinOp(eqx.Module,LinearMixin,BatchedHomMixin):
    """A linear operator that can be applied to vectors and batched vectors.

    Attributes:
        matvec: A function that applies the linear operator to a vector.
    """
    matvec: PBatchedHom[IsBatchedItem, IsBatchedItem]

    def __init__(self, matvec: LinOpProtocol):
        self.matvec = ensure_batch_fn(matvec)

    def __call__(self,x:IsBatchedItem,/)->IsBatchedItem:
        return self.matvec(x)
    
    def __matmul__(self, x:IsBatchedItem, /)->IsBatchedItem:
        return self.__call__(x)

    
class DiagLinOp(LinOp):
    """A diagonal linear operator represented by a vector.

    Attributes:
        diag: A vector representing the diagonal elements of the operator.
    """

    def __init__(self, diag: Vector):
        self.diag = diag
        # @overload
        @jaxtyped(typechecker=typechecked)
        def matvec(x: Batch[Vector], /) -> Batch[Vector]:
            return self.diag * x
        
        setattr(matvec, "__batched__", True)
        setattr(matvec, "__linear__", True)
        super().__init__(matvec)
        # self.matvec = matvec


def DiagOp(diag: Vector) -> DiagLinOp:
    """対角演算子を明示的に生成するユーティリティ。"""
    return DiagLinOp(diag)

__all__ = [
    "LinOp",
    "DiagLinOp",
    "DiagOp",
]

if __name__ == "__main__":

    import jax.numpy as jnp
    def f(v:Vector) -> Vector:
        return 2.0 * v
    

    LinOp_f:LinOp = LinOp(f)

    
    print(LinOp_f(jnp.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])))