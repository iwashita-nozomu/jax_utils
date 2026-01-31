from _type_aliaces import *

import equinox as eqx
from typing import  TypeVar, cast,Literal


from jaxtyping import Array,jaxtyped
from typeguard import typechecked

_ArrayT = TypeVar("_ArrayT",bound=Array)

class LinOp(eqx.Module):
    """A linear operator that can be applied to vectors and batched vectors.

    Attributes:
        matvec: A function that applies the linear operator to a vector.
    """
    matvec: BLinearMap

    __batched__: Literal[True] = eqx.field(init=False,default=True)
    __linear__: Literal[True] = eqx.field(init=False,default=True)
    def __init__(self,matvec:LinearMap|BLinearMap):
        self.matvec = cast(BLinearMap, ensure_batch(matvec))
        setattr(self.matvec,"__batched__",True)
    def __call__(self,x:Batch[Vector]) -> Batch[Vector]:
        return self.matvec(x)
    
    # def __matmul__(self,other:"LinOp") -> "LinOp":
    #     return LinOp(lambda x: self.matvec(other.matvec(x)))
    
class DiagLinOp(LinOp):
    """A diagonal linear operator represented by a vector.

    Attributes:
        diag: A vector representing the diagonal elements of the operator.
    """

    def __init__(self,diag:Vector):
        self.diag = diag
        # @overload
        @jaxtyped(typechecker=typechecked)
        def matvec(x:Batch[Vector],/) -> Batch[Vector]:
            return self.diag * x
        super().__init__(matvec=cast(BLinearMap, matvec))
        # self.matvec = matvec


if __name__ == "__main__":

    import jax.numpy as jnp
    def f(v:Vector) -> Vector:
        return 2.0 * v
    
    batchf = ensure_batch(f)

    print(batchf(jnp.array([[1.0,2.0,3.0],[4.0,5.0,6.0]])))
