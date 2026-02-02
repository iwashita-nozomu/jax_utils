from __future__ import annotations

from typing import (
    Protocol,
    runtime_checkable,
    TypeAlias,
    overload,
    Tuple,
)

from jaxtyping import Array, Float,  Bool, Int

Scalar: TypeAlias = Float[Array, ""]
Vector: TypeAlias = Float[Array, "n"]
Matrix: TypeAlias = Float[Array, "m n"]

Boolean : TypeAlias = Bool[Array, ""]
Integer : TypeAlias = Int[Array, ""]



@runtime_checkable
class Operator(Protocol):
    #__isoperator__: Literal[True]


    #適用
    @overload
    def __call__(self,other:Matrix)-> Matrix: ...
    @overload
    def __call__(self,other:Operator)-> Operator: ...


    #作用素の明示合成
    @overload
    def __mul__(self,other:Scalar)-> "Operator": ...
    @overload
    def __mul__(self,other:"Operator")-> "Operator": ...


@runtime_checkable
class LinearOperator(Protocol):
    # __islinearoperator__: Literal[True]
    @property
    def shape(self) -> Tuple[int,...]: ...
    @overload
    def __matmul__(self,other:Matrix,/)-> Matrix: ...
    @overload
    def __matmul__(self,other:LinearOperator,/)-> LinearOperator: ...

    #作用素の明示合成
    @overload
    def __mul__(self,other:Scalar,/)-> "LinearOperator": ...
    @overload
    def __mul__(self,other:"LinearOperator",/)-> "LinearOperator": ...
    
    @overload
    def __rmul__(self,other:Scalar,/)-> "LinearOperator":  ...
    @overload
    def __rmul__(self,other:"LinearOperator",/)-> "LinearOperator": ...
    ...

__all__ = [
    "Operator",
    "LinearOperator",
    "Scalar",
    "Vector",
    "Matrix",
    "Boolean",
    "Integer",
]

if __name__ == "__main__":

    import jax

    A :LinearOperator= jax.numpy.array([[1.,2.],[3.,4.]]) 