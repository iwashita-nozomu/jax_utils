from __future__ import annotations

from typing import (
    Protocol,
    runtime_checkable,
    TypeAlias,
    TypeVar,
    overload,
    Tuple,
    Any,
    Dict,
    Callable,
    # Generic,
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


    def __call__(self,other:Matrix)-> Matrix: ...

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
    
    def __add__(self,other:"LinearOperator",/)-> "LinearOperator": ...
    ...


class SolverLike(Protocol):
    def __call__(self, *args: Any,**kwargs: Any) -> Tuple[Vector, Any, Dict[str,Any]]: ...
    ...

class ScalarFn(Protocol):
    def __call__(self,x:Vector, /) -> Scalar: ...
    ...

class VectorFn(Protocol):
    def __call__(self,x:Vector, /) -> Vector: ...
    ...


T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
@runtime_checkable
class OptimizationProblem(Protocol[T]):
    objective: Callable[[T], Scalar]
    ...

@runtime_checkable
class ConstraintedOptimizationProblem(OptimizationProblem[T], Protocol[T,U,V]):
    constraint_eq: Callable[[T], U]
    constraint_ineq: Callable[[T], V]
    ...

class OptimizationState(Protocol[T]):
    x: T
    ...

W = TypeVar("W") #双対空間

class ConstrainedOptimizationState(OptimizationState[T], Protocol[T,W]):
    lam_eq: W
    lam_ineq: W
    slack: T
    ...

__all__ = [
    "Operator",
    "LinearOperator",
    "Scalar",
    "Vector",
    "Matrix",
    "Boolean",
    "Integer",
    "SolverLike",
    "ScalarFn",
    "VectorFn",
    "OptimizationProblem",
    "ConstraintedOptimizationProblem",
    "OptimizationState",
    "ConstrainedOptimizationState"
]