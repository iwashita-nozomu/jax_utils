from __future__ import annotations

from typing import (
    # Annotated,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
    Generic,
    # overload,
    # # Union,
    ClassVar,
    # Callable
)

from jaxtyping import Array, Shaped
# from typeguard import typechecked

T = TypeVar("T",covariant=True)
# Batch=Shaped[T, "batch ..."]# バッチ次元を持つ型のエイリアス
# X:TypeAlias = Batch[Array]
# @runtime_checkable
# class PBatched(Generic[T],Protocol):
#     __batched__: ClassVar[Literal[True]]
# @runtime_checkable
# class PBatched(Protocol[T]):...
    # __batched__: ClassVar[Literal[True]]
# 射
# Protocol の関数型として扱うため、引数側は「反変」、戻り値側は「共変」に設定します。
# これにより、型チェック（pyright など）が要求する分散ルールに一致します。
C = TypeVar("C", contravariant=True)
D = TypeVar("D", covariant=True)

# PHom: TypeAlias = Callable[[C], D]

@runtime_checkable
class PHom(Protocol[C, D]):
    def __call__(self, x: C, /) -> D: ...

@runtime_checkable
class IsBatchedItem( Protocol):
    __batcheditem__: ClassVar[Literal[True]]#axis = 0 に対応するバッチ次元を持つことを示す属性

@runtime_checkable
class IsBatchedHom(Protocol):
    __batchedhom__: ClassVar[Literal[True]]

BatchC = TypeVar("BatchC", bound=IsBatchedItem,contravariant=True)
BatchD = TypeVar("BatchD", bound=IsBatchedItem,covariant=True)

@runtime_checkable
class PBatchedHom(PHom[BatchC, BatchD], IsBatchedHom, Protocol[BatchC, BatchD]):
    ...

# Batch: TypeAlias = Annotated[T, PBatched]

# 自己写像
A = TypeVar("A")
# @runtime_checkable
class PEndo(PHom[A, A], Protocol[A]):
    ...

# _ArrayT = TypeVar("_ArrayT",bound=Array)
Scalar: TypeAlias = Shaped[Array, ""]
Vector: TypeAlias = Shaped[Array, "n"]
Matrix: TypeAlias = Shaped[Array, "m n"]

# @runtime_checkable
class PMap(PHom[Shaped[Array, "n"], Shaped[Array, "m"]], Protocol):
    ...

class PLinear(Protocol):
    __linear__: ClassVar[Literal[True]]
    ...

# _BVec: TypeAlias = Batch[Vector]

# class PLinOp(PBatchedHom[Vector, Vector], PLinear, Protocol):
#     __synthesizable__: ClassVar[Literal[True]] # 合成可能であることを示す属性
#     def __matmul__(self,other:PBatched[Vector]) -> PBatched[Vector]: ...



__all__ = [
    "PHom",
    "PEndo",
    "Scalar",
    "Vector",
    "Matrix",
    "PMap",
    # "PBatched",
    "PLinear",
    # "PLinOp",
    "PBatchedHom",
    "IsBatchedItem",
    "IsBatchedHom",
]