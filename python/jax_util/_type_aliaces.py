from __future__ import annotations
from typing import TypeVar,Protocol,TypeAlias,runtime_checkable,cast,Annotated,Literal,overload
from jaxtyping import Array,Shaped
# from typeguard import typechecked

import jax

#射
B = TypeVar("B",covariant=True)
C = TypeVar("C",contravariant=True)
D = TypeVar("D",covariant=True)
class Hom(Protocol[C,D]):
    def __call__(self,x:C ,/,) -> D: ...
    # def __matmul__(self,other:"Hom[B,C]") -> "Hom[B,D]": ...#関数の合成 self @ other

T = TypeVar("T")

# t : TypeAlias = Shaped [T,"*batch ..."]

class IsBatched:...

Batch:TypeAlias = Annotated[T, IsBatched]

# @runtime_checkable
# class Batch(t[T],Protocol):
#     __batched__: bool
#     ...

#BHom = Annotated[Hom[C,D], IsBatched]

@runtime_checkable
class BHom(Hom[Batch[C],Batch[D]],Protocol[C,D]):
    __batched__: Literal[True]
    ...

#自己写像
A = TypeVar("A")
@runtime_checkable
class Endo(Hom[A,A],Protocol[A]):
    ...

# _ArrayT = TypeVar("_ArrayT",bound=Array)
Scalar : TypeAlias = Shaped [Array,""]
Vector : TypeAlias = Shaped [Array,"n"]
Matrix : TypeAlias = Shaped [Array,"m n"]

@runtime_checkable
class Map(Hom[Vector,Shaped [Array,"m"]],Protocol):
    ...

@runtime_checkable
class BMap(BHom[Shaped[Array,"n"],Shaped[Array,"m"]],Map,Protocol):
    ...   

class LinearMap(Map,Protocol):
    __linear__ : Literal[True]
    ...

class BLinearMap(BMap,LinearMap,Protocol):
    ...

_T = TypeVar("_T")
_U = TypeVar("_U")

@overload
def ensure_batch(f:BHom[_T,_U]) -> BHom[_T,_U]: ...
@overload
def ensure_batch(f:Hom[_T,_U]) -> BHom[_T,_U]: ...

def ensure_batch(f:Hom[_T,_U]) -> BHom[_T,_U]:
    """Ensure that a function is batched.

    If the function is already batched, it is returned as is.
    If not, it is wrapped to handle batched inputs.

    Args:
        f: A function that maps from Shaped[Array,"n"] to Shaped[Array,"m"].

    Returns:
        A function that maps from Shaped[Array,"b n"] to Shaped[Array,"b m"].
    """
    if getattr(f,"__batched__",False):
        return cast(BHom[_T,_U],f)


    def batched_f(v:Batch[_T]) -> Batch[_U]:
        return  jax.vmap(f)(v)
    
    setattr(batched_f,"__batched__",True)
    return cast(BHom[_T,_U],batched_f)


__all__ = [
    # "ArrayT",
    "Hom",
    "BHom",
    "Endo",
    "Scalar",
    "Vector",
    "Matrix",
    "Map",
    "BMap",
    "Batch",
    "ensure_batch",
    "LinearMap",
    "BLinearMap",
]


if __name__ == "__main__":
    m = 10

    from jax import numpy as jnp
    # NOTE:
    # インデクシング（x[..., :m]）の結果は静的型解析上「Array」になりやすく、
    # 型変数 ArrayT（=具体的な Array サブタイプ）を厳密に保てないため型エラーになります。
    # そのため、このサンプルでは入出力の型を明示的に Array に揃えて、
    # 静的型解析と実装の挙動が一致するように単純化します。
    def f(x: Batch[Array]) -> Batch[Array]:
        return x[..., :m]
    
    setattr(f,"__batched__",True)

    h = cast(BMap, ensure_batch(f))

    # NOTE:
    # BMap は「バッチされた入出力が同じ型である写像」を表す Protocol です。
    # 型引数には TypeVar（ArrayT）ではなく、具体的な型（例: jaxtyping.Array）を指定します。
    g = cast(BMap, f)

    v :Vector = jnp.zeros((m,m,m),)