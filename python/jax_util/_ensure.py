from __future__ import annotations

from typing import TypeVar ,cast,Any, Callable

from jax_typing.base_protocol import *

import jax

_T = TypeVar("_T")
_U = TypeVar("_U")

# _BatchT = TypeVar("_BatchT", bound=IsBatchedItem)
# _BatchU = TypeVar("_BatchU", bound=IsBatchedItem)

def ensure_batch_fn(f: PHom[_T, _U]) -> PBatchedHom[IsBatchedItem, IsBatchedItem]:#バッチ入力に対応していることを保証する
    """Ensure that a function is batched.

    If the function is already batched, it is returned as is.
    If not, it is wrapped to handle batched inputs.

    Args:
        f: A function that maps from Shaped[Array,"n"] to Shaped[Array,"m"].

    Returns:
        A function that maps from Shaped[Array,"b n"] to Shaped[Array,"b m"].
    """

    if isinstance(f, PBatchedHom):
        return f

    # if getattr(f, "__batchedfn__", False):
    #     return cast(PBatchedHom[IsBatchedItem, IsBatchedItem], f)

    # def vmap_f(v: PBatched[T],/) -> PBatched[U]:
    #     ret = jax.vmap(f)(v)
    #     setattr(ret, "__batched__", True)
    #     return ret
    print("wrapping with vmap:",f)
    vmap_f: Callable[[Any], Any] = lambda v : jax.vmap(f)(v)
    setattr(vmap_f, "__batchedfn__", True)

    return cast(PBatchedHom[IsBatchedItem, IsBatchedItem], vmap_f)

def ensure_batched_item(x: object) -> IsBatchedItem:
    """Ensure that an object is a batched item.

    If the object is already a batched item, it is returned as is.
    If not, it is wrapped to handle batched inputs.

    Args:
        x: An object.
    Returns:
        An object that is a batched item.
    """
    if isinstance(x, IsBatchedItem):
        return x
    # setattr(x, "__batcheditem__", True)
    # return cast(IsBatchedItem, x)

    setattr(x, "__batcheditem__", True)
    return cast(IsBatchedItem, x)
    


if __name__ == "__main__":
    from typing import Callable
    def f(x: int) -> int:
        return x + 1
    

    g :Callable[[int], int] = f
    h :PHom[int,int] = f

    # i : PHom[PBatched[int], PBatched[int]]=f
    
    class MyBatchedInt(int, IsBatchedItem):
        def __init__(self, value: int):
            int.__init__(value)
            setattr(self, "__batcheditem__", True)

    j : PBatchedHom[IsBatchedItem, IsBatchedItem]=ensure_batch_fn(f)
    a = [10,20,30]
    # setattr(a,"__batcheditem__", True)
    # a = cast(IsBatchedItem, a)
    # b = j(a)
    # print(b)

    import jax.numpy as jnp
    a = jnp.array([10,20,30])
    a = ensure_batched_item(a)
    b = j(a)
    print(b)