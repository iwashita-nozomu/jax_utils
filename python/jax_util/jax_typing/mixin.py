from __future__ import annotations

from base_protocol import*

from typing import (TypeAlias, 
                    Annotated, 
                    TypeVar, 
                    Protocol, 
                    runtime_checkable, 
                    Literal,
                    Generic,
                    )

import equinox as eqx

A = TypeVar("A")
B = TypeVar("B")


class BatchedItemMixin:#オブジェクトをバッチで持つ
    __batcheditem__: bool = eqx.static_field(default=True)

class BatchedHomMixin:#バッチ入力に対応する射
    __batchedhom__: bool = eqx.static_field(default=True)

class LinearMixin:
    __linear__: bool = eqx.static_field(default=True)

__all__ = [
    "BatchedItemMixin",
    "BatchedHomMixin",
    "LinearMixin",
]

if __name__ == "__main__":
    m = 10

    from jax import numpy as jnp
    from typing import Callable, Any

    # from jaxtyping import Array
    # NOTE:
    # インデクシング（x[..., :m]）の結果は静的型解析上「Array」になりやすく、
    # 型変数 ArrayT（=具体的な Array サブタイプ）を厳密に保てないため型エラーになります。
    # そのため、このサンプルでは入出力の型を明示的に Vector に揃えて、
    # 静的型解析と実装の挙動が一致するように単純化します。
    def f(x: Vector|Batch[Vector]) -> Vector|Batch[Vector]:
        return x[..., :m]
    
    # setattr(f,"__batched__",True)

    X:  PHom[Vector, Vector] = f

    # _ = ensure_PBatched(f)
    print(getattr(f, "__batched__", False))

    v :Vector = jnp.zeros((m,m,m),)