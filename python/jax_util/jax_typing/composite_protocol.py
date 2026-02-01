from __future__ import annotations

from jax_typing.base_protocol import*

from typing import Protocol


class LinOpProtocol(PMap, IsBatchedHom, PLinear, Protocol):
    def __matmul__(self, x: IsBatchedItem, /) -> IsBatchedItem:
        ...
    ...
__all__ = [
    "LinOpProtocol",
]
