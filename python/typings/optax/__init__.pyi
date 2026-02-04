from __future__ import annotations

from typing import Any, Protocol, TypeVar


Params = Any
Updates = Any
OptState = Any


class GradientTransformation(Protocol):
    def init(self, params: Params, /) -> OptState: ...
    def update(self, updates: Updates, state: OptState, params: Params | None = None, /) -> tuple[Updates, OptState]: ...


def apply_updates(params: Params, updates: Updates, /) -> Params: ...


def sgd(
    learning_rate: float,
    momentum: float = 0.0,
    nesterov: bool = False,
) -> GradientTransformation: ...
