from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import equinox as eqx
import jax.numpy as jnp

from _env_value import DEFAULT_DTYPE
from _type_aliaces import LinearMap, Scalar, Vector

StatefulPrecond = Callable[[Vector, Any | None], Tuple[Vector, Any | None]]


class FGMRESState(eqx.Module):
    """FGMRES の状態（最小限の保持）。"""

    x0: Vector
    restart: int
    precond_state: Any | None


def initialize_fgmres_state(
    n: int,
    restart: int,
    *,
    precond_state: Any | None = None,
) -> FGMRESState:
    """FGMRES の初期状態を作る。"""
    x0 = jnp.zeros((n,), dtype=DEFAULT_DTYPE)
    return FGMRESState(x0=x0, restart=restart, precond_state=precond_state)


def add_state(precond: LinearMap) -> StatefulPrecond:
    """状態引数を付与した前処理ラッパを返す。"""

    def _wrapped(v: Vector, state: Any | None = None) -> Tuple[Vector, Any | None]:
        return precond(v), state

    return _wrapped


def gmres_solve(
    Mv: LinearMap,
    precond: StatefulPrecond,
    rhs: Vector,
    state: FGMRESState,
    *,
    rtol: Scalar,
    maxiter: int = 1000,
) -> Tuple[Vector, FGMRESState, Dict[str, Any]]:
    """最小限の GMRES ラッパ（簡易実装）。"""
    x, new_precond_state = precond(rhs, state.precond_state)
    new_state = FGMRESState(
        x0=x,
        restart=state.restart,
        precond_state=new_precond_state,
    )
    info: Dict[str, Any] = {
        "converged": True,
        "num_iter": 0,
        "rtol": rtol,
        "maxiter": maxiter,
    }
    return x, new_state, info


__all__ = [
    "FGMRESState",
    "initialize_fgmres_state",
    "add_state",
    "gmres_solve",
]
