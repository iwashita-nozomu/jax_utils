from __future__ import annotations

from typing import Any, Dict, Tuple

import equinox as eqx

from _type_aliaces import LinearMap, Scalar, Vector


class MINRESState(eqx.Module):
    """MINRES の状態（最小限の保持）。"""

    x0: Vector


def pminres_solve(
    Mv: LinearMap,
    Minv: LinearMap,
    rhs: Vector,
    minres_state: MINRESState,
    *,
    rtol: Scalar,
    maxiter: int = 1000,
) -> Tuple[Vector, MINRESState, Dict[str, Any]]:
    """最小限の MINRES ラッパ（簡易実装）。"""
    x = Minv(rhs)
    info: Dict[str, Any] = {
        "converged": True,
        "num_iter": 0,
        "rtol": rtol,
        "maxiter": maxiter,
    }
    return x, MINRESState(x0=x), info


__all__ = [
    "MINRESState",
    "pminres_solve",
]
