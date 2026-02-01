from __future__ import annotations

from typing import Tuple

import equinox as eqx
import jax.numpy as jnp

from _env_value import DEFAULT_DTYPE
from _linop_utils import DiagLinOp, DiagOp
from _type_aliaces import LinearMap, Vector


class KKTScalingState(eqx.Module):
    """KKT ブロックのスケーリング状態を保持する。"""

    h_scale: Vector
    b_scale: Vector


def initialize_kkt_scaling_state(
    Hv: LinearMap,
    Bv: LinearMap,
    BTv: LinearMap,
    n_primal: int,
    n_dual: int,
    r_Hv: int,
    r_Bv: int,
) -> KKTScalingState:
    """スケーリングの初期状態を作る。

    現状は簡易実装として、スケールは 1 を返す。
    """
    h_scale = jnp.ones((n_primal,), dtype=DEFAULT_DTYPE)
    b_scale = jnp.ones((n_dual,), dtype=DEFAULT_DTYPE)
    return KKTScalingState(h_scale=h_scale, b_scale=b_scale)


def update_kkt_scaling_state(
    Hv: LinearMap,
    Bv: LinearMap,
    BTv: LinearMap,
    state: KKTScalingState,
    *,
    scale: bool = False,
) -> Tuple[KKTScalingState, DiagLinOp, DiagLinOp]:
    """スケーリング状態とスケール演算子を返す。

    現状は簡易実装として、状態を維持しながら対角演算子を返す。
    """
    h_scale = DiagOp(state.h_scale)
    b_scale = DiagOp(state.b_scale)
    return state, h_scale, b_scale


__all__ = [
    "KKTScalingState",
    "initialize_kkt_scaling_state",
    "update_kkt_scaling_state",
]
