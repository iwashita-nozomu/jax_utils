from __future__ import annotations


from .protocols import *
from .linearoperator import LinOp

import equinox as eqx
import jax
from typing import Callable, Tuple


# 責務: 関数値と Jacobian の線形作用素を同時に返します。
def linearize(
    f: Callable[[Vector], Vector],
    x0: Vector,
) -> tuple[Vector, LinearOperator]:
    val, jvp_fn = jax.linearize(f, x0)  # jvp_fn(v) = J @ v

    linop = LinOp(mv=jvp_fn, shape=(val.shape[0], x0.shape[0]))  # (out_dim, in_dim)
    return val, linop


# 責務: 関数値と Jacobian の随伴作用素を同時に返します。
def adjoint(
    f: Callable[[Vector], Vector],
    x0: Vector,
) -> Tuple[Vector, LinearOperator]:
    val, vjp_fn = eqx.filter_vjp(f, x0)  # val = f(x0)

    # 責務: VJP を使って随伴作用素の matvec を実装します。
    def _adjoint_mv(v: Vector) -> Vector:
        (xbar,) = vjp_fn(v)  # xbar = J(x0)^T @ v
        return xbar

    # shape: (in_dim, out_dim) = (x-dim, y-dim)
    linop = LinOp(mv=_adjoint_mv, shape=(x0.shape[0], val.shape[0]))
    return val, linop
