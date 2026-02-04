from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from ..base import *


SOURCE_FILE = Path(__file__).name


# =========================
# PCG implementation
# =========================
class PCGState(eqx.Module):
    x0: Vector


def pcg_solve(
    Mv: LinearOperator,
    precond: LinearOperator,  # r -> M^{-1} r
    rhs: Vector,
    pcg_state: PCGState,
    *,
    maxiter: int = 200,
    rtol: Optional[Scalar] = None,
    atol: Optional[Scalar] = None,
    proj: LinearOperator = LinOp(lambda x: x),  # v -> Pv

) -> Tuple[Vector, PCGState, Dict[str, Any]]:
    """前処理付き PCG で A x = rhs を解く（A は SPD 前提）。"""
    pMv :LinearOperator = Mv * proj  # 投影付きMv
    pprecond :LinearOperator = proj * precond * proj  # 投影付き前処理

    rhs_p= proj @ rhs
    if DEBUG:
        jax.debug.print(
            "{{\"case\":\"pcg\",\"source_file\":\"{source_file}\","
            "\"func\":\"pcg_solve\",\"event\":\"rhs_norm\",\"value\":{value}}}",
            source_file=SOURCE_FILE,
            value=jnp.linalg.norm(rhs_p),
        )
    x0 = proj @ pcg_state.x0

    r0 = rhs_p - pMv @ x0
    z0 = pprecond @ r0
    p0 = z0
    rs0 = jnp.dot(r0, z0)

    # squared residual norm
    r0_norm = jnp.dot(r0, r0)
    r0_norm_safe = jnp.where(r0_norm == 0, jnp.asarray(1.0, DEFAULT_DTYPE), r0_norm)

    # tol is in "squared-norm" scale because we compare against r·r
    atol_val = jnp.asarray(atol, DEFAULT_DTYPE) if atol is not None else jnp.asarray(0.0, DEFAULT_DTYPE)
    rtol_val = jnp.asarray(rtol, DEFAULT_DTYPE) if rtol is not None else jnp.asarray(0.0, DEFAULT_DTYPE)
    tol = jnp.maximum(atol_val * atol_val, (rtol_val * rtol_val) * r0_norm)

    if DEBUG:
        jax.debug.print(
            "{{\"case\":\"pcg\",\"source_file\":\"{source_file}\","
            "\"func\":\"pcg_solve\",\"event\":\"init\","
            "\"r0_norm\":{r0},\"tol\":{tol},\"maxiter\":{maxiter}}}",
            source_file=SOURCE_FILE,
            r0=r0_norm,
            tol=tol,
            maxiter=maxiter,
        )

    done0 = (r0_norm <= tol)

    state0 = (jnp.asarray(0, jnp.int32), x0, r0, z0, p0, rs0, done0)

    def cond_fun(state: Tuple[Any, ...]) -> bool:
        i, x, r, z, p, rs, done = state
        return (i < maxiter) & (~done)

    def body_fun(state:Tuple[Any, ...]) -> Tuple[Any, ...]:
        i, x, r, z, p, rs, done = state

        Ap = pMv @ p
        denom = jnp.dot(p, Ap)
        denom = jnp.where(jnp.abs(denom) < AVOID_ZERO_DIV, AVOID_ZERO_DIV, denom)
        alpha = rs / denom

        x_new = proj @ (x + alpha * p)
        r_new = proj @ (r - alpha * Ap)
        z_new = pprecond @ r_new

        r_norm = jnp.dot(r_new, r_new)
        done_new = (r_norm <= tol)

        rs_new = jnp.dot(r_new, z_new)
        rs_safe = jnp.where(jnp.abs(rs) < AVOID_ZERO_DIV, AVOID_ZERO_DIV, rs)
        beta = rs_new / rs_safe
        p_new = proj @ (z_new + beta * p)

        return (i + 1, x_new, r_new, z_new, p_new, rs_new, done_new)

    i_f, x_f, r_f, z_f, p_f, rs_f, done_f = lax.while_loop(cond_fun, body_fun, state0)

    rf_norm = jnp.linalg.norm(r_f)
    info: Dict[str, Any] = {
        "final_norm_r": rf_norm,
        "final_rel_r": rf_norm / jnp.sqrt(r0_norm_safe),
        "converged": done_f,
        "num_iter": i_f,
    }
    if DEBUG:
        jax.debug.print(
            "{{\"case\":\"pcg\",\"source_file\":\"{source_file}\","
            "\"func\":\"pcg_solve\",\"event\":\"summary\","
            "\"num_iter\":{num_iter},\"final_norm_r\":{final_norm_r},"
            "\"final_rel_r\":{final_rel_r},\"converged\":{converged}}}",
            source_file=SOURCE_FILE,
            num_iter=i_f,
            final_norm_r=info["final_norm_r"],
            final_rel_r=info["final_rel_r"],
            converged=info["converged"],
        )
        jax.debug.print(
            "{{\"case\":\"pcg\",\"source_file\":\"{source_file}\","
            "\"func\":\"pcg_solve\",\"event\":\"return\"}}"
            ,
            source_file=SOURCE_FILE,
        )
        jax.debug.print("")
    return x_f, PCGState(x0=x_f), info


