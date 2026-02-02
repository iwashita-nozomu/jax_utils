from __future__ import annotations
import os
if __name__ == "__main__":
    # 必要なら GPU 固定（不要なら消してOK）
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from typing import Any, Dict, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from base import *


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
        jax.debug.print("PCG rhs norm: {norm}", norm=jnp.linalg.norm(rhs_p))
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
        beta = rs_new / rs_safe #pyright: ignore
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
        jax.debug.print("PCG info: {info}", info=info)
    return x_f, PCGState(x0=x_f), info


# =========================
# Tests
# =========================
def _make_spd(n: int, key: Vector) -> Matrix:
    """SPD matrix: A = Q^T Q + mu I"""
    Q = jax.random.normal(key, (n, n), dtype=DEFAULT_DTYPE)
    mu = jnp.asarray(0.1, DEFAULT_DTYPE)
    return Q.T @ Q + mu * jnp.eye(n, dtype=DEFAULT_DTYPE)


def test_pcg_dense_spd_identity_precond():
    key = jax.random.PRNGKey(0)
    n = 32
    A = _make_spd(n, key)

    x_true = jax.random.normal(jax.random.PRNGKey(1), (n,), dtype=DEFAULT_DTYPE)
    b = A @ x_true

    Mv = LinOp(lambda v: A @ v) #pyright: ignore
    precond = lambda r: r  # identity #pyright: ignore

    x0 = jnp.zeros((n,), dtype=DEFAULT_DTYPE)
    x, st, info = pcg_solve(Mv, LinOp(precond), b, PCGState(x0=x0), maxiter=500, rtol=EPS, atol=ZERO)

    # check against direct solve
    x_ref = jnp.linalg.solve(A, b)
    err = jnp.linalg.norm(x - x_ref) / jnp.linalg.norm(x_ref)

    print("[dense/identity] info:", {k: (int(v) if k == "num_iter" else v) for k, v in info.items()})
    print("[dense/identity] rel_err(x):", err)

    assert bool(info["converged"]), "PCG did not converge"
    assert float(err) < 1e-5, f"solution error too large: {err}"


def test_pcg_dense_spd_jacobi_precond():
    key = jax.random.PRNGKey(2)
    n = 64
    A = _make_spd(n, key)

    x_true = jax.random.normal(jax.random.PRNGKey(3), (n,), dtype=DEFAULT_DTYPE)
    b = A @ x_true

    Mv = LinOp(lambda v: A @ v) #pyright: ignore

    diag = jnp.diag(A)
    inv_diag = 1.0 / jnp.where(jnp.abs(diag) < 1e-12, 1e-12, diag)
    precond = LinOp(lambda r: inv_diag * r)  # Jacobi#pyright: ignore
    x0 = jnp.zeros((n,), dtype=DEFAULT_DTYPE)
    x, st, info = pcg_solve(Mv, precond, b, PCGState(x0=x0), maxiter=500, rtol=EPS, atol=ZERO)

    x_ref = jnp.linalg.solve(A, b)
    err = jnp.linalg.norm(x - x_ref) / jnp.linalg.norm(x_ref)

    print("[dense/jacobi] info:", {k: (int(v) if k == "num_iter" else v) for k, v in info.items()})
    print("[dense/jacobi] rel_err(x):", err)

    assert bool(info["converged"]), "PCG did not converge"
    assert float(err) < 1e-5, f"solution error too large: {err}"


def test_initial_residual_zero():
    key = jax.random.PRNGKey(4)
    n = 32
    A = _make_spd(n, key)

    x0 = jax.random.normal(jax.random.PRNGKey(5), (n,), dtype=DEFAULT_DTYPE)
    b = A @ x0  # so residual is exactly zero (up to fp)
    Mv = LinOp(lambda v: A @ v) #pyright: ignore
    precond = LinOp(lambda r: r)  # identity #pyright: ignore

    x, st, info = pcg_solve(Mv, precond, b, PCGState(x0=x0), maxiter=50, rtol=EPS, atol=ZERO)

    print("[r0=0] info:", {k: (int(v) if k == "num_iter" else v) for k, v in info.items()})
    # should stop immediately
    assert int(info["num_iter"]) == 0, "should not iterate when initial residual is zero"
    assert float(jnp.linalg.norm(x - x0)) == 0.0, "solution should equal initial x0"


def test_jit_compiles():
    key = jax.random.PRNGKey(6)
    n = 16
    A = _make_spd(n, key)
    b = jax.random.normal(jax.random.PRNGKey(7), (n,), dtype=DEFAULT_DTYPE)

    Mv = LinOp(lambda v: A @ v) #pyright: ignore
    precond = LinOp(lambda r: r)  # identity #pyright: ignore

    @jax.jit
    def solve(b, x0): #pyright: ignore
        return pcg_solve(Mv, precond, b, PCGState(x0=x0), maxiter=200, rtol=EPS, atol=ZERO)

    x0 = jnp.zeros((n,), dtype=DEFAULT_DTYPE)
    x, st, info = solve(b, x0)
    print("[jit] info:", {k: (int(v) if k == "num_iter" else v) for k, v in info.items()})


def run_all_tests():
    test_pcg_dense_spd_identity_precond()
    test_pcg_dense_spd_jacobi_precond()
    test_initial_residual_zero()
    test_jit_compiles()
    print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()
