from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from jax_util.solvers.kkt_solver import (
    KKTState,
    initialize_kkt_state,
    kkt_block_solver,
)
from jax_util.base import LinOp, Vector

SOURCE_FILE = Path(__file__).name


def test_kkt_state_initialize() -> None:
    """KKT 状態が初期化できることを確認します。"""

    def hv(v: Vector) -> Vector:
        return v

    def bv(v: Vector) -> Vector:
        return jnp.zeros((1,), dtype=v.dtype)

    def btv(v: Vector) -> Vector:
        return jnp.zeros((2,), dtype=v.dtype)

    state = initialize_kkt_state(
        Hv_initial=LinOp(hv),
        Bv_initial=LinOp(bv),
        BTv_initial=LinOp(btv),
        n_primal=2,
        n_dual=1,
        r_Hv_min=1,
        r_Sv_min=1,
        method="minres",
    )
    print(
        json.dumps(
            {
                "case": "kkt_state_init",
                "source_file": SOURCE_FILE,
                "test": "test_kkt_state_initialize",
                "expected_method": "minres",
                "method": state.method,
            }
        )
    )
    assert isinstance(state, KKTState)
    assert state.method == "minres"


def test_kkt_block_solver_large_system() -> None:
    """大きめの KKT 系が解けることを確認します。"""
    n_primal = 80
    n_dual = 20
    key = jax.random.PRNGKey(5)
    diag = jnp.linspace(1.0, 3.0, n_primal)
    A = jnp.diag(diag)
    B = jax.random.normal(key, (n_dual, n_primal)) * 0.1

    def hv(v: Vector) -> Vector:
        return A @ v

    def bv(v: Vector) -> Vector:
        return B @ v

    def btv(v: Vector) -> Vector:
        return B.T @ v

    state = initialize_kkt_state(
        Hv_initial=LinOp(hv),
        Bv_initial=LinOp(bv),
        BTv_initial=LinOp(btv),
        n_primal=n_primal,
        n_dual=n_dual,
        r_Hv_min=1,
        r_Sv_min=1,
        method="minres",
    )

    rhs_x = jax.random.normal(key, (n_primal,))
    rhs_lam = jax.random.normal(key, (n_dual,))

    (x, lam), _, info = kkt_block_solver(
        Hv=LinOp(hv),
        Bv=LinOp(bv),
        BTv=LinOp(btv),
        rhs_x=rhs_x,
        rhs_lam=rhs_lam,
        kkt_state=state,
        kkt_tol=jnp.asarray(1e-8),
        maxiter=800,
    )

    kkt_mat = jnp.block([[A, B.T], [B, jnp.zeros((n_dual, n_dual))]])
    rhs = jnp.concatenate([rhs_x, rhs_lam], axis=0)
    sol = jnp.linalg.solve(kkt_mat, rhs)

    res_norm = float(jnp.asarray(info["res_norm"]))
    rel_res = float(jnp.asarray(info["rel_res"]))
    print(
        json.dumps(
            {
                "case": "kkt_large",
                "source_file": SOURCE_FILE,
                "test": "test_kkt_block_solver_large_system",
                "expected_head": sol[:5].tolist(),
                "res_norm": res_norm,
                "rel_res": rel_res,
            }
        )
    )
    assert jnp.allclose(x, sol[:n_primal], rtol=1e-5, atol=1e-5)
    assert jnp.allclose(lam, sol[n_primal:], rtol=1e-5, atol=1e-5)


def test_kkt_block_solver_ill_conditioned() -> None:
    """悪条件な KKT 系が解けることを確認します。"""
    n_primal = 120
    n_dual = 30
    key = jax.random.PRNGKey(9)
    diag = jnp.logspace(0.0, 8.0, n_primal)
    A = jnp.diag(diag)
    B = jax.random.normal(key, (n_dual, n_primal)) * 0.05

    def hv(v: Vector) -> Vector:
        return A @ v

    def bv(v: Vector) -> Vector:
        return B @ v

    def btv(v: Vector) -> Vector:
        return B.T @ v

    state = initialize_kkt_state(
        Hv_initial=LinOp(hv),
        Bv_initial=LinOp(bv),
        BTv_initial=LinOp(btv),
        n_primal=n_primal,
        n_dual=n_dual,
        r_Hv_min=1,
        r_Sv_min=1,
        method="minres",
    )

    rhs_x = jax.random.normal(key, (n_primal,))
    rhs_lam = jax.random.normal(key, (n_dual,))

    (x, lam), _, info = kkt_block_solver(
        Hv=LinOp(hv),
        Bv=LinOp(bv),
        BTv=LinOp(btv),
        rhs_x=rhs_x,
        rhs_lam=rhs_lam,
        kkt_state=state,
        kkt_tol=jnp.asarray(1e-6),
        maxiter=1200,
    )

    kkt_mat = jnp.block([[A, B.T], [B, jnp.zeros((n_dual, n_dual))]])
    rhs = jnp.concatenate([rhs_x, rhs_lam], axis=0)
    sol = jnp.linalg.solve(kkt_mat, rhs)

    res_norm = float(jnp.asarray(info["res_norm"]))
    rel_res = float(jnp.asarray(info["rel_res"]))
    print(
        json.dumps(
            {
                "case": "kkt_ill_conditioned",
                "source_file": SOURCE_FILE,
                "test": "test_kkt_block_solver_ill_conditioned",
                "expected_head": sol[:5].tolist(),
                "res_norm": res_norm,
                "rel_res": rel_res,
            }
        )
    )
    assert rel_res < 5e-1


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_kkt_state_initialize()
    test_kkt_block_solver_large_system()
    test_kkt_block_solver_ill_conditioned()


if __name__ == "__main__":
    _run_all_tests()
