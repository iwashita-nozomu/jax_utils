from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from jax_util.Algorithms._fgmres import GMRESState, GMRESWorkspace, gmres_solve, initialize_fgmres_state
from jax_util.base import LinOp, Vector


def test_fgmres_known_solution() -> None:
    """大きめの系で GMRES が解に近づくことを確認します。"""
    n = 150
    key = jax.random.PRNGKey(3)
    diag = jnp.linspace(1.0, 3.0, n)
    A = jnp.diag(diag)

    def mv(v: Vector) -> Vector:
        return A @ v

    op = LinOp(mv)
    precond = LinOp(lambda v: v)

    x_true = jax.random.normal(key, (n,))
    rhs = A @ x_true

    state = initialize_fgmres_state(n=n, restart=20, precond_state=None)
    x, new_state, info = gmres_solve(
        Mv=op,
        precond=precond,
        rhs=rhs,
        state=state,
        maxiter=200,
        rtol=1e-8,
    )
    print(json.dumps({
        "case": "fgmres_known",
        "expected_head": x_true[:5].tolist(),
        "expected_norm": float(jnp.linalg.norm(x_true)),
        "num_iter": int(info["num_iter"]),
    }))
    assert jnp.allclose(x, x_true, rtol=1e-6, atol=1e-6)
    assert "num_iter" in info
    assert isinstance(new_state[0], GMRESWorkspace)
    assert isinstance(new_state[1], GMRESState)


def test_fgmres_nonsymmetric_system() -> None:
    """非対称系でも解が復元できることを確認します。"""
    n = 120
    key = jax.random.PRNGKey(4)
    M = jax.random.normal(key, (n, n))
    A = M + 0.05 * jnp.eye(n)

    def mv(v: Vector) -> Vector:
        return A @ v

    op = LinOp(mv)
    precond = LinOp(lambda v: v)

    x_true = jax.random.normal(key, (n,))
    rhs = A @ x_true

    state = initialize_fgmres_state(n=n, restart=30, precond_state=None)
    x, _, info = gmres_solve(
        Mv=op,
        precond=precond,
        rhs=rhs,
        state=state,
        maxiter=300,
        rtol=1e-8,
    )
    print(json.dumps({
        "case": "fgmres_nonsym",
        "expected_head": x_true[:5].tolist(),
        "expected_norm": float(jnp.linalg.norm(x_true)),
        "num_iter": int(info["num_iter"]),
    }))
    assert jnp.allclose(x, x_true, rtol=1e-6, atol=1e-6)
    assert "num_iter" in info


def test_fgmres_ill_conditioned_system() -> None:
    """悪条件な非対称系でも解が復元できることを確認します。"""
    n = 300
    key = jax.random.PRNGKey(8)
    diag = jnp.logspace(0.0, 6.0, n)
    D = jnp.diag(diag)
    U = jax.random.normal(key, (n, n)) * 0.01
    A = D + U

    def mv(v: Vector) -> Vector:
        return A @ v

    op = LinOp(mv)
    precond = LinOp(lambda v: v)

    x_true = jax.random.normal(key, (n,))
    rhs = A @ x_true

    state = initialize_fgmres_state(n=n, restart=30, precond_state=None)
    x, _, _ = gmres_solve(
        Mv=op,
        precond=precond,
        rhs=rhs,
        state=state,
        maxiter=400,
        rtol=1e-6,
    )
    print(json.dumps({
        "case": "fgmres_ill",
        "expected_head": x_true[:5].tolist(),
        "expected_norm": float(jnp.linalg.norm(x_true)),
        "x_norm": float(jnp.linalg.norm(x)),
    }))
    assert jnp.allclose(x, x_true, rtol=1e-4, atol=1e-4)


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_fgmres_known_solution()
    test_fgmres_nonsymmetric_system()
    test_fgmres_ill_conditioned_system()


if __name__ == "__main__":
    _run_all_tests()
