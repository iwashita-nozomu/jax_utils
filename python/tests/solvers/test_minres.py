from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from jax_util.solvers._minres import MINRESState, pminres_solve
from jax_util.base import LinOp, Vector

SOURCE_FILE = Path(__file__).name


def test_minres_known_solution() -> None:
    """大きめの対称系で解が復元できることを確認します。"""
    n = 200
    diag = jnp.linspace(1.0, 4.0, n)
    A = jnp.diag(diag)

    def mv(v: Vector) -> Vector:
        return A @ v

    op = LinOp(mv)
    precond = LinOp(lambda v: v)

    key = jax.random.PRNGKey(1)
    x_true = jax.random.normal(key, (n,))
    rhs = A @ x_true

    x, state, info = pminres_solve(
        Mv=op,
        Minv=precond,
        rhs=rhs,
        minres_state=MINRESState.initialize(x0=jnp.zeros_like(x_true)),
        maxiter=200,
        rtol=jnp.asarray(1e-8),
    )
    print(
        json.dumps(
            {
                "case": "minres_known",
                "source_file": SOURCE_FILE,
                "test": "test_minres_known_solution",
                "expected_head": x_true[:5].tolist(),
                "expected_norm": float(jnp.linalg.norm(x_true)),
                "num_iter": int(info["num_iter"]),
            }
        )
    )
    assert jnp.allclose(x, x_true, rtol=1e-6, atol=1e-6)
    assert "num_iter" in info
    assert state.x0.shape == x_true.shape


def test_minres_indefinite_system() -> None:
    """対称だが非 SPD の大きめの系でも解が復元できることを確認します。"""
    n = 200
    diag = jnp.concatenate([jnp.full((n // 2,), -1.0), jnp.full((n - n // 2,), 2.0)])
    A = jnp.diag(diag)

    def mv(v: Vector) -> Vector:
        return A @ v

    op = LinOp(mv)
    precond = LinOp(lambda v: v)

    key = jax.random.PRNGKey(2)
    x_true = jax.random.normal(key, (n,))
    rhs = A @ x_true

    x, _, _ = pminres_solve(
        Mv=op,
        Minv=precond,
        rhs=rhs,
        minres_state=MINRESState.initialize(x0=jnp.zeros_like(x_true)),
        maxiter=300,
        rtol=jnp.asarray(1e-8),
    )
    print(
        json.dumps(
            {
                "case": "minres_indef",
                "source_file": SOURCE_FILE,
                "test": "test_minres_indefinite_system",
                "expected_head": x_true[:5].tolist(),
                "expected_norm": float(jnp.linalg.norm(x_true)),
                "x_norm": float(jnp.linalg.norm(x)),
            }
        )
    )
    assert jnp.allclose(x, x_true, rtol=1e-6, atol=1e-6)


def test_minres_ill_conditioned_system() -> None:
    """悪条件な対称系でも解が復元できることを確認します。"""
    n = 300
    diag = jnp.logspace(0.0, 8.0, n)
    A = jnp.diag(diag)

    def mv(v: Vector) -> Vector:
        return A @ v

    op = LinOp(mv)
    precond = LinOp(lambda v: v)

    key = jax.random.PRNGKey(6)
    x_true = jax.random.normal(key, (n,))
    rhs = A @ x_true

    x, _, info = pminres_solve(
        Mv=op,
        Minv=precond,
        rhs=rhs,
        minres_state=MINRESState.initialize(x0=jnp.zeros_like(x_true)),
        maxiter=600,
        rtol=jnp.asarray(1e-6),
    )
    final_rel_r = float(jnp.asarray(info["final_rel_r"]))
    print(
        json.dumps(
            {
                "case": "minres_ill",
                "source_file": SOURCE_FILE,
                "test": "test_minres_ill_conditioned_system",
                "expected_head": x_true[:5].tolist(),
                "expected_norm": float(jnp.linalg.norm(x_true)),
                "x_norm": float(jnp.linalg.norm(x)),
                "final_rel_r": final_rel_r,
            }
        )
    )
    assert final_rel_r < 1e-3


def test_minres_zero_rhs() -> None:
    """右辺ゼロならゼロ解に収束することを確認します。"""
    n = 60
    A = jnp.diag(jnp.linspace(1.0, 3.0, n))

    def mv(v: Vector) -> Vector:
        return A @ v

    op = LinOp(mv)
    precond = LinOp(lambda v: v)
    rhs = jnp.zeros((n,))

    x, _, info = pminres_solve(
        Mv=op,
        Minv=precond,
        rhs=rhs,
        minres_state=MINRESState.initialize(x0=jnp.zeros_like(rhs)),
        maxiter=20,
        rtol=jnp.asarray(1e-8),
    )
    x_norm = float(jnp.linalg.norm(x))
    num_iter = int(jnp.asarray(info["num_iter"]))
    print(
        json.dumps(
            {
                "case": "minres_zero_rhs",
                "source_file": SOURCE_FILE,
                "test": "test_minres_zero_rhs",
                "expected_norm": 0.0,
                "x_norm": x_norm,
                "num_iter": num_iter,
            }
        )
    )
    assert x_norm < 1e-12


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_minres_known_solution()
    test_minres_indefinite_system()
    test_minres_ill_conditioned_system()
    test_minres_zero_rhs()


if __name__ == "__main__":
    _run_all_tests()
