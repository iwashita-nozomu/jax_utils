from __future__ import annotations

import json

import jax
import jax.numpy as jnp

from pathlib import Path
from jax_util.solvers.pcg import PCGState, pcg_solve
from jax_util.base import LinOp, Vector


SOURCE_FILE = Path(__file__).name


def test_pcg_known_solution() -> None:
    """大きめの SPD 系で解が復元できることを確認します。"""
    n = 200
    key = jax.random.PRNGKey(0)
    diag = jnp.linspace(1.0, 3.0, n)
    A = jnp.diag(diag)
    x_true = jax.random.normal(key, (n,))

    def mv(v: Vector) -> Vector:
        return A @ v

    op = LinOp(mv)
    precond = LinOp(lambda v: v)

    rhs = A @ x_true

    x, state, info = pcg_solve(
        Mv=op,
        precond=precond,
        rhs=rhs,
        pcg_state=PCGState(x0=jnp.zeros_like(x_true)),
        maxiter=200,
        rtol=jnp.asarray(1e-8),
    )
    print(json.dumps({
        "case": "pcg_known",
        "test": "test_pcg_known_solution",
        "source_file": SOURCE_FILE,
        "expected_norm": float(jnp.linalg.norm(x_true)),
        "num_iter": int(info["num_iter"]),
    }))
    assert jnp.allclose(x, x_true, rtol=1e-6, atol=1e-6)
    assert "num_iter" in info
    assert state.x0.shape == x_true.shape


def test_pcg_with_projection() -> None:
    """投影付きで解が射影空間に乗ることを確認します。"""
    n = 200
    A = jnp.eye(n)

    def mv(v: Vector) -> Vector:
        return A @ v

    op = LinOp(mv)
    precond = LinOp(lambda v: v)

    def proj(v: Vector) -> Vector:
        return v.at[1:].set(0.0)

    proj_op = LinOp(proj)
    rhs = jnp.ones((n,)) * 2.0

    x, _, _ = pcg_solve(
        Mv=op,
        precond=precond,
        rhs=rhs,
        pcg_state=PCGState(x0=jnp.zeros_like(rhs)),
        maxiter=50,
        rtol=jnp.asarray(1e-8),
        proj=proj_op,
    )
    expected = jnp.zeros((n,))
    expected = expected.at[0].set(2.0)
    print(json.dumps({
        "case": "pcg_projection",
        "source_file": SOURCE_FILE,
        "test": "test_pcg_with_projection",
        "expected_head": expected[:5].tolist(),
        "x0": float(x[0]),
    }))
    assert jnp.allclose(x, expected)


def test_pcg_ill_conditioned_system() -> None:
    """悪条件な SPD 系でも解が復元できることを確認します。"""
    n = 300
    diag = jnp.logspace(0.0, 6.0, n)
    A = jnp.diag(diag)

    def mv(v: Vector) -> Vector:
        return A @ v

    op = LinOp(mv)
    precond = LinOp(lambda v: v)

    key = jax.random.PRNGKey(7)
    x_true = jax.random.normal(key, (n,))
    rhs = A @ x_true

    x, _, info = pcg_solve(
        Mv=op,
        precond=precond,
        rhs=rhs,
        pcg_state=PCGState(x0=jnp.zeros_like(x_true)),
        maxiter=400,
        rtol=jnp.asarray(1e-6),
    )
    final_rel_r = float(jnp.asarray(info["final_rel_r"]))
    print(json.dumps({
        "case": "pcg_ill",
        "source_file": SOURCE_FILE,
        "test": "test_pcg_ill_conditioned_system",
        "expected_head": x_true[:5].tolist(),
        "expected_norm": float(jnp.linalg.norm(x_true)),
        "x_norm": float(jnp.linalg.norm(x)),
        "final_rel_r": final_rel_r,
    }))
    assert final_rel_r < 1e-3


def test_pcg_zero_rhs() -> None:
    """右辺ゼロならゼロ解に収束することを確認します。"""
    n = 50
    A = jnp.diag(jnp.linspace(1.0, 2.0, n))

    def mv(v: Vector) -> Vector:
        return A @ v

    op = LinOp(mv)
    precond = LinOp(lambda v: v)
    rhs = jnp.zeros((n,))

    x, _, info = pcg_solve(
        Mv=op,
        precond=precond,
        rhs=rhs,
        pcg_state=PCGState(x0=jnp.zeros_like(rhs)),
        maxiter=20,
        rtol=jnp.asarray(1e-8),
    )
    x_norm = float(jnp.linalg.norm(x))
    num_iter = int(jnp.asarray(info["num_iter"]))
    print(json.dumps({
        "case": "pcg_zero_rhs",
        "source_file": SOURCE_FILE,
        "test": "test_pcg_zero_rhs",
        "expected_norm": 0.0,
        "x_norm": x_norm,
        "num_iter": num_iter,
    }))
    assert x_norm < 1e-12


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_pcg_known_solution()
    test_pcg_with_projection()
    test_pcg_ill_conditioned_system()
    test_pcg_zero_rhs()


if __name__ == "__main__":
    _run_all_tests()
