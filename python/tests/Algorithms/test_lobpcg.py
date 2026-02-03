from __future__ import annotations

import json

import jax.numpy as jnp

from jax_util.Algorithms.lobpcg import init_spectral_precond, update_subspace
from jax_util.base import LinOp, Vector


def test_lobpcg_large_case() -> None:
    """大きめの行列でスペクトル前処理の更新ができることを確認します。"""
    n = 120
    diag = jnp.linspace(1.0, 3.0, n)
    A = jnp.diag(diag)

    def mv(v: Vector) -> Vector:
        return A @ v

    op = LinOp(mv)
    state = init_spectral_precond(op, n=n, r=3, which="smallest")
    basis, new_state, info = update_subspace(
        op,
        base_precond=LinOp(lambda v: v),
        old_state=state,
        maxiter=30,
    )
    print(json.dumps({
        "case": "lobpcg_large",
        "num_iter": int(info["num_iter"]),
    }))
    assert basis.Q.shape[0] == n
    assert new_state.X.shape[0] == n
    assert "num_iter" in info


def test_lobpcg_eigenvalue_accuracy() -> None:
    """最小固有値が概ね正しいことを確認します。"""
    n = 120
    diag = jnp.linspace(2.0, 5.0, n)
    A = jnp.diag(diag)

    def mv(v: Vector) -> Vector:
        return A @ v

    op = LinOp(mv)
    state = init_spectral_precond(op, n=n, r=2, which="smallest")
    basis, _, _ = update_subspace(
        op,
        base_precond=LinOp(lambda v: v),
        old_state=state,
        maxiter=40,
    )
    lam = basis.eigenvalues[0]
    print(json.dumps({
        "case": "lobpcg_eig",
        "expected_min": 2.0,
        "lambda_min": float(lam),
    }))
    assert jnp.allclose(lam, 2.0, rtol=1e-2, atol=1e-2)


def test_lobpcg_ill_conditioned_spectrum() -> None:
    """悪条件なスペクトルでも最小固有値が近いことを確認します。"""
    n = 200
    diag = jnp.logspace(0.0, 6.0, n)
    A = jnp.diag(diag)

    def mv(v: Vector) -> Vector:
        return A @ v

    op = LinOp(mv)
    state = init_spectral_precond(op, n=n, r=2, which="smallest")
    basis, _, _ = update_subspace(
        op,
        base_precond=LinOp(lambda v: v),
        old_state=state,
        maxiter=50,
    )
    lam = basis.eigenvalues[0]
    print(json.dumps({
        "case": "lobpcg_ill",
        "expected_min": 1.0,
        "lambda_min": float(lam),
    }))
    assert jnp.allclose(lam, 1.0, rtol=1e-2, atol=1e-2)


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_lobpcg_large_case()
    test_lobpcg_eigenvalue_accuracy()
    test_lobpcg_ill_conditioned_spectrum()


if __name__ == "__main__":
    _run_all_tests()
