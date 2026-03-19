from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from jax_util.base import LinOp, Vector

SOURCE_FILE = Path(__file__).name
from jax_util.base.nonlinearoperator import adjoint, linearize


def test_linearize_and_adjoint() -> None:
    """線形化と随伴作用素の生成が可能なことを確認します。"""
    A = jnp.array([[2.0, 0.0], [0.0, 3.0]])

    def f(v: Vector) -> Vector:
        return A @ v

    x0 = jnp.array([1.0, 2.0])
    val, linop = linearize(f, x0)
    assert jnp.allclose(val, A @ x0)

    v = jnp.array([3.0, 4.0])
    y = linop @ v
    expected = A @ v
    print(
        json.dumps(
            {
                "case": "linearize",
                "source_file": SOURCE_FILE,
                "test": "test_linearize_and_adjoint",
                "expected": expected.tolist(),
                "y": y.tolist(),
            }
        )
    )
    assert jnp.allclose(y, A @ v)

    _, adj = adjoint(f, x0)
    adj_y = adj @ v
    expected_adj = A.T @ v
    print(
        json.dumps(
            {
                "case": "adjoint",
                "source_file": SOURCE_FILE,
                "test": "test_linearize_and_adjoint",
                "expected": expected_adj.tolist(),
                "adj_y": adj_y.tolist(),
            }
        )
    )
    assert jnp.allclose(adj_y, A.T @ v)


def test_nonlinear_linop_roundtrip() -> None:
    """LinOp が作れることを確認します。"""

    def mv(v: Vector) -> Vector:
        return v

    op = LinOp(mv)
    x = jnp.array([1.0, 0.0])
    y = op @ x
    print(
        json.dumps(
            {
                "case": "nonlinear_linop",
                "source_file": SOURCE_FILE,
                "test": "test_nonlinear_linop_roundtrip",
                "expected": x.tolist(),
                "y": y.tolist(),
            }
        )
    )
    assert jnp.allclose(y, x)


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_linearize_and_adjoint()
    test_nonlinear_linop_roundtrip()


if __name__ == "__main__":
    _run_all_tests()
