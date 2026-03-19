from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from jax_util.solvers.matrix_util import orthonormalize

SOURCE_FILE = Path(__file__).name


def test_orthonormalize_basic() -> None:
    """直交化で列が正規直交になることを確認します。"""
    X = jnp.array([[1.0, 0.0], [0.0, 2.0]])
    Q = orthonormalize(X)
    eye = jnp.eye(Q.shape[1])
    print(
        json.dumps(
            {
                "case": "orthonormalize_basic",
                "source_file": SOURCE_FILE,
                "test": "test_orthonormalize_basic",
                "expected": eye.tolist(),
                "qtq": (Q.T @ Q).tolist(),
            }
        )
    )
    assert jnp.allclose(Q.T @ Q, eye)


def test_orthonormalize_idempotent() -> None:
    """既に直交な行列に対しては変化が小さいことを確認します。"""
    Q0 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    Q1 = orthonormalize(Q0)
    print(
        json.dumps(
            {
                "case": "orthonormalize_idempotent",
                "source_file": SOURCE_FILE,
                "test": "test_orthonormalize_idempotent",
                "expected": Q0.tolist(),
                "q1": Q1.tolist(),
            }
        )
    )
    assert jnp.allclose(Q1, Q0)


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_orthonormalize_basic()
    test_orthonormalize_idempotent()


if __name__ == "__main__":
    _run_all_tests()
