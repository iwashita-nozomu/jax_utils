from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import pytest

from jax_util.base import LinOp, Vector
from jax_util.base.linearoperator import hstack_linops, vstack_linops


SOURCE_FILE = Path(__file__).name


def test_linearoperator_matmul_vector() -> None:
    """LinOp の @ 演算でベクトル適用できることを確認します。"""
    A = jnp.array([[4.0, 1.0], [1.0, 3.0]])

    def mv(v: Vector) -> Vector:
        return A @ v

    op = LinOp(mv)
    x = jnp.array([1.0, 2.0])
    y = op @ x
    expected = A @ x
    print(json.dumps({
        "case": "linop_matmul",
        "source_file": SOURCE_FILE,
        "test": "test_linearoperator_matmul_vector",
        "expected": expected.tolist(),
        "y": y.tolist(),
    }))
    assert jnp.allclose(y, A @ x)


def test_linearoperator_composition() -> None:
    """LinOp の合成ができることを確認します。"""
    A = jnp.array([[2.0, 0.0], [0.0, 3.0]])
    B = jnp.array([[1.0, 1.0], [0.0, 1.0]])

    def mv_a(v: Vector) -> Vector:
        return A @ v

    def mv_b(v: Vector) -> Vector:
        return B @ v

    op_a = LinOp(mv_a)
    op_b = LinOp(mv_b)
    op_c = op_a * op_b

    x = jnp.array([1.0, 2.0])
    y = op_c @ x
    expected = A @ (B @ x)
    print(json.dumps({
        "case": "linop_compose",
        "source_file": SOURCE_FILE,
        "test": "test_linearoperator_composition",
        "expected": expected.tolist(),
        "y": y.tolist(),
    }))
    assert jnp.allclose(y, A @ (B @ x))


def test_linearoperator_batched_input() -> None:
    """LinOp が行列入力をバッチとして処理できることを確認します。"""
    A = jnp.array([[2.0, 0.0], [0.0, 3.0]])

    def mv(v: Vector) -> Vector:
        return A @ v

    op = LinOp(mv)
    X = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    Y = op @ X
    expected = A @ X
    print(json.dumps({
        "case": "linop_batch",
        "source_file": SOURCE_FILE,
        "test": "test_linearoperator_batched_input",
        "expected_shape": list(expected.shape),
        "y_shape": list(Y.shape),
    }))
    assert jnp.allclose(Y, A @ X)


def test_linearoperator_mul_rejects_vector_operand() -> None:
    """LinOp の `*` がベクトルを受けたとき ValueError を返すことを確認します。"""
    op = LinOp(lambda v: v)

    with pytest.raises(ValueError, match="vector"):
        _ = op * jnp.array([1.0, 2.0])


def test_hstack_linops_block_row_sum() -> None:
    """hstack_linops が block-row 加算合成を実装していることを確認します。"""
    op_a = LinOp(lambda v: jnp.array([2.0 * v[0]]), shape=(1, 1))
    op_b = LinOp(lambda v: jnp.array([3.0 * v[0]]), shape=(1, 1))

    op = hstack_linops([op_a, op_b])
    x = jnp.array([4.0, 5.0])
    y = op @ x
    expected = jnp.array([2.0 * 4.0 + 3.0 * 5.0])
    print(json.dumps({
        "case": "hstack_block_row_sum",
        "source_file": SOURCE_FILE,
        "test": "test_hstack_linops_block_row_sum",
        "expected": expected.tolist(),
        "y": y.tolist(),
    }))
    assert jnp.allclose(y, expected)


def test_vstack_linops_reports_input_dimension_mismatch() -> None:
    """vstack_linops の次元エラーが入力次元として報告されることを確認します。"""
    op_a = LinOp(lambda v: v, shape=(2, 2))
    op_b = LinOp(lambda v: v, shape=(3, 3))

    with pytest.raises(ValueError, match="input dimension"):
        _ = vstack_linops([op_a, op_b])


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_linearoperator_matmul_vector()
    test_linearoperator_composition()
    test_linearoperator_batched_input()
    test_linearoperator_mul_rejects_vector_operand()
    test_hstack_linops_block_row_sum()
    test_vstack_linops_reports_input_dimension_mismatch()


if __name__ == "__main__":
    _run_all_tests()
