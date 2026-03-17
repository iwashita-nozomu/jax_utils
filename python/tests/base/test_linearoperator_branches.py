from __future__ import annotations

import jax.numpy as jnp
import pytest

from jax_util.base.linearoperator import LinOp, hstack_linops, stack_linops, vstack_linops


def _linop_from_matrix(matrix: jnp.ndarray) -> LinOp:
    return LinOp(lambda v: matrix @ v, shape=matrix.shape)


def test_linop_shape_property_requires_explicit_shape() -> None:
    op = LinOp(lambda v: v)

    with pytest.raises(ValueError, match="Shape is not specified"):
        _ = op.shape


def test_linop_rejects_rank_three_input() -> None:
    op = _linop_from_matrix(jnp.eye(2))

    with pytest.raises(ValueError, match="1D or 2D"):
        _ = op @ jnp.ones((2, 2, 1))


def test_linop_operator_matmul_builds_composition() -> None:
    op_a = _linop_from_matrix(jnp.array([[2.0, 0.0], [0.0, 3.0]]))
    op_b = _linop_from_matrix(jnp.array([[1.0, 1.0], [0.0, 1.0]]))

    composed = op_a @ op_b
    x = jnp.array([1.0, 2.0])

    assert jnp.allclose(composed @ x, jnp.array([6.0, 6.0]))


def test_linop_mul_supports_scalar_and_matrix_operands() -> None:
    op = _linop_from_matrix(jnp.array([[1.0, 2.0], [0.0, 1.0]]))
    x = jnp.array([2.0, 3.0])

    scaled = op * jnp.asarray(2.0)
    multiplied = op * jnp.array([[3.0, 0.0], [0.0, 4.0]])

    assert jnp.allclose(scaled @ x, 2.0 * (op @ x))
    assert jnp.allclose(multiplied @ x, op @ (jnp.array([[3.0, 0.0], [0.0, 4.0]]) @ x))

    with pytest.raises(ValueError, match="vector"):
        _ = op * jnp.array([1.0, 2.0])
    with pytest.raises(ValueError, match="3D array"):
        _ = op * jnp.ones((2, 2, 1))


def test_linop_rmul_supports_scalar_and_matrix_operands() -> None:
    op = _linop_from_matrix(jnp.array([[1.0, 2.0], [0.0, 1.0]]))
    x = jnp.array([2.0, 3.0])

    scaled = op.__rmul__(jnp.asarray(3.0))
    multiplied = op.__rmul__(jnp.array([[3.0, 0.0], [0.0, 4.0]]))
    composed = op.__rmul__(_linop_from_matrix(jnp.array([[2.0, 0.0], [0.0, 5.0]])))

    assert jnp.allclose(scaled @ x, 3.0 * (op @ x))
    assert jnp.allclose(multiplied @ x, jnp.array([[3.0, 0.0], [0.0, 4.0]]) @ (op @ x))
    assert jnp.allclose(composed @ x, jnp.array([[2.0, 0.0], [0.0, 5.0]]) @ (op @ x))

    with pytest.raises(ValueError, match="vector"):
        _ = op.__rmul__(jnp.array([1.0, 2.0])) @ x
    with pytest.raises(ValueError, match="3D array"):
        _ = op.__rmul__(jnp.ones((2, 2, 1))) @ x


def test_linop_add_and_block_combinators_cover_remaining_branches() -> None:
    op_a = _linop_from_matrix(jnp.array([[1.0, 0.0], [0.0, 1.0]]))
    op_b = _linop_from_matrix(jnp.array([[2.0, 0.0], [0.0, 3.0]]))
    x = jnp.array([1.0, 2.0])

    added = op_a + op_b
    assert jnp.allclose(added @ x, jnp.array([3.0, 8.0]))

    with pytest.raises(ValueError, match="same output dimension"):
        _ = hstack_linops(
            [
                LinOp(lambda v: jnp.array([v[0]]), shape=(1, 1)),
                LinOp(lambda v: jnp.array([v[0], v[0]]), shape=(2, 1)),
            ]
        )

    stacked_vertically = vstack_linops(
        [
            LinOp(lambda v: jnp.array([v[0] + v[1]]), shape=(1, 2)),
            LinOp(lambda v: jnp.array([2.0 * v[0] - v[1]]), shape=(1, 2)),
        ]
    )
    assert jnp.allclose(stacked_vertically @ x, jnp.array([3.0, 0.0]))

    block_op = stack_linops(
        [
            [
                LinOp(lambda v: jnp.array([v[0]]), shape=(1, 1)),
                LinOp(lambda v: jnp.array([2.0 * v[0]]), shape=(1, 1)),
            ],
            [
                LinOp(lambda v: jnp.array([3.0 * v[0]]), shape=(1, 1)),
                LinOp(lambda v: jnp.array([4.0 * v[0]]), shape=(1, 1)),
            ],
        ]
    )
    assert jnp.allclose(block_op @ x, jnp.array([5.0, 11.0]))
