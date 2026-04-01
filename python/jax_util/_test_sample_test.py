"""改善版コードのテストスイート。

スキルセクション 1.2 のテストコード規約に従った実装。
"""

from __future__ import annotations

import pytest
import jax.numpy as jnp
from python.jax_util._test_sample_for_review_fixed import (
    solve_linear_system,
    compute_eigenvalues,
    LinearSolver,
    matrix_multiply_and_accumulate,
)


# ============================================================================
# solve_linear_system テスト
# ============================================================================

class TestSolveLinearSystem:
    """solve_linear_system() の検証。"""

    def test_solve_with_identity_matrix(self):
        """単位行列での求解が正しいことを検証。"""
        A = jnp.eye(3)
        b = jnp.array([1.0, 2.0, 3.0])
        x = solve_linear_system(A, b)
        assert jnp.allclose(x, b, atol=1e-10)

    def test_solve_with_spd_matrix(self):
        """対称正定値行列での検証。"""
        # 2x2 対称正定値行列
        A = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        b = jnp.array([3.0, 4.0])
        x = solve_linear_system(A, b)

        # Ax = b を確認
        residual = A @ x - b
        assert jnp.allclose(residual, 0.0, atol=1e-10)

    def test_solve_raises_for_non_square_matrix(self):
        """非正方行列で ValueError を raise することを検証。"""
        A = jnp.ones((3, 4))
        b = jnp.ones(3)
        with pytest.raises(ValueError, match="must be square"):
            solve_linear_system(A, b)

    def test_solve_with_singular_matrix(self):
        """特異行列（解くことができない）で例外を raise することを検証。"""
        A = jnp.array([[1.0, 2.0], [1.0, 2.0]])  # 特異行列
        b = jnp.array([1.0, 2.0])
        # JAX の linalg.solve が例外を raise するか確認
        with pytest.raises(Exception):  # LinAlgError
            solve_linear_system(A, b)


# ============================================================================
# compute_eigenvalues テスト
# ============================================================================

class TestComputeEigenvalues:
    """compute_eigenvalues() の検証。"""

    def test_eigenvalues_of_identity_matrix(self):
        """単位行列の固有値が [1, 1, 1] であることを検証。"""
        matrix = jnp.eye(3)
        eigenvalues = compute_eigenvalues(matrix)
        expected = jnp.ones(3)
        assert jnp.allclose(eigenvalues, expected, atol=1e-10)

    def test_eigenvalues_sorted_ascending(self):
        """固有値が昇順で返されることを検証。"""
        matrix = jnp.diag(jnp.array([3.0, 1.0, 2.0]))
        eigenvalues = compute_eigenvalues(matrix)
        expected = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(eigenvalues, expected, atol=1e-10)

    def test_eigenvalues_raises_for_non_square(self):
        """非正方行列で ValueError を raise することを検証。"""
        matrix = jnp.ones((3, 4))
        with pytest.raises(ValueError, match="square matrix"):
            compute_eigenvalues(matrix)

    def test_eigenvalues_raises_for_1d_array(self):
        """1D 配列で ValueError を raise することを検証。"""
        matrix = jnp.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="square matrix"):
            compute_eigenvalues(matrix)


# ============================================================================
# LinearSolver テスト
# ============================================================================

class TestLinearSolver:
    """LinearSolver クラスの検証。"""

    def test_init_with_default_method(self):
        """デフォルト method が 'lu' であることを検証。"""
        solver = LinearSolver()
        assert solver.method == "lu"

    def test_init_with_qr_method(self):
        """method='qr' での初期化を検証。"""
        solver = LinearSolver(method="qr")
        assert solver.method == "qr"

    def test_init_raises_for_invalid_method(self):
        """不正な method で ValueError を raise することを検証。"""
        with pytest.raises(ValueError, match="'lu' or 'qr'"):
            LinearSolver(method="svd")

    def test_solve_lu_method(self):
        """LU 法での求解が正しいことを検証。"""
        solver = LinearSolver(method="lu")
        A = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        b = jnp.array([3.0, 4.0])
        x = solver.solve(A, b)

        residual = A @ x - b
        assert jnp.allclose(residual, 0.0, atol=1e-10)

    def test_solve_qr_method(self):
        """QR 法での求解が正しいことを検証。"""
        solver = LinearSolver(method="qr")
        A = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        b = jnp.array([3.0, 4.0])
        x = solver.solve(A, b)

        residual = A @ x - b
        assert jnp.allclose(residual, 0.0, atol=1e-10)

    def test_solve_lu_vs_qr_equivalence(self):
        """LU 法と QR 法が同じ結果を返すことを検証。"""
        A = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        b = jnp.array([3.0, 4.0])

        solver_lu = LinearSolver(method="lu")
        solver_qr = LinearSolver(method="qr")

        x_lu = solver_lu.solve(A, b)
        x_qr = solver_qr.solve(A, b)

        assert jnp.allclose(x_lu, x_qr, atol=1e-10)

    def test_solve_raises_for_non_square(self):
        """非正方行列で ValueError を raise することを検証。"""
        solver = LinearSolver()
        A = jnp.ones((3, 4))
        b = jnp.ones(3)
        with pytest.raises(ValueError, match="must be square"):
            solver.solve(A, b)


# ============================================================================
# matrix_multiply_and_accumulate テスト
# ============================================================================

class TestMatrixMultiplyAndAccumulate:
    """matrix_multiply_and_accumulate() の検証。"""

    def test_single_matrix(self):
        """単一行列での返却値が元の行列であることを検証。"""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        result = matrix_multiply_and_accumulate([A])
        assert jnp.allclose(result, A)

    def test_two_matrices_multiplication(self):
        """2 つの行列の積が正しく計算されることを検証。"""
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        result = matrix_multiply_and_accumulate([A, B])
        expected = A @ B
        assert jnp.allclose(result, expected)

    def test_three_matrices_multiplication(self):
        """3 つの行列の連続積が正しく計算されることを検証。"""
        A = jnp.eye(2)
        B = jnp.array([[2.0, 0.0], [0.0, 3.0]])
        C = jnp.array([[1.0, 1.0], [1.0, 1.0]])
        result = matrix_multiply_and_accumulate([A, B, C])
        expected = A @ B @ C
        assert jnp.allclose(result, expected)

    def test_raises_for_empty_list(self):
        """空リストで ValueError を raise することを検証。"""
        with pytest.raises(ValueError, match="must not be empty"):
            matrix_multiply_and_accumulate([])

    def test_raises_for_incompatible_shapes(self):
        """互換性のない行列形状で ValueError を raise することを検証。"""
        A = jnp.ones((2, 3))
        B = jnp.ones((4, 5))  # A @ B は計算不可
        with pytest.raises(Exception):  # 行列積エラー
            matrix_multiply_and_accumulate([A, B])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
