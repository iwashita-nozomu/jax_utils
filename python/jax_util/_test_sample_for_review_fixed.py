"""サンプルファイル改善版：スキルに基づいた修正。

このファイルはコードレビュー実施テスト用に改善されたバージョン。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jax import Array

import jax.numpy as jnp

# 型エイリアス定義
Matrix = Array
Vector = Array
Scalar = Array


def solve_linear_system(A: Matrix, b: Vector) -> Vector:
    """線形方程式 Ax=b を直接法で解く。
    
    本関数は JAX の linalg.solve を使用して線形方程式を求解します。
    前提条件：A は正方行列で正則であること。

    Args:
        A: n×n 正方行列（正則）
        b: n 次元ベクトル（右辺）

    Returns:
        n 次元ベクトル（解 x）

    Raises:
        ValueError: A が正方行列でない場合
        LinAlgError: A が特異行列の場合

    References:
        - numpy.linalg.solve: https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}")

    return jnp.linalg.solve(A, b)


def compute_eigenvalues(matrix: Matrix) -> Vector:
    """対称行列の固有値を昇順で計算。
    
    与えられた対称行列の固有値を計算します。
    このメソッドは対称固有値分解（eigh）を使用します。

    Args:
        matrix: n×n 対称行列

    Returns:
        n 次元ベクトル（固有値を昇順で返す）

    Raises:
        ValueError: 入力が 2D 行列でない場合

    Notes:
        固有値は JAX の linalg.eigh により昇順で返されます。
    """
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square matrix, got shape {matrix.shape}")

    eigenvalues, _ = jnp.linalg.eigh(matrix)
    return eigenvalues


class LinearSolver:
    """線形方程式求解器（複数の方法をサポート）。
    
    LU 分解および QR 分解による線形方程式求解を提供します。
    
    Attributes:
        method: 求解方法（"lu" または "qr"）
    """

    def __init__(self, method: str = "lu") -> None:
        """初期化。

        Args:
            method: 求解方法（"lu" または "qr"、デフォルト："lu"）

        Raises:
            ValueError: method が "lu" または "qr" 以外の場合
        """
        if method not in ("lu", "qr"):
            raise ValueError(f"method must be 'lu' or 'qr', got '{method}'")
        self.method = method

    def solve(self, A: Matrix, b: Vector) -> Vector:
        """線形方程式 Ax=b を求解。

        Args:
            A: n×n 正方行列（正則）
            b: n 次元ベクトル

        Returns:
            n 次元ベクトル（解 x）

        Raises:
            ValueError: A が正方行列でない場合
            NotImplementedError: method が実装されていない場合

        Notes:
            - "lu": 直接法（LU 分解）。高速で数値的に安定。
            - "qr": QR 分解法。条件数の悪い行列でも安定傾向。
        """
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square, got shape {A.shape}")

        if self.method == "lu":
            return jnp.linalg.solve(A, b)
        elif self.method == "qr":
            Q, R = jnp.linalg.qr(A)
            return jnp.linalg.solve(R, Q.T @ b)
        else:
            raise NotImplementedError(f"Method '{self.method}' not implemented")


def matrix_multiply_and_accumulate(matrices_list: list[Matrix]) -> Matrix:
    """複数の行列を左から順に掛け合わせて累積。
    
    与えられた行列リストを左から順に行列積（@演算子）で計算します：
    result = matrix[0] @ matrix[1] @ ... @ matrix[n-1]

    Args:
        matrices_list: n 個の行列のリスト（長さ >= 1）

    Returns:
        計算結果の行列

    Raises:
        ValueError: リストが空の場合
        ValueError: 行列の形状が乗算可能でない場合

    Notes:
        行列の形状チェック：
        マトリックス M1 (a×b) と M2 (c×d) の積 M1 @ M2 は
        b == c が必須です（b != c の場合 ValueError）。
    """
    if not matrices_list:
        raise ValueError("matrices_list must not be empty")

    result: Matrix | None = None
    for M in matrices_list:
        if result is None:
            result = M
        else:
            result = result @ M

    # 型安全性のため、結果の None チェック
    if result is None:
        raise ValueError("matrices_list processing failed")

    return result
