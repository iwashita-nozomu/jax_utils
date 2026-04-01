"""サンプルファイル：意図的に複数の問題を含める。

このファイルはコードレビュー実施テスト用。
以下の問題を含んでいます：
- 型注釈の欠落
- Docstring 不完全
- テストカバレッジ不足
"""

import jax.numpy as jnp
from typing import Optional, List


def solve_linear_system(A: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """線形方程式を解く。
    
    Args:
        A: 係数行列
        b: 右辺ベクトル
        
    Returns:
        解ベクトル
    """
    return jnp.linalg.solve(A, b)


def compute_eigenvalues(matrix: jnp.ndarray) -> jnp.ndarray:
    """固有値を計算。
    
    Args:
        matrix: 対象行列
        
    Returns:
        固有値
    """
    eigenvalues, _ = jnp.linalg.eigh(matrix)
    return eigenvalues


class LinearSolver:
    """線形方程式求解器。
    
    Notes:
        実装が未完成。
    """

    def __init__(self, method: str = "lu") -> None:
        self.method = method

    def solve(self, A: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """求解を実行。ただし前提条件チェックなし。
        
        Args:
            A: 係数行列
            b: 右辺ベクトル
            
        Returns:
            解
        """
        if self.method == "lu":
            return jnp.linalg.solve(A, b)
        elif self.method == "qr":
            Q, R = jnp.linalg.qr(A)
            return jnp.linalg.solve(R, Q.T @ b)
        else:
            raise ValueError(f"Unknown method: {self.method}")


def matrix_multiply_and_accumulate(matrices_list: List[jnp.ndarray]) -> Optional[jnp.ndarray]:
    """複数の行列を掛け合わせて累積。
    
    Args:
        matrices_list: 行列のリスト
        
    Returns:
        累積結果（空リストの場合は None）
        
    Raises:
        ValueError: リストが空の場合
    """
    result: Optional[jnp.ndarray] = None
    for M in matrices_list:
        if result is None:
            result = M
        else:
            result = result @ M

    return result
