"""
JAX ユーティリティ：数値計算モジュール
テスト用サンプル - 意図的に複数の問題を含める
"""

import jax
import jax.numpy as jnp
from typing import Tuple


def matrix_multiply(A, B):
    """行列掛け算"""
    # 問題1: 戻り値型の説明がない
    # 問題2: 引き数の型注釈がない
    result = jnp.dot(A, B)
    return result


def eigenvalue_decompose(matrix):
    # 問題3: docstring がない
    # 問題4: 型注釈がない
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    return eigenvalues, eigenvectors


def solve_linear_system(A: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """
    連立一次方程式を解く
    
    Args:
        A: 係数行列 (n, n)
        b: 右辺ベクトル (n,) または (n, m)
    
    Returns:
        解ベクトル x
    
    Raises:
        # 問題5: 特異行列の場合の例外説明がない
    """
    try:
        x = jnp.linalg.solve(A, b)
    except Exception:
        # 問題6: 単なる Exception を catch している
        return None
    
    return x


def normalize_columns(matrix: jnp.ndarray):
    """
    行列の列を単位ベクトルに正規化する
    
    Args:
        matrix: 入力行列 (m, n)
    
    Problem7: Returns type がない
    """
    # 問題8: type hint との矛盾
    norms = jnp.linalg.norm(matrix, axis=0)
    result = matrix / (norms + 1e-8)
    return result


@jax.jit
def compute_frobenius_norm(A):
    """
    Frobenius ノルムを計算
    
    Problem9: Args/Returns 説明が不完全
    """
    return jnp.sqrt(jnp.sum(A ** 2))


def apply_activation(x, activation_type):  # Problem10: type hint なし
    """活性化関数を適用"""
    
    if activation_type == "relu":
        return jnp.maximum(0, x)
    elif activation_type == "sigmoid":
        # Problem11: jnp.sigmoid は存在しない（手実装が必要）
        return 1.0 / (1.0 + jnp.exp(-x))
    else:
        # Problem12: 例外処理がない
        return x


# Problem13: テストコードが存在しない（モジュール自体にテスト不在）

if __name__ == "__main__":
    # Problem14: 簡便な動作確認も不足
    A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    b = jnp.array([1.0, 2.0])
    # 呼び出しのみで結果確認なし
    _ = solve_linear_system(A, b)
