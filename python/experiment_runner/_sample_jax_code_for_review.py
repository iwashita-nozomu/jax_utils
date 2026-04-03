"""
JAX ユーティリティ：数値計算モジュール
テスト用サンプル - 型定義を完全に
"""

import jax
import jax.numpy as jnp


def matrix_multiply(A: jnp.ndarray, B: jnp.ndarray) -> jnp.ndarray:
    """
    行列掛け算
    
    Args:
        A: 第一行列
        B: 第二行列
    
    Returns:
        A と B の積
    """
    result = jnp.dot(A, B)
    return result


def eigenvalue_decompose(matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    行列の固有値分解を実行
    
    Args:
        matrix: 対称行列
    
    Returns:
        固有値と固有ベクトルのタプル
    """
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
        RuntimeError: 計算失敗時
    """
    try:
        x = jnp.linalg.solve(A, b)
    except (ValueError, RuntimeError) as e:
        # 具体的なエラーのみキャッチ - 型チェックはpyright に任せる
        raise RuntimeError(f"Linear solver failed: {e}") from e
    
    return x


def normalize_columns(matrix: jnp.ndarray) -> jnp.ndarray:
    """
    行列の列を単位ベクトルに正規化する
    
    Args:
        matrix: 入力行列 (m, n)
    
    Returns:
        正規化された行列 (m, n)
    """
    norms = jnp.linalg.norm(matrix, axis=0)
    result = matrix / (norms + 1e-8)
    return result


@jax.jit
def compute_frobenius_norm(A: jnp.ndarray) -> jnp.ndarray:
    """
    Frobenius ノルムを計算
    
    Args:
        A: 入力行列
    
    Returns:
        Frobenius ノルム値
    """
    return jnp.sqrt(jnp.sum(A ** 2))


def apply_activation(x: jnp.ndarray, activation_type: str) -> jnp.ndarray:
    """
    活性化関数を適用
    
    Args:
        x: 入力配列
        activation_type: 活性化関数の種類 ('relu', 'sigmoid')
    
    Returns:
        活性化後の配列
    
    Raises:
        ValueError: 未知の活性化関数型
    """
    
    if activation_type == "relu":
        return jnp.maximum(0, x)
    elif activation_type == "sigmoid":
        return 1.0 / (1.0 + jnp.exp(-x))
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")

