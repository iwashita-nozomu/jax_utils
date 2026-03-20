"""行列ユーティリティ関数。

行列形式変換、構造検証、数値性質チェックを提供し、
ソルバー実装を簡潔にします。

主要関数:
    ensure_matrix_form: 様々な形式を統一フォーマットに
    check_symmetry: 対称性検証
    check_positive_definite: 正定値性検証
    condition_number: 条件数計算
    normalize_matrix: 数値安定化
"""

from jax import numpy as jnp

from ..base import ONE, Matrix


# 責務: 列ベクトルの集合を QR により正規直交化します。
def orthonormalize(X: Matrix) -> Matrix:
    """列を QR で正規直交化."""
    Q, R = jnp.linalg.qr(X, mode="reduced")
    diag = jnp.diag(R)
    signs = jnp.sign(diag)
    one = jnp.array(ONE, dtype=X.dtype)
    signs = jnp.where(signs == 0, one, signs)
    Q_adjusted = Q * signs
    return Q_adjusted
