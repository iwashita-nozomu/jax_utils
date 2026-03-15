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
