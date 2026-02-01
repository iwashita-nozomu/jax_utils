from jax import numpy as jnp

from _env_value import ONE
from _type_aliaces import Matrix

def orthonormalize(X: Matrix) -> Matrix:
    """列を QR で正規直交化."""
    Q, R = jnp.linalg.qr(X, mode="reduced")
    diag = jnp.diag(R)
    signs = jnp.sign(diag)
    one = jnp.array(ONE, dtype=X.dtype)
    signs = jnp.where(signs == 0, one, signs)
    Q = Q * signs#pyright: ignore[reportConstantRedefinition]
    return Q