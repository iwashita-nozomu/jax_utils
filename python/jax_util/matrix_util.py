from jax import numpy as jnp

from base import ONE,Matrix


def orthonormalize(X: Matrix) -> Matrix:
    """列を QR で正規直交化."""
    Q, R = jnp.linalg.qr(X, mode="reduced")
    diag = jnp.diag(R)
    signs = jnp.sign(diag)
    one = jnp.array(ONE, dtype=X.dtype)
    signs = jnp.where(signs == 0, one, signs)
    Q = Q * signs#pyright: ignore[reportConstantRedefinition]
    return Q


if __name__ == "__main__":
    import jax.numpy as jnp

    def test_orthonormalize() -> None:
        X: Matrix = jnp.asarray([[1.0, 0.0], [0.0, 2.0]])
        Q = orthonormalize(X)
        ident = jnp.eye(2)
        assert jnp.allclose(Q.T @ Q, ident, rtol=1e-6, atol=1e-6)

    test_orthonormalize()