# from typing import Optional
import jax
import jax.numpy as jnp

if __name__ == "__main__":
    from protocols import Scalar

else:
    from .protocols import Scalar

# 共通 dtype
jax.config.update("jax_enable_x64", True)


FP32=jnp.float32
DEFAULT_DTYPE = jnp.float64
# スカラー定数（JAX Array として定義）
ZERO: Scalar = jnp.asarray(0.0, dtype=DEFAULT_DTYPE)
ONE: Scalar = jnp.asarray(1.0, dtype=DEFAULT_DTYPE)
HALF: Scalar = jnp.asarray(0.5, dtype=DEFAULT_DTYPE)
WEAK_EPS: Scalar = jnp.asarray(1e-6, dtype=DEFAULT_DTYPE)
EPS: Scalar = jnp.asarray(1e-12, dtype=DEFAULT_DTYPE)#1e-12
AVOID_ZERO_DIV: Scalar = jnp.asarray(1e-30, dtype=DEFAULT_DTYPE)

DEBUG: bool = True



__all__ = [
    
    "DEFAULT_DTYPE",
    "ZERO",
    "ONE",
    "HALF",
    "WEAK_EPS",
    "EPS",
    "AVOID_ZERO_DIV",
    "DEBUG",
]


if __name__ == "__main__":
    def test_constants_shape() -> None:
        assert ZERO.shape == ()
        assert ONE.shape == ()
        assert HALF.shape == ()
        assert WEAK_EPS.shape == ()
        assert EPS.shape == ()
        assert AVOID_ZERO_DIV.shape == ()

    def test_constants_dtype() -> None:
        assert ZERO.dtype == DEFAULT_DTYPE
        assert ONE.dtype == DEFAULT_DTYPE
        assert HALF.dtype == DEFAULT_DTYPE

    test_constants_shape()
    test_constants_dtype()

