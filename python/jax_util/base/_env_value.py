import os

import jax

from .protocols import Scalar


def _get_bool_env(name: str, default: bool) -> bool:
    """環境変数を bool として取得します。"""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    return default


def _get_float_env(name: str, default: float) -> float:
    """環境変数を float として取得します。"""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# 共通 dtype
jax.config.update("jax_enable_x64", _get_bool_env("JAX_UTIL_ENABLE_X64", True))

import jax.numpy as jnp

FP32 = jnp.float32
_dtype_name = os.getenv("JAX_UTIL_DEFAULT_DTYPE", "float64").strip().lower()
if _dtype_name in ("float32", "f32"):
    DEFAULT_DTYPE = jnp.float32
else:
    DEFAULT_DTYPE = jnp.float64

# スカラー定数（JAX Array として定義）
ZERO: Scalar = jnp.asarray(0.0, dtype=DEFAULT_DTYPE)
ONE: Scalar = jnp.asarray(1.0, dtype=DEFAULT_DTYPE)
HALF: Scalar = jnp.asarray(0.5, dtype=DEFAULT_DTYPE)
WEAK_EPS: Scalar = jnp.asarray(_get_float_env("JAX_UTIL_WEAK_EPS", 1e-6), dtype=DEFAULT_DTYPE)
EPS: Scalar = jnp.asarray(_get_float_env("JAX_UTIL_EPS", 1e-12), dtype=DEFAULT_DTYPE)
AVOID_ZERO_DIV: Scalar = jnp.asarray(
    _get_float_env("JAX_UTIL_AVOID_ZERO_DIV", 1e-30),
    dtype=DEFAULT_DTYPE,
)

DEBUG: bool = _get_bool_env("JAX_UTIL_DEBUG", True)



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

