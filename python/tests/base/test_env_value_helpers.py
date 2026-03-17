from __future__ import annotations

import importlib

import jax.numpy as jnp
import pytest

import jax_util.base._env_value as env_value


@pytest.mark.parametrize(
    ("raw_value", "default", "expected"),
    [
        ("true", False, True),
        ("On", False, True),
        ("0", True, False),
        ("off", True, False),
    ],
)
def test_get_bool_env_parses_common_spellings(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
    default: bool,
    expected: bool,
) -> None:
    monkeypatch.setenv("JAX_UTIL_TEST_BOOL", raw_value)

    assert env_value._get_bool_env("JAX_UTIL_TEST_BOOL", default) is expected
    assert env_value.get_bool_env("JAX_UTIL_TEST_BOOL", default) is expected


def test_get_bool_and_float_env_fall_back_for_missing_or_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JAX_UTIL_TEST_BOOL", raising=False)
    monkeypatch.delenv("JAX_UTIL_TEST_FLOAT", raising=False)
    assert env_value._get_bool_env("JAX_UTIL_TEST_BOOL", True) is True
    assert env_value._get_float_env("JAX_UTIL_TEST_FLOAT", 1.25) == pytest.approx(1.25)

    monkeypatch.setenv("JAX_UTIL_TEST_BOOL", "maybe")
    monkeypatch.setenv("JAX_UTIL_TEST_FLOAT", "not-a-float")
    assert env_value._get_bool_env("JAX_UTIL_TEST_BOOL", False) is False
    assert env_value._get_float_env("JAX_UTIL_TEST_FLOAT", 2.5) == pytest.approx(2.5)

    monkeypatch.setenv("JAX_UTIL_TEST_FLOAT", "3.75")
    assert env_value._get_float_env("JAX_UTIL_TEST_FLOAT", 0.0) == pytest.approx(3.75)


@pytest.mark.parametrize(
    ("dtype_name", "expected_dtype"),
    [
        ("float16", jnp.float16),
        ("bf16", jnp.bfloat16),
        ("f32", jnp.float32),
    ],
)
def test_reload_env_module_applies_dtype_aliases(
    monkeypatch: pytest.MonkeyPatch,
    dtype_name: str,
    expected_dtype: jnp.dtype,
) -> None:
    monkeypatch.setenv("JAX_UTIL_DEFAULT_DTYPE", dtype_name)
    reloaded = importlib.reload(env_value)
    assert reloaded.DEFAULT_DTYPE == expected_dtype

    monkeypatch.delenv("JAX_UTIL_DEFAULT_DTYPE", raising=False)
    importlib.reload(env_value)
