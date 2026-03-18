"""ヤコビアン計算・入力感度のテストモジュール。

検証項目:
  - compute_jacobian の形状確認
  - input_sensitivity の形状確認
  - ヤコビアンの数値的整合性（有限差分との比較）
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from jax_util.base import DEFAULT_DTYPE

try:
    from jax_util.neuralnetwork import build_neuralnetwork
    from jax_util.neuralnetwork.jacobian import compute_jacobian, input_sensitivity
except (ModuleNotFoundError, TypeError) as exc:
    if "NamedTuple" in str(exc) or "optimizers.protocols" in str(exc):
        pytest.skip(
            f"neuralnetwork module is not importable in this environment: {exc}",
            allow_module_level=True,
        )
    raise


SOURCE_FILE = Path(__file__).name


def _log_case(case: str, payload: dict[str, object]) -> None:
    """テスト結果を 1 行 JSON で標準出力に書き出します。"""
    print(json.dumps({"case": case, "source_file": SOURCE_FILE, **payload}))


# --------------------------------------------------------
# compute_jacobian のテスト
# --------------------------------------------------------


def test_jacobian_shape_standard() -> None:
    """標準 NN のヤコビアン形状を検証します。"""
    key = jax.random.PRNGKey(0)
    input_dim, hidden_dim, output_dim = 3, 5, 2
    batch_size = 4
    model = build_neuralnetwork(
        network_type="standard",
        layer_sizes=(input_dim, hidden_dim, output_dim),
        activation="tanh",
        random_key=key,
    )
    x = jax.random.normal(key, (input_dim, batch_size), dtype=DEFAULT_DTYPE)

    # ヤコビアン: (batch_size, output_dim, input_dim)
    J = compute_jacobian(model, x)
    expected_shape = (batch_size, output_dim, input_dim)

    _log_case("jacobian_shape_standard", {
        "expected_shape": list(expected_shape),
        "actual_shape": list(J.shape),
    })
    assert J.shape == expected_shape


def test_jacobian_shape_icnn() -> None:
    """ICNN のヤコビアン形状を検証します。"""
    key = jax.random.PRNGKey(1)
    input_dim, hidden_dim, output_dim = 2, 4, 1
    batch_size = 3
    model = build_neuralnetwork(
        network_type="icnn",
        layer_sizes=(input_dim, hidden_dim, output_dim),
        activation="softplus",
        random_key=key,
    )
    x = jax.random.normal(key, (input_dim, batch_size), dtype=DEFAULT_DTYPE)

    J = compute_jacobian(model, x)
    expected_shape = (batch_size, output_dim, input_dim)

    _log_case("jacobian_shape_icnn", {
        "expected_shape": list(expected_shape),
        "actual_shape": list(J.shape),
    })
    assert J.shape == expected_shape


def test_jacobian_finite_difference() -> None:
    """有限差分との比較でヤコビアンの数値的整合性を検証します。

    前向き有限差分: $J_{ij} \\approx (f(x + h e_j)_i - f(x)_i) / h$
    """
    key = jax.random.PRNGKey(42)
    input_dim, output_dim = 3, 2
    h = 1e-4  # 有限差分のステップ幅

    model = build_neuralnetwork(
        network_type="standard",
        layer_sizes=(input_dim, 4, output_dim),
        activation="tanh",
        random_key=key,
    )

    # バッチサイズ 1 で比較（単一サンプル）
    x = jax.random.normal(key, (input_dim, 1), dtype=DEFAULT_DTYPE)

    # 自動微分によるヤコビアン (1, output_dim, input_dim)
    J_auto = compute_jacobian(model, x)[0]  # (output_dim, input_dim)

    # 有限差分によるヤコビアン
    y0 = model(x)[:, 0]  # (output_dim,)
    J_fd = jnp.zeros((output_dim, input_dim), dtype=DEFAULT_DTYPE)
    for j in range(input_dim):
        e_j = jnp.zeros((input_dim, 1), dtype=DEFAULT_DTYPE).at[j, 0].set(h)
        y1 = model(x + e_j)[:, 0]  # (output_dim,)
        J_fd = J_fd.at[:, j].set((y1 - y0) / h)

    rel_err = float(jnp.linalg.norm(J_auto - J_fd) / (jnp.linalg.norm(J_fd) + 1e-10))

    _log_case("jacobian_finite_difference", {
        "rel_err": rel_err,
        "threshold": 1e-2,
        "h": h,
    })
    assert rel_err < 1e-2, f"有限差分との相対誤差が大きすぎます: {rel_err}"


# --------------------------------------------------------
# input_sensitivity のテスト
# --------------------------------------------------------


def test_input_sensitivity_shape() -> None:
    """感度行列の形状を検証します。"""
    key = jax.random.PRNGKey(2)
    input_dim, output_dim, batch_size = 4, 2, 6
    model = build_neuralnetwork(
        network_type="standard",
        layer_sizes=(input_dim, 5, output_dim),
        activation="relu",
        random_key=key,
    )
    x = jax.random.normal(key, (input_dim, batch_size), dtype=DEFAULT_DTYPE)

    s = input_sensitivity(model, x)
    expected_shape = (batch_size, input_dim)

    _log_case("input_sensitivity_shape", {
        "expected_shape": list(expected_shape),
        "actual_shape": list(s.shape),
    })
    assert s.shape == expected_shape


def test_input_sensitivity_nonnegative() -> None:
    """感度値が非負であることを確認します（L2 ノルムなので常に非負）。"""
    key = jax.random.PRNGKey(3)
    input_dim, output_dim = 3, 1
    model = build_neuralnetwork(
        network_type="standard",
        layer_sizes=(input_dim, 4, output_dim),
        activation="tanh",
        random_key=key,
    )
    x = jax.random.normal(key, (input_dim, 5), dtype=DEFAULT_DTYPE)

    s = input_sensitivity(model, x)
    min_val = float(jnp.min(s))

    _log_case("input_sensitivity_nonnegative", {
        "min_val": min_val,
    })
    assert min_val >= 0.0


def _run_all_tests() -> None:
    """このファイル内のテストをすべて順番に実行します。"""
    test_jacobian_shape_standard()
    test_jacobian_shape_icnn()
    test_jacobian_finite_difference()
    test_input_sensitivity_shape()
    test_input_sensitivity_nonnegative()
    print("All jacobian tests passed.")


if __name__ == "__main__":
    _run_all_tests()
