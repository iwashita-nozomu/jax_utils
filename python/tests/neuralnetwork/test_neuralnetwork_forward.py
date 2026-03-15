from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp

from jax_util.base import DEFAULT_DTYPE
from jax_util.neuralnetwork import build_neuralnetwork


SOURCE_FILE = Path(__file__).name


def _log_case(case: str, payload: dict[str, object]) -> None:
    print(json.dumps({
        "case": case,
        "source_file": SOURCE_FILE,
        **payload,
    }))


def test_neuralnetwork_standard_forward() -> None:
    """標準 NN の forward が実行できることを確認します。"""
    key = jax.random.PRNGKey(0)
    model = build_neuralnetwork(
        network_type="standard",
        layer_sizes=(2, 3, 1),
        activation="tanh",
        random_key=key,
    )
    x = jnp.ones((2, 4), dtype=DEFAULT_DTYPE)
    y = model(x)
    _log_case("nn_forward_standard", {
        "x_shape": x.shape,
        "y_shape": y.shape,
        "y_dtype": str(y.dtype),
    })
    assert y.shape == (1, 4)


def test_neuralnetwork_icnn_forward() -> None:
    """ICNN の forward が実行できることを確認します。"""
    key = jax.random.PRNGKey(1)
    model = build_neuralnetwork(
        network_type="icnn",
        layer_sizes=(2, 4, 1),
        activation="softplus",
        random_key=key,
    )
    x = jnp.ones((2, 3), dtype=DEFAULT_DTYPE)
    y = model(x)
    _log_case("nn_forward_icnn", {
        "x_shape": x.shape,
        "y_shape": y.shape,
        "y_dtype": str(y.dtype),
    })
    assert y.shape == (1, 3)


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_neuralnetwork_standard_forward()
    test_neuralnetwork_icnn_forward()


if __name__ == "__main__":
    _run_all_tests()
