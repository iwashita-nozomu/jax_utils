from __future__ import annotations

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

from jax_util.base import DEFAULT_DTYPE

try:
    from jax_util.neuralnetwork import build_neuralnetwork, train_step
    from jax_util.neuralnetwork.protocols import Params
except (ModuleNotFoundError, TypeError) as exc:
    if "NamedTuple" in str(exc) or "optimizers.protocols" in str(exc):
        pytest.skip(f"neuralnetwork module is not importable in this environment: {exc}", allow_module_level=True)
    raise


SOURCE_FILE = Path(__file__).name


def _log_case(case: str, payload: dict[str, object]) -> None:
    print(json.dumps({
        "case": case,
        "source_file": SOURCE_FILE,
        **payload,
    }))


def test_neuralnetwork_train_step() -> None:
    """train_step が 1 回動作することを確認します。"""
    key = jax.random.PRNGKey(2)
    model = build_neuralnetwork(
        network_type="standard",
        layer_sizes=(2, 2, 1),
        activation="relu",
        random_key=key,
    )
    x = jnp.ones((2, 5), dtype=DEFAULT_DTYPE)
    y = jnp.zeros((1, 5), dtype=DEFAULT_DTYPE)
    batch = jnp.vstack([x, y])

    params, static = eqx.partition(model, eqx.is_inexact_array)

    def loss_fn(p: Params, b: jnp.ndarray) -> jnp.ndarray:
        bx = b[:2, :]
        by = b[2:3, :]
        m = eqx.combine(p, static)
        pred = m(bx)
        return jnp.mean((pred - by) ** 2)

    optimizer = optax.sgd(0.1)
    opt_state = optimizer.init(params)

    params, opt_state, metrics = train_step(
        params=params,
        batch=batch,
        optimizer=optimizer,
        opt_state=opt_state,
        loss_fn=loss_fn,
    )
    _log_case("nn_train_step", {
        "loss": float(metrics["loss"]),
    })
    assert "loss" in metrics
    assert metrics["loss"].shape == ()


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_neuralnetwork_train_step()


if __name__ == "__main__":
    _run_all_tests()
