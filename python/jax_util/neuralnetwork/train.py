from __future__ import annotations

from typing import Dict

import jax
from jax import numpy as jnp
import optax

from ..base import DEBUG, Matrix, Scalar
from .protocols import LossFn, Params


def train_step(
    params: Params,
    batch: Matrix,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_fn: LossFn,
) -> tuple[Params, optax.OptState, Dict[str, Scalar]]:
    """1 step の学習を行います。"""
    loss_val, grads = jax.value_and_grad(loss_fn)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

    metrics = {
        "loss": loss_val,
    }
    if DEBUG:
        jax.debug.print("{metrics}", metrics=metrics)
    return params, opt_state, metrics


def train_loop(
    params: Params,
    batch: Matrix,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_fn: LossFn,
    num_steps: int,
) -> tuple[Params, optax.OptState]:
    """最小の学習ループを提供します。"""
    def step(
        carry: tuple[Params, optax.OptState],
        step_index: jnp.ndarray,
    ) -> tuple[tuple[Params, optax.OptState], None]:
        p, s = carry
        p, s, metrics = train_step(
            params=p,
            batch=batch,
            optimizer=optimizer,
            opt_state=s,
            loss_fn=loss_fn,
        )
        _ = step_index
        _ = metrics
        return (p, s), None

    (params, opt_state), _ = jax.lax.scan(
        step,
        (params, opt_state),
        jnp.arange(num_steps),
    )
    return params, opt_state


__all__ = [
    "train_step",
    "train_loop",
]
