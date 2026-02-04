from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from ..base import *
from pathlib import Path


SOURCE_FILE = Path(__file__).name

def check_self_adjoint(
    Mv: LinearOperator,
    shape: Tuple[int, ...],
    *,
    num_trials: int = 16,
) -> Dict[str, Any]:
    """自己随伴性の簡易レポートを返す。"""
    n = int(shape[0])
    if n <= 0:
        raise ValueError("Shape must have positive size for checks.")

    # 乱数は固定シードで生成し、検査の再現性を保ちます。
    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, num_trials * 2)

    max_err = ZERO
    for i in range(num_trials):
        x = jax.random.normal(keys[2 * i], (n,), dtype=DEFAULT_DTYPE)
        y = jax.random.normal(keys[2 * i + 1], (n,), dtype=DEFAULT_DTYPE)

        x_my = jnp.vdot(x, Mv @ y)
        y_mx = jnp.vdot(y, Mv @ x)
        denom = jnp.abs(x_my) + jnp.abs(y_mx) + AVOID_ZERO_DIV
        err = jnp.abs(x_my - y_mx) / denom
        max_err = jnp.maximum(max_err, err)

    ok = bool(max_err < WEAK_EPS)
    return {
        "ok": ok,
        "max_err": max_err,
        "num_trials": num_trials,
        "shape": shape,
    }


def check_spd_quadratic_form(
    Mv: LinearOperator,
    shape: Tuple[int, ...],
    *,
    num_trials: int = 16,
) -> Dict[str, Any]:
    """SPD 性の簡易レポートを返す。"""
    n = int(shape[0])
    if n <= 0:
        raise ValueError("Shape must have positive size for checks.")

    # 乱数は固定シードで生成し、検査の再現性を保ちます。
    key = jax.random.PRNGKey(1)
    keys = jax.random.split(key, num_trials)

    min_quad = jnp.asarray(jnp.inf, dtype=DEFAULT_DTYPE)
    for i in range(num_trials):
        x = jax.random.normal(keys[i], (n,), dtype=DEFAULT_DTYPE)
        quad = jnp.vdot(x, Mv @ x)
        min_quad = jnp.minimum(min_quad, quad)

    ok = bool(min_quad > WEAK_EPS)
    return {
        "ok": ok,
        "min_quad": min_quad,
        "num_trials": num_trials,
        "shape": shape,
    }


def print_Mv_report(
    self_adjoint_report: Dict[str, Any],
    spd_report: Optional[Dict[str, Any]],
    *,
    name: str,
) -> None:
    """Mv 検査結果を表示する。"""
    print(json.dumps({
        "case": "mv_report",
        "source_file": SOURCE_FILE,
        "func": "print_Mv_report",
        "event": "self_adjoint",
        "name": name,
        "report": self_adjoint_report,
    }))
    if spd_report is not None:
        print(json.dumps({
            "case": "mv_report",
            "source_file": SOURCE_FILE,
            "func": "print_Mv_report",
            "event": "spd",
            "name": name,
            "report": spd_report,
        }))


__all__ = [
    "check_self_adjoint",
    "check_spd_quadratic_form",
    "print_Mv_report",
]
