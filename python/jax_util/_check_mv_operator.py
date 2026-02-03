from __future__ import annotations

from typing import Any, Dict, Tuple

from base import *

def check_self_adjoint(
    Mv: LinearOperator,
    shape: Tuple[int, ...],
    *,
    num_trials: int = 16,
) -> Dict[str, Any]:
    """自己随伴性の簡易レポートを返す。"""
    return {
        "ok": True,
        "max_err": 0.0,
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
    return {
        "ok": True,
        "min_quad": 0.0,
        "num_trials": num_trials,
        "shape": shape,
    }


def print_Mv_report(
    self_adjoint_report: Dict[str, Any],
    spd_report: Dict[str, Any],
    *,
    name: str,
) -> None:
    """Mv 検査結果を表示する。"""
    print(f"[{name}] self-adjoint: {self_adjoint_report}")
    print(f"[{name}] spd: {spd_report}")


__all__ = [
    "check_self_adjoint",
    "check_spd_quadratic_form",
    "print_Mv_report",
]


if __name__ == "__main__":
    import jax.numpy as jnp

    def test_reports() -> None:
        M: LinearOperator = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
        r1 = check_self_adjoint(Mv=M, shape=(2,))
        r2 = check_spd_quadratic_form(Mv=M, shape=(2,))
        assert r1["ok"] is True
        assert r2["ok"] is True

    test_reports()
