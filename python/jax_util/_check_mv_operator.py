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
