from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from jax_util.Algorithms._check_mv_operator import (
    check_self_adjoint,
    check_spd_quadratic_form,
    print_Mv_report,
)
from jax_util.base import LinOp, Vector


SOURCE_FILE = Path(__file__).name


def test_check_mv_operator_reports() -> None:
    """簡易レポートが辞書で返ることを確認します。"""
    A = jnp.array([[2.0, 0.0], [0.0, 3.0]])

    def mv(v: Vector) -> Vector:
        return A @ v

    op = LinOp(mv)
    report1 = check_self_adjoint(op, shape=(2,), num_trials=2)
    report2 = check_spd_quadratic_form(op, shape=(2,), num_trials=2)

    assert isinstance(report1, dict)
    assert isinstance(report2, dict)
    assert "ok" in report1
    assert "ok" in report2

    # レポート出力関数が例外なく動作することも確認します。
    print_Mv_report(report1, report2, name="test")
    print(json.dumps({
        "case": "check_mv_operator",
        "source_file": SOURCE_FILE,
        "test": "test_check_mv_operator_reports",
        "expected_ok": True,
        "self_adjoint": report1,
        "spd": report2,
    }))


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_check_mv_operator_reports()


if __name__ == "__main__":
    _run_all_tests()
