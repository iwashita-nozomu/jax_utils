from __future__ import annotations

import json

from jax_util.base import AVOID_ZERO_DIV, DEFAULT_DTYPE, EPS, ONE, ZERO


def test_env_values_exist() -> None:
    """数値定数が取得できることを確認します。"""
    assert DEFAULT_DTYPE is not None
    assert EPS > ZERO
    assert ONE > ZERO
    assert AVOID_ZERO_DIV > ZERO
    print(json.dumps({
        "case": "env_value",
        "eps": float(EPS),
        "one": float(ONE),
        "avoid_zero_div": float(AVOID_ZERO_DIV),
    }))


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_env_values_exist()


if __name__ == "__main__":
    _run_all_tests()
