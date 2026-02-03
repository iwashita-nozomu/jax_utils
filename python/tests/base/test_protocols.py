from __future__ import annotations

import json

from jax_util.base.protocols import LinearOperator, Operator


def test_protocols_exist() -> None:
    """プロトコルが import できることを確認します。"""
    assert LinearOperator is not None
    assert Operator is not None
    print(json.dumps({
        "case": "protocols",
        "linear_operator": True,
        "operator": True,
    }))


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_protocols_exist()


if __name__ == "__main__":
    _run_all_tests()
