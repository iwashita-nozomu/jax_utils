from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from jax_util.base import DEFAULT_DTYPE, EPS, LinOp, Vector

SOURCE_FILE = Path(__file__).name


def test_base_exports_basic() -> None:
    """base の主要エクスポートが機能することを確認します。"""

    def mv(v: Vector) -> Vector:
        return v

    op = LinOp(mv)
    x = jnp.ones((2,), dtype=DEFAULT_DTYPE)
    y = op @ x
    print(
        json.dumps(
            {
                "case": "base_exports",
                "source_file": SOURCE_FILE,
                "test": "test_base_exports_basic",
                "expected": x.tolist(),
                "y": y.tolist(),
                "eps": float(EPS),
            }
        )
    )
    assert y.shape == x.shape
    assert EPS > 0


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_base_exports_basic()


if __name__ == "__main__":
    _run_all_tests()
