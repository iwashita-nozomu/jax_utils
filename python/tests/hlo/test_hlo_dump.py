from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import jax.numpy as jnp

from jax_util.hlo import dump_hlo_jsonl


def test_dump_hlo_jsonl_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """既定では HLO ダンプが無効で、ファイルが作られないことを確認します。"""
    monkeypatch.delenv("JAX_UTIL_ENABLE_HLO_DUMP", raising=False)

    out_path = tmp_path / "hlo.jsonl"

    def f(x: jnp.ndarray) -> jnp.ndarray:
        return x + 1

    dump_hlo_jsonl(f, jnp.ones((2, 3)), out_path=out_path, tag="test")
    assert not out_path.exists()


def test_dump_hlo_jsonl_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """フラグ有効時に JSONL が 1 行出力されることを確認します。"""
    monkeypatch.setenv("JAX_UTIL_ENABLE_HLO_DUMP", "1")

    out_path = tmp_path / "hlo.jsonl"

    def f(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.tanh(x) * 2

    dump_hlo_jsonl(f, jnp.ones((2, 3)), out_path=out_path, tag="test")

    assert out_path.exists()
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    rec: dict[str, Any] = json.loads(lines[0])
    assert rec["case"] == "hlo"
    assert rec["tag"] == "test"
    assert isinstance(rec["hlo"], str)
    assert len(rec["hlo"]) > 0
