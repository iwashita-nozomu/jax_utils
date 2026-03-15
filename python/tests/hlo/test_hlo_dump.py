from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any

import pytest

import jax.numpy as jnp

from jax_util.hlo import dump_hlo_jsonl


SOURCE_FILE = Path(__file__).name


def _run_dump_hlo_jsonl_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """既定では HLO ダンプが無効で、ファイルが作られないことを確認します。"""
    monkeypatch.delenv("JAX_UTIL_ENABLE_HLO_DUMP", raising=False)

    out_path = tmp_path / "hlo.jsonl"

    def f(x: jnp.ndarray) -> jnp.ndarray:
        return x + 1

    dump_hlo_jsonl(f, jnp.ones((2, 3)), out_path=out_path, tag="test")
    print(json.dumps({
        "case": "hlo_dump_disabled",
        "source_file": SOURCE_FILE,
        "test": "test_dump_hlo_jsonl_disabled",
        "exists": out_path.exists(),
    }))
    assert not out_path.exists()


def test_dump_hlo_jsonl_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run_dump_hlo_jsonl_disabled(tmp_path, monkeypatch)


def _run_dump_hlo_jsonl_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
    print(json.dumps({
        "case": "hlo_dump_enabled",
        "source_file": SOURCE_FILE,
        "test": "test_dump_hlo_jsonl_enabled",
        "tag": rec["tag"],
        "line_count": len(lines),
    }))
    assert rec["case"] == "hlo"
    assert rec["tag"] == "test"
    assert isinstance(rec["hlo"], str)
    assert len(rec["hlo"]) > 0


def test_dump_hlo_jsonl_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _run_dump_hlo_jsonl_enabled(tmp_path, monkeypatch)


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        monkeypatch = pytest.MonkeyPatch()
        try:
            _run_dump_hlo_jsonl_disabled(tmp_path, monkeypatch)
            _run_dump_hlo_jsonl_enabled(tmp_path, monkeypatch)
        finally:
            monkeypatch.undo()


if __name__ == "__main__":
    _run_all_tests()
