from __future__ import annotations

import json

import pytest

from jax_util.hlo.dump import _get_hlo_text, _to_jsonable


class _FakeLowered:
    def __init__(self, ir_by_dialect: dict[str, object | Exception]) -> None:
        self._ir_by_dialect = ir_by_dialect

    def compiler_ir(self, *, dialect: str) -> object:
        value = self._ir_by_dialect[dialect]
        if isinstance(value, Exception):
            raise value
        return value


class _FakeJit:
    def __init__(self, ir_by_dialect: dict[str, object | Exception]) -> None:
        self._ir_by_dialect = ir_by_dialect

    def lower(self, *_args: object, **_kwargs: object) -> _FakeLowered:
        return _FakeLowered(self._ir_by_dialect)


def test_to_jsonable_normalizes_nested_non_serializable_values() -> None:
    normalized = _to_jsonable({"tuple": (1, object())})

    assert json.loads(json.dumps(normalized))["tuple"][0] == 1
    assert isinstance(normalized["tuple"][1], str)


def test_get_hlo_text_falls_back_to_hlo_and_raises_when_both_dialects_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "jax_util.hlo.dump.jax.jit",
        lambda _func: _FakeJit({"stablehlo": RuntimeError("no stablehlo"), "hlo": "HLO TEXT"}),
    )
    assert _get_hlo_text(lambda x: x, 1) == "HLO TEXT"

    monkeypatch.setattr(
        "jax_util.hlo.dump.jax.jit",
        lambda _func: _FakeJit({"stablehlo": RuntimeError("bad"), "hlo": RuntimeError("bad")}),
    )
    with pytest.raises(RuntimeError, match="Failed to get HLO text"):
        _get_hlo_text(lambda x: x, 1)


def _run_all_tests() -> None:
    """全テストを実行します。
    
    補助的なpython file.py実行時に使用されます。
    pytest -s python/tests/hlo/test_hlo_dump_helpers.py
    と同等の実行が可能になります。
    """
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    _run_all_tests()
