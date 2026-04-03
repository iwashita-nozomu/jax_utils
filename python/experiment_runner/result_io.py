"""JSON-compatible result helpers for experiment outputs."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

__all__ = [
    "append_jsonl_record",
    "json_compatible",
    "read_jsonl_records",
]


def json_compatible(value: object, /) -> object:
    """Convert array-like values into plain JSON-compatible Python objects."""
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        array_value = np.asarray(value)
        if array_value.ndim == 0:
            return array_value.item()
        return array_value.tolist()
    return value


def append_jsonl_record(output_path: Path, record: Mapping[str, object], /) -> None:
    """Append one JSON-compatible record to a JSONL file with file locking."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(json_compatible(dict(record)), ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_jsonl_records(output_path: Path, /) -> list[dict[str, Any]]:
    """Read JSONL records as dictionaries, skipping blank lines."""
    if not output_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise TypeError("JSONL record must decode to dict.")
        records.append(cast(dict[str, Any], parsed))
    return records
