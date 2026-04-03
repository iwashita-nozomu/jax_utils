from __future__ import annotations

import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from experiment_runner.result_io import (
    append_jsonl_record,
    json_compatible,
    read_jsonl_records,
)


def test_json_compatible_handles_numpy_and_jax_values() -> None:
    normalized = json_compatible(
        {
            "scalar": np.float32(1.5),
            "scalar_array": np.asarray(7.0),
            "array": np.asarray([[1, 2], [3, 4]]),
            "jax_array": jnp.asarray([1.0, 2.0]),
            "jax_scalar": jnp.asarray(8.0),
            "items": (np.int64(3),),
        }
    )
    assert normalized == {
        "scalar": 1.5,
        "scalar_array": 7.0,
        "array": [[1, 2], [3, 4]],
        "jax_array": [1.0, 2.0],
        "jax_scalar": 8.0,
        "items": [3],
    }


def test_jsonl_helpers_append_and_read_records(tmp_path: Path) -> None:
    output_path = tmp_path / "records.jsonl"
    append_jsonl_record(output_path, {"value": np.int64(3)})
    append_jsonl_record(output_path, {"value": np.int64(4), "items": np.asarray([1, 2])})

    raw_lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line]
    assert raw_lines == [
        json.dumps({"value": 3}),
        json.dumps({"value": 4, "items": [1, 2]}),
    ]
    assert read_jsonl_records(output_path) == [
        {"value": 3},
        {"value": 4, "items": [1, 2]},
    ]
