from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Mapping, cast

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import pytest

from jax_util.experiment_runner import (
    WorkerSlot,
    build_worker_slots,
    json_compatible,
    run_cases_with_subprocess_scheduler,
)

SOURCE_FILE = Path(__file__).name


def _discover_gpu_indices() -> list[int]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    return [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]


def _parent_failure_result(
    case: Mapping[str, object],
    worker_slot: WorkerSlot,
    failure_kind: str,
    error: str,
    traceback_text: str,
    /,
) -> dict[str, object]:
    return {
        "status": "failed",
        "case_id": _result_int(case, "case_id"),
        "worker_label": worker_slot.worker_label,
        "assigned_gpu_index": worker_slot.gpu_index,
        "failure_kind": failure_kind,
        "error": error,
        "traceback": traceback_text,
    }


def _result_int(result: Mapping[str, object], key: str, /) -> int:
    value = result.get(key)
    if not isinstance(value, int):
        raise TypeError(f"{key} must be int.")
    return value


def _result_float(result: Mapping[str, object], key: str, /) -> float:
    value = result.get(key)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be float.")
    return float(value)


def _result_str(result: Mapping[str, object], key: str, /) -> str:
    value = result.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be str.")
    return value


def _result_int_list(result: Mapping[str, object], key: str, /) -> list[int]:
    value = result.get(key)
    if not isinstance(value, list):
        raise TypeError(f"{key} must be list.")
    if not all(isinstance(item, int) for item in value):
        raise TypeError(f"{key} must contain ints.")
    return cast(list[int], value)


def _skip_if_gpu_backend_unavailable(results: list[Mapping[str, object]]) -> None:
    failed_results = [result for result in results if result.get("status") != "ok"]
    if not failed_results:
        return

    unavailable_markers = (
        "CUDA_ERROR_OUT_OF_MEMORY",
        "Unable to initialize backend 'cuda'",
        "no supported devices found for platform CUDA",
    )
    if all(
        isinstance(result.get("traceback"), str)
        and any(marker in result["traceback"] for marker in unavailable_markers)
        for result in failed_results
    ):
        pytest.skip("GPU backend was visible but unavailable for child workers during this run.")


def test_subprocess_scheduler_dispatches_heavy_cases_to_multiple_gpus(tmp_path: Path) -> None:
    gpu_indices = _discover_gpu_indices()
    if len(gpu_indices) < 2:
        pytest.skip("requires at least two GPUs")

    selected_gpu_indices = gpu_indices[:2]
    worker_slots = build_worker_slots("gpu", selected_gpu_indices, 1)
    child_script = Path(__file__).with_name("_gpu_child_probe.py")
    jsonl_output_path = tmp_path / "runner_gpu_probe.jsonl"
    run_config = {
        "platform": "gpu",
        "disable_gpu_preallocation": True,
    }
    cases = [
        {"case_id": 0, "matrix_size": 1024, "min_work_seconds": 2.5},
        {"case_id": 1, "matrix_size": 1024, "min_work_seconds": 2.5},
    ]

    results = run_cases_with_subprocess_scheduler(
        cases,
        worker_slots,
        timeout_seconds=120,
        build_child_command=lambda case, worker_slot: [
            sys.executable,
            str(child_script),
            "--case-json",
            json.dumps(json_compatible(case), ensure_ascii=True),
            "--run-config-json",
            json.dumps(json_compatible(run_config), ensure_ascii=True),
            "--worker-slot-json",
            json.dumps(json_compatible(worker_slot.to_dict()), ensure_ascii=True),
            "--jsonl-output",
            str(jsonl_output_path),
        ],
        build_parent_failure_result=_parent_failure_result,
        fallback_jsonl_output_path=jsonl_output_path,
        cwd=Path(__file__).resolve().parents[3],
    )

    assert len(results) == len(cases)
    _skip_if_gpu_backend_unavailable(results)
    assert all(_result_str(result, "status") == "ok" for result in results)
    assert {_result_int(result, "assigned_gpu_index") for result in results} == set(
        selected_gpu_indices
    )
    assert all(_result_int(result, "gpu_device_count") == 1 for result in results)
    assert all(_result_int_list(result, "visible_gpu_ids") == [0] for result in results)
    assert all(_result_float(result, "work_seconds") >= 2.0 for result in results)

    jsonl_lines = [
        line for line in jsonl_output_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(jsonl_lines) == len(cases)

    print(
        json.dumps(
            {
                "case": "experiment_runner_multi_gpu_heavy_scheduler",
                "source_file": SOURCE_FILE,
                "test": "test_subprocess_scheduler_dispatches_heavy_cases_to_multiple_gpus",
                "selected_gpu_indices": selected_gpu_indices,
                "worker_labels": [_result_str(result, "worker_label") for result in results],
                "work_seconds": [_result_float(result, "work_seconds") for result in results],
                "iterations": [_result_int(result, "iterations") for result in results],
            }
        )
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-s"]))
