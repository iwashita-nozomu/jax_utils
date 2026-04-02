from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from experiment_runner.subprocess_scheduler import (
    CHILD_COMPLETE_PREFIX,
    WorkerSlot,
    _available_cpu_indices,
    _extract_completion_record,
    _mapping_int,
    _mapping_str,
    _partition_cpu_indices,
    _terminate_process,
    append_jsonl_record,
    apply_worker_environment,
    build_worker_slots,
    json_compatible,
    run_cases_with_subprocess_scheduler,
    worker_slot_from_mapping,
)


def _slot(
    *,
    label: str = "gpu-0-w0",
    gpu_index: int | None = 0,
    gpu_slot: int = 0,
    cpu_affinity: tuple[int, ...] = (0, 1),
) -> WorkerSlot:
    return WorkerSlot(
        worker_label=label,
        gpu_index=gpu_index,
        gpu_slot=gpu_slot,
        cpu_affinity=cpu_affinity,
    )


def _write_script(path: Path, body: str) -> Path:
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def _failure_result(
    case: dict[str, object],
    worker_slot: WorkerSlot,
    failure_kind: str,
    error: str,
    traceback_text: str,
) -> dict[str, object]:
    return {
        "status": "failed",
        "case_id": int(case["case_id"]),
        "worker_label": worker_slot.worker_label,
        "failure_kind": failure_kind,
        "error": error,
        "traceback": traceback_text,
    }


def test_worker_slot_mapping_and_json_helpers_cover_scalar_and_array_paths() -> None:
    worker_slot = worker_slot_from_mapping(
        {
            "worker_label": "gpu-2-w0",
            "gpu_index": 2,
            "gpu_slot": 0,
            "cpu_affinity": [1, 3],
        }
    )
    assert worker_slot == _slot(label="gpu-2-w0", gpu_index=2, cpu_affinity=(1, 3))

    with pytest.raises(TypeError, match="must be int"):
        _mapping_int({"value": "x"}, "value")
    with pytest.raises(TypeError, match="must be str"):
        _mapping_str({"value": 1}, "value")

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


def test_partition_cpu_indices_and_build_worker_slots_cover_edge_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)
    assert _available_cpu_indices() == [0, 1, 2]

    monkeypatch.setattr(
        "experiment_runner.subprocess_scheduler._available_cpu_indices",
        lambda: [],
    )
    assert _partition_cpu_indices(2) == [[], []]

    monkeypatch.setattr(
        "experiment_runner.subprocess_scheduler._available_cpu_indices",
        lambda: [0, 1],
    )
    assert _partition_cpu_indices(4) == [[0], [1], [0], [1]]

    with pytest.raises(ValueError, match="positive"):
        _partition_cpu_indices(0)
    with pytest.raises(ValueError, match="positive"):
        build_worker_slots("gpu", [0], 0)

    gpu_slots = build_worker_slots("gpu", [2, 3], 1)
    cpu_slots = build_worker_slots("cpu", [], 1)

    assert [slot.worker_label for slot in gpu_slots] == ["gpu-2-w0", "gpu-3-w0"]
    assert cpu_slots == [_slot(label="cpu-0", gpu_index=None, cpu_affinity=(0, 1))]


def test_append_jsonl_record_and_apply_worker_environment_set_expected_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "records.jsonl"
    append_jsonl_record(output_path, {"value": np.int64(3)})
    assert output_path.read_text(encoding="utf-8").strip() == json.dumps({"value": 3})

    affinity_calls: list[tuple[int, set[int]]] = []
    monkeypatch.setattr(
        os,
        "sched_setaffinity",
        lambda pid, cpus: affinity_calls.append((pid, set(cpus))),
    )
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    apply_worker_environment(
        platform="gpu",
        worker_slot=_slot(gpu_index=4),
        disable_gpu_preallocation=True,
    )
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4"
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == "4"
    assert os.environ["SMOLYAK_GPU_INDEX"] == "4"
    assert os.environ["EXPERIMENT_RUNNER_WORKER_LABEL"] == "gpu-0-w0"
    assert affinity_calls[-1] == (0, {0, 1})

    apply_worker_environment(
        platform="cpu",
        worker_slot=_slot(label="cpu-0", gpu_index=None),
        disable_gpu_preallocation=False,
    )
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert os.environ["NVIDIA_VISIBLE_DEVICES"] == ""
    assert os.environ["SMOLYAK_GPU_INDEX"] == "cpu"


def test_completion_record_extraction_and_process_termination_cover_all_paths(
    tmp_path: Path,
) -> None:
    assert _extract_completion_record("plain output") is None
    record = _extract_completion_record(
        "\n".join(
            [
                f"{CHILD_COMPLETE_PREFIX}{json.dumps({'status': 'old'})}",
                f"{CHILD_COMPLETE_PREFIX}{json.dumps({'status': 'new'})}",
            ]
        )
    )
    assert record == {"status": "new"}

    terminates_script = _write_script(
        tmp_path / "terminates.py",
        """
        import time
        time.sleep(30)
        """,
    )
    terminates_process = subprocess.Popen(
        [sys.executable, str(terminates_script)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout_text, stderr_text = _terminate_process(terminates_process)
    assert stdout_text == ""
    assert stderr_text == ""

    class _FakeProcess:
        def __init__(self) -> None:
            self.killed = False
            self.terminated = False
            self.calls = 0

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            self.calls += 1
            if timeout is not None and self.calls == 1:
                raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)
            return ("", "")

        def kill(self) -> None:
            self.killed = True

    fake_process = _FakeProcess()
    stdout_text, stderr_text = _terminate_process(fake_process)  # type: ignore[arg-type]
    assert stdout_text == ""
    assert stderr_text == ""
    assert fake_process.terminated
    assert fake_process.killed


def test_run_cases_with_subprocess_scheduler_reports_success_and_callbacks(
    tmp_path: Path,
) -> None:
    success_script = _write_script(
        tmp_path / "success_child.py",
        f"""
        import json
        import sys

        print({CHILD_COMPLETE_PREFIX!r} + json.dumps({{
            "status": "ok",
            "case_id": int(sys.argv[1]),
            "worker_label": sys.argv[2],
        }}), flush=True)
        """,
    )
    started: list[tuple[int, str]] = []
    finished: list[tuple[int, str, str]] = []
    worker_slots = [_slot(label="cpu-0", gpu_index=None, cpu_affinity=())]
    cases = [{"case_id": 1}, {"case_id": 2}]

    results = run_cases_with_subprocess_scheduler(
        cases,
        worker_slots,
        timeout_seconds=5,
        build_child_command=lambda case, worker_slot: [
            sys.executable,
            str(success_script),
            str(case["case_id"]),
            worker_slot.worker_label,
        ],
        build_parent_failure_result=_failure_result,
        fallback_jsonl_output_path=None,
        cwd=tmp_path,
        on_case_started=lambda case, worker_slot: started.append(
            (int(case["case_id"]), worker_slot.worker_label)
        ),
        on_case_finished=lambda case, worker_slot, result: finished.append(
            (int(case["case_id"]), worker_slot.worker_label, str(result["status"]))
        ),
    )

    assert results == [
        {"status": "ok", "case_id": 1, "worker_label": "cpu-0"},
        {"status": "ok", "case_id": 2, "worker_label": "cpu-0"},
    ]
    assert started == [(1, "cpu-0"), (2, "cpu-0")]
    assert finished == [(1, "cpu-0", "ok"), (2, "cpu-0", "ok")]


def test_run_cases_with_subprocess_scheduler_handles_empty_slots_failures_and_timeouts(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        run_cases_with_subprocess_scheduler(
            [],
            [],
            timeout_seconds=1,
            build_child_command=lambda _case, _slot: [],
            build_parent_failure_result=_failure_result,
            fallback_jsonl_output_path=None,
            cwd=tmp_path,
        )

    fail_script = _write_script(
        tmp_path / "fail_child.py",
        """
        import sys
        print("boom", file=sys.stderr)
        raise SystemExit(1)
        """,
    )
    timeout_script = _write_script(
        tmp_path / "timeout_child.py",
        """
        import time
        time.sleep(30)
        """,
    )
    fallback_output = tmp_path / "fallback.jsonl"
    worker_slots = [_slot(label="cpu-0", gpu_index=None, cpu_affinity=())]
    timeout_finished: list[tuple[int, str]] = []

    failed_results = run_cases_with_subprocess_scheduler(
        [{"case_id": 1}],
        worker_slots,
        timeout_seconds=5,
        build_child_command=lambda _case, _slot: [sys.executable, str(fail_script)],
        build_parent_failure_result=_failure_result,
        fallback_jsonl_output_path=fallback_output,
        cwd=tmp_path,
    )
    assert failed_results[0]["failure_kind"] == "worker_terminated"
    assert "boom" in str(failed_results[0]["traceback"])

    timeout_results = run_cases_with_subprocess_scheduler(
        [{"case_id": 2}],
        worker_slots,
        timeout_seconds=0,
        build_child_command=lambda _case, _slot: [sys.executable, str(timeout_script)],
        build_parent_failure_result=_failure_result,
        fallback_jsonl_output_path=fallback_output,
        cwd=tmp_path,
        on_case_finished=lambda case, _worker_slot, result: timeout_finished.append(
            (int(case["case_id"]), str(result["failure_kind"]))
        ),
        poll_interval_seconds=0.0,
    )
    assert timeout_results[0]["failure_kind"] == "timeout"
    assert "TimeoutError" in str(timeout_results[0]["error"])
    assert timeout_finished == [(2, "timeout")]

    lines = [
        line
        for line in fallback_output.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 2


def _run_all_tests() -> None:
    """Run the unit test module directly."""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    _run_all_tests()
