from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
from typing import Mapping, cast

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from experiment_runner.resource_scheduler import (
    GPUDeviceCapacity,
    FullResourceCapacity,
    FullResourceEstimate,
    StandardFullResourceScheduler,
    detect_gpu_devices,
    detect_host_memory_bytes,
    detect_max_workers,
)
from experiment_runner.runner import (
    StandardRunner,
    StandardWorker,
    SUCCESS_EXIT_CODE,
)
from experiment_runner.protocols import (
    TaskContext,
    WORKER_PROTOCOL_ERROR_EXIT_CODE,
)


SOURCE_FILE = Path(__file__).name


def _write_json(path: Path, record: Mapping[str, object], /) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(record, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )


def _read_json_records(directory: Path, /) -> list[dict[str, object]]:
    if not directory.exists():
        return []
    return [
        cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(directory.glob("*.json"))
    ]


def _case_id(case: Mapping[str, object], /) -> int:
    value = case.get("case_id")
    if not isinstance(value, int):
        raise TypeError("case_id must be int.")
    return value


def _sleep_seconds(case: Mapping[str, object], /) -> float:
    value = case.get("sleep_seconds")
    if not isinstance(value, (int, float)):
        raise TypeError("sleep_seconds must be numeric.")
    return float(value)


def _resource_estimate(case: Mapping[str, object], /) -> FullResourceEstimate:
    host_memory_bytes = case.get("host_memory_bytes", 0)
    gpu_count = case.get("gpu_count", 0)
    gpu_memory_bytes = case.get("gpu_memory_bytes", 0)
    if not isinstance(host_memory_bytes, int):
        raise TypeError("host_memory_bytes must be int.")
    if not isinstance(gpu_count, int):
        raise TypeError("gpu_count must be int.")
    if not isinstance(gpu_memory_bytes, int):
        raise TypeError("gpu_memory_bytes must be int.")
    return FullResourceEstimate(
        host_memory_bytes=host_memory_bytes,
        gpu_count=gpu_count,
        gpu_memory_bytes=gpu_memory_bytes,
    )


def _midpoints(records: list[dict[str, object]], /) -> list[float]:
    sorted_times = sorted(
        {
            cast(float, record["started_at"])
            for record in records
        }
        | {
            cast(float, record["finished_at"])
            for record in records
        }
    )
    return [
        (sorted_times[index] + sorted_times[index + 1]) / 2.0
        for index in range(len(sorted_times) - 1)
    ]


def _active_records_at(
    records: list[dict[str, object]],
    timestamp: float,
    /,
) -> list[dict[str, object]]:
    return [
        record
        for record in records
        if cast(float, record["started_at"]) <= timestamp < cast(float, record["finished_at"])
    ]


def _has_parallel_overlap(records: list[dict[str, object]], /) -> bool:
    intervals = [
        (
            cast(float, record["started_at"]),
            cast(float, record["finished_at"]),
        )
        for record in records
    ]
    for index, (left_start, left_end) in enumerate(intervals):
        for inner_index, (right_start, right_end) in enumerate(intervals):
            if index == inner_index:
                continue
            if left_start < right_end and right_start < left_end:
                return True
    return False


@dataclass(frozen=True)
class FullResourceRecordingTask:
    records_dir: Path

    def __call__(self, case: dict[str, object], context: TaskContext) -> dict[str, object]:
        case_id = _case_id(case)
        started_at = time.time()
        time.sleep(_sleep_seconds(case))
        finished_at = time.time()
        env_vars = cast(dict[str, str], context.get("environment_variables", {}))

        record: dict[str, object] = {
            "case_id": case_id,
            "pid": os.getpid(),
            "gpu_ids": env_vars.get("gpu_ids", ""),
            "cuda_visible_devices": env_vars.get("CUDA_VISIBLE_DEVICES", ""),
            "started_at": started_at,
            "finished_at": finished_at,
        }
        _write_json(self.records_dir / f"case_{case_id}.json", record)
        return record


@dataclass(frozen=True)
class EstimatedCaseTask:
    def __call__(self, case: dict[str, object], context: TaskContext) -> dict[str, object]:
        return {"case_id": _case_id(case), "context": dict(context)}

    def resource_estimate(self, case: Mapping[str, object], /) -> FullResourceEstimate:
        return _resource_estimate(case)


def _run_detected_full_resource_capacity() -> None:
    capacity = FullResourceCapacity.from_system(
        max_workers=detect_max_workers(12),
        host_memory_bytes=detect_host_memory_bytes(4096, 1024),
        environ={"NVIDIA_VISIBLE_DEVICES": "2, 5"},
        gpu_query_rows=[
            (0, 8 * 1024 * 1024 * 1024),
            (2, 24 * 1024 * 1024 * 1024),
            (5, 48 * 1024 * 1024 * 1024),
        ],
        gpu_max_slots=2,
    )

    assert capacity.max_workers == 12
    assert capacity.host_memory_bytes == int(4096 * 1024 * 0.8)
    assert capacity.gpu_devices == (
        GPUDeviceCapacity(
            gpu_id=2,
            memory_bytes=int(24 * 1024 * 1024 * 1024 * 0.8),
            max_slots=2,
        ),
        GPUDeviceCapacity(
            gpu_id=5,
            memory_bytes=int(48 * 1024 * 1024 * 1024 * 0.8),
            max_slots=2,
        ),
    )
    assert detect_gpu_devices(
        environ={"NVIDIA_VISIBLE_DEVICES": "all"},
        query_rows=[
            (0, 16 * 1024 * 1024 * 1024),
            (1, 24 * 1024 * 1024 * 1024),
        ],
    ) == (
        GPUDeviceCapacity(gpu_id=0, memory_bytes=int(16 * 1024 * 1024 * 1024 * 0.8), max_slots=1),
        GPUDeviceCapacity(gpu_id=1, memory_bytes=int(24 * 1024 * 1024 * 1024 * 0.8), max_slots=1),
    )

    print(
        json.dumps(
            {
                "case": "detected_full_resource_capacity",
                "source_file": SOURCE_FILE,
                "test": "test_detected_full_resource_capacity",
                "gpu_ids": [gpu_device.gpu_id for gpu_device in capacity.gpu_devices],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def test_detected_full_resource_capacity() -> None:
    _run_detected_full_resource_capacity()


def _run_standard_full_resource_scheduler_assigns_and_releases_resources() -> None:
    worker = StandardWorker(
        EstimatedCaseTask(),
        resource_estimator=EstimatedCaseTask().resource_estimate,
    )
    scheduler = StandardFullResourceScheduler.from_worker(
        resource_capacity=FullResourceCapacity(
            max_workers=4,
            host_memory_bytes=256,
            gpu_devices=(
                GPUDeviceCapacity(gpu_id=0, memory_bytes=24, max_slots=1),
                GPUDeviceCapacity(gpu_id=1, memory_bytes=24, max_slots=1),
                GPUDeviceCapacity(gpu_id=2, memory_bytes=24, max_slots=1),
            ),
        ),
        cases=[
            {"case_id": 0, "host_memory_bytes": 32, "gpu_count": 2, "gpu_memory_bytes": 16},
            {"case_id": 1, "host_memory_bytes": 32, "gpu_count": 2, "gpu_memory_bytes": 16},
            {"case_id": 2, "host_memory_bytes": 32, "gpu_count": 1, "gpu_memory_bytes": 16},
        ],
        worker=worker,
        disable_gpu_preallocation=True,
    )

    first_job = scheduler.next_case()
    second_job = scheduler.next_case()
    third_job = scheduler.next_case()

    assert first_job is not None
    assert second_job is not None
    assert third_job is None

    first_case, first_context = first_job
    second_case, second_context = second_job

    assert _case_id(first_case) == 0
    assert first_context["environment_variables"]["gpu_ids"] == "0,1"
    assert first_context["environment_variables"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert first_context["environment_variables"]["NVIDIA_VISIBLE_DEVICES"] == "0,1"
    assert first_context["environment_variables"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert _case_id(second_case) == 2
    assert second_context["environment_variables"]["gpu_ids"] == "2"
    assert second_context["environment_variables"]["gpu_id"] == "2"
    assert second_context["environment_variables"]["NVIDIA_VISIBLE_DEVICES"] == "2"
    assert second_context["environment_variables"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"

    scheduler.on_finish(first_case, first_context, SUCCESS_EXIT_CODE)
    third_job = scheduler.next_case()

    assert third_job is not None
    third_case, third_context = third_job
    assert _case_id(third_case) == 1
    assert third_context["environment_variables"]["gpu_ids"] == "0,1"
    assert third_context["environment_variables"]["NVIDIA_VISIBLE_DEVICES"] == "0,1"

    scheduler.on_finish(second_case, second_context, WORKER_PROTOCOL_ERROR_EXIT_CODE)
    scheduler.on_finish(third_case, third_context, SUCCESS_EXIT_CODE)

    assert scheduler.is_completed()
    assert [completion.exit_code for completion in scheduler.completions] == [
        SUCCESS_EXIT_CODE,
        WORKER_PROTOCOL_ERROR_EXIT_CODE,
        SUCCESS_EXIT_CODE,
    ]

    print(
        json.dumps(
            {
                "case": "standard_full_resource_scheduler_assigns_and_releases_resources",
                "source_file": SOURCE_FILE,
                "test": "test_standard_full_resource_scheduler_assigns_and_releases_resources",
                "gpu_sequences": [
                    cast(
                        dict[str, str],
                        completion.context.get("environment_variables", {}),
                    ).get("gpu_ids", "")
                    for completion in scheduler.completions
                ],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def test_standard_full_resource_scheduler_assigns_and_releases_resources() -> None:
    _run_standard_full_resource_scheduler_assigns_and_releases_resources()


def _run_standard_full_resource_scheduler_with_runner(tmp_path: Path) -> None:
    records_dir = tmp_path / "resource_records"
    cases: list[dict[str, object]] = [
        {
            "case_id": 0,
            "sleep_seconds": 0.40,
            "host_memory_bytes": 60,
            "gpu_count": 1,
            "gpu_memory_bytes": 28,
        },
        {
            "case_id": 1,
            "sleep_seconds": 0.20,
            "host_memory_bytes": 60,
            "gpu_count": 1,
            "gpu_memory_bytes": 28,
        },
        {
            "case_id": 2,
            "sleep_seconds": 0.20,
            "host_memory_bytes": 30,
            "gpu_count": 0,
            "gpu_memory_bytes": 0,
        },
        {
            "case_id": 3,
            "sleep_seconds": 0.20,
            "host_memory_bytes": 30,
            "gpu_count": 0,
            "gpu_memory_bytes": 0,
        },
    ]
    host_memory_by_case_id = {
        _case_id(case): cast(int, case["host_memory_bytes"]) for case in cases
    }
    scheduler = StandardFullResourceScheduler(
        resource_capacity=FullResourceCapacity(
            max_workers=4,
            host_memory_bytes=90,
            gpu_devices=(
                GPUDeviceCapacity(gpu_id=0, memory_bytes=32, max_slots=1),
                GPUDeviceCapacity(gpu_id=1, memory_bytes=32, max_slots=1),
            ),
        ),
        cases=cases,
        estimate_builder=_resource_estimate,
    )
    runner = StandardRunner(scheduler)
    worker = StandardWorker(FullResourceRecordingTask(records_dir))

    started_at = time.perf_counter()
    runner.run(worker)
    elapsed_seconds = time.perf_counter() - started_at
    records = _read_json_records(records_dir)

    assert sorted(_case_id(record) for record in records) == [0, 1, 2, 3]
    assert len({cast(int, record["pid"]) for record in records}) >= 2
    assert _has_parallel_overlap(records)
    assert elapsed_seconds < 0.95
    assert len(scheduler.completions) == 4

    for midpoint in _midpoints(records):
        active_records = _active_records_at(records, midpoint)
        active_host_memory_bytes = sum(
            host_memory_by_case_id[_case_id(record)] for record in active_records
        )
        assert active_host_memory_bytes <= 90

        active_gpu_ids = [
            gpu_id
            for record in active_records
            for gpu_id in cast(str, record["gpu_ids"]).split(",")
            if gpu_id
        ]
        assert len(active_gpu_ids) == len(set(active_gpu_ids))

    print(
        json.dumps(
            {
                "case": "standard_full_resource_scheduler_with_runner",
                "source_file": SOURCE_FILE,
                "test": "test_standard_full_resource_scheduler_with_runner",
                "elapsed_seconds": elapsed_seconds,
                "gpu_assignments": {
                    str(_case_id(record)): cast(str, record["gpu_ids"]) for record in records
                },
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def test_standard_full_resource_scheduler_with_runner(tmp_path: Path) -> None:
    _run_standard_full_resource_scheduler_with_runner(tmp_path)


def _run_all_tests() -> None:
    _run_detected_full_resource_capacity()
    _run_standard_full_resource_scheduler_assigns_and_releases_resources()
    with TemporaryDirectory() as tmp_dir:
        _run_standard_full_resource_scheduler_with_runner(Path(tmp_dir))


if __name__ == "__main__":
    _run_all_tests()
