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

from jax_util.experiment_runner import (
    GPUResourceCapacity,
    StandardGPUScheduler,
    StandardRunner,
    StandardWorker,
    SUCCESS_EXIT_CODE,
    TaskContext,
    WORKER_PROTOCOL_ERROR_EXIT_CODE,
    visible_gpu_ids_from_environment,
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
    case_id = case.get("case_id")
    if not isinstance(case_id, int):
        raise TypeError("case_id must be int.")
    return case_id


def _sleep_seconds(case: Mapping[str, object], /) -> float:
    value = case.get("sleep_seconds")
    if not isinstance(value, (int, float)):
        raise TypeError("sleep_seconds must be numeric.")
    return float(value)


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
class GPURecordingTask:
    records_dir: Path

    def __call__(self, case: dict[str, object], context: TaskContext) -> dict[str, object]:
        case_id = _case_id(case)
        started_at = time.time()
        time.sleep(_sleep_seconds(case))
        finished_at = time.time()

        record: dict[str, object] = {
            "case_id": case_id,
            "pid": os.getpid(),
            "gpu_id": context["gpu_id"],
            "cuda_visible_devices": context["CUDA_VISIBLE_DEVICES"],
            "started_at": started_at,
            "finished_at": finished_at,
        }
        _write_json(self.records_dir / f"case_{case_id}.json", record)
        return record


def _run_visible_gpu_ids_from_environment() -> None:
    environ = {"CUDA_VISIBLE_DEVICES": "2, 5"}
    resource_capacity = GPUResourceCapacity.from_environment(environ)

    assert visible_gpu_ids_from_environment(environ) == (2, 5)
    assert resource_capacity.gpu_ids == (2, 5)
    assert resource_capacity.max_workers == 2

    print(
        json.dumps(
            {
                "case": "visible_gpu_ids_from_environment",
                "source_file": SOURCE_FILE,
                "test": "test_visible_gpu_ids_from_environment",
                "gpu_ids": list(resource_capacity.gpu_ids),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def test_visible_gpu_ids_from_environment() -> None:
    _run_visible_gpu_ids_from_environment()


def _run_standard_gpu_scheduler_assigns_and_releases_gpu_slots() -> None:
    scheduler = StandardGPUScheduler(
        resource_capacity=GPUResourceCapacity(max_workers=2, gpu_ids=(3, 7)),
        cases=[
            {"case_id": 0},
            {"case_id": 1},
            {"case_id": 2},
        ],
    )

    first_job = scheduler.next_case()
    second_job = scheduler.next_case()
    third_job = scheduler.next_case()

    assert first_job is not None
    assert second_job is not None
    assert third_job is None

    first_case, first_context = first_job
    second_case, second_context = second_job

    assert first_context["gpu_id"] == "3"
    assert first_context["CUDA_VISIBLE_DEVICES"] == "3"
    assert second_context["gpu_id"] == "7"
    assert second_context["CUDA_VISIBLE_DEVICES"] == "7"

    scheduler.on_finish(first_case, first_context, SUCCESS_EXIT_CODE)
    third_job = scheduler.next_case()

    assert third_job is not None
    third_case, third_context = third_job
    assert _case_id(third_case) == 2
    assert third_context["gpu_id"] == "3"

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
                "case": "standard_gpu_scheduler_assigns_and_releases_gpu_slots",
                "source_file": SOURCE_FILE,
                "test": "test_standard_gpu_scheduler_assigns_and_releases_gpu_slots",
                "gpu_sequence": [
                    completion.context["gpu_id"] for completion in scheduler.completions
                ],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def test_standard_gpu_scheduler_assigns_and_releases_gpu_slots() -> None:
    _run_standard_gpu_scheduler_assigns_and_releases_gpu_slots()


def _run_standard_gpu_scheduler_with_runner(tmp_path: Path) -> None:
    records_dir = tmp_path / "gpu_records"
    cases: list[dict[str, object]] = [
        {"case_id": 0, "sleep_seconds": 0.20},
        {"case_id": 1, "sleep_seconds": 0.20},
        {"case_id": 2, "sleep_seconds": 0.20},
        {"case_id": 3, "sleep_seconds": 0.20},
    ]
    scheduler = StandardGPUScheduler(
        resource_capacity=GPUResourceCapacity(max_workers=2, gpu_ids=(0, 1)),
        cases=cases,
    )
    runner = StandardRunner(scheduler)
    worker = StandardWorker(GPURecordingTask(records_dir))

    started_at = time.perf_counter()
    runner.run(worker)
    elapsed_seconds = time.perf_counter() - started_at
    records = _read_json_records(records_dir)

    assert sorted(_case_id(record) for record in records) == [0, 1, 2, 3]
    assert {cast(str, record["gpu_id"]) for record in records} == {"0", "1"}
    assert all(record["gpu_id"] == record["cuda_visible_devices"] for record in records)
    assert len({cast(int, record["pid"]) for record in records}) >= 2
    assert _has_parallel_overlap(records)
    assert elapsed_seconds < 0.70
    assert len(scheduler.completions) == 4

    print(
        json.dumps(
            {
                "case": "standard_gpu_scheduler_with_runner",
                "source_file": SOURCE_FILE,
                "test": "test_standard_gpu_scheduler_with_runner",
                "elapsed_seconds": elapsed_seconds,
                "gpu_ids": sorted({cast(str, record["gpu_id"]) for record in records}),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def test_standard_gpu_scheduler_with_runner(tmp_path: Path) -> None:
    _run_standard_gpu_scheduler_with_runner(tmp_path)


def _run_all_tests() -> None:
    _run_visible_gpu_ids_from_environment()
    _run_standard_gpu_scheduler_assigns_and_releases_gpu_slots()
    with TemporaryDirectory() as tmp_dir:
        _run_standard_gpu_scheduler_with_runner(Path(tmp_dir))


if __name__ == "__main__":
    _run_all_tests()
