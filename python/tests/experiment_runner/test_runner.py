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
    SUCCESS_EXIT_CODE,
    StandardResourceCapacity,
    StandardRunner,
    StandardScheduler,
    StandardWorker,
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


def _case_id(case: dict[str, object], /) -> int:
    case_id = case.get("case_id")
    if not isinstance(case_id, int):
        raise TypeError("case_id must be int.")
    return case_id


def _float_value(case: dict[str, object], key: str, /) -> float:
    value = case.get(key)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be numeric.")
    return float(value)


def _int_value(case: dict[str, object], key: str, /) -> int:
    value = case.get(key)
    if not isinstance(value, int):
        raise TypeError(f"{key} must be int.")
    return value


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
class RecordingTask:
    records_dir: Path

    def __call__(self, case: dict[str, object], context: TaskContext) -> dict[str, object]:
        case_id = _case_id(case)
        started_at = time.time()
        time.sleep(_float_value(case, "sleep_seconds"))

        task_key = context.get("task_key", "identity")
        value = _int_value(case, "value")

        if task_key == "fail":
            raise RuntimeError(f"expected failure case_id={case_id}")
        if task_key == "double":
            result_value = value * 2
        elif task_key == "increment":
            result_value = value + 1
        else:
            result_value = value

        finished_at = time.time()
        record: dict[str, object] = {
            "case_id": case_id,
            "pid": os.getpid(),
            "task_key": task_key,
            "result_value": result_value,
            "started_at": started_at,
            "finished_at": finished_at,
        }
        _write_json(self.records_dir / f"case_{case_id}.json", record)
        return record


class RecordingScheduler(StandardScheduler[dict[str, object]]):
    def __init__(
        self,
        resource_capacity: StandardResourceCapacity,
        cases: list[dict[str, object]],
    ) -> None:
        super().__init__(resource_capacity=resource_capacity, cases=cases)
        self.completions: list[tuple[int, int]] = []

    def next_case(self) -> tuple[dict[str, object], TaskContext] | None:
        job = super().next_case()
        if job is None:
            return None
        case, context = job
        task_key = case.get("task_key")
        if isinstance(task_key, str):
            context = dict(context)
            context["task_key"] = task_key
        return case, context

    def on_finish(
        self,
        case: dict[str, object],
        context: TaskContext,
        exit_code: int,
    ) -> None:
        del context
        self.completions.append((_case_id(case), exit_code))


def _run_standard_runner_parallel_processes(tmp_path: Path) -> None:
    records_dir = tmp_path / "parallel_records"
    cases: list[dict[str, object]] = [
        {"case_id": 0, "sleep_seconds": 0.30, "value": 10},
        {"case_id": 1, "sleep_seconds": 0.30, "value": 20},
        {"case_id": 2, "sleep_seconds": 0.30, "value": 30},
        {"case_id": 3, "sleep_seconds": 0.30, "value": 40},
    ]
    scheduler = StandardScheduler(
        resource_capacity=StandardResourceCapacity(max_workers=2),
        cases=list(cases),
    )
    runner = StandardRunner(scheduler)
    worker = StandardWorker(RecordingTask(records_dir))

    started_at = time.perf_counter()
    runner.run(worker)
    elapsed_seconds = time.perf_counter() - started_at
    records = _read_json_records(records_dir)

    assert sorted(_case_id(record) for record in records) == [0, 1, 2, 3]
    assert len({cast(int, record["pid"]) for record in records}) >= 2
    assert _has_parallel_overlap(records)
    assert elapsed_seconds < 1.10

    print(
        json.dumps(
            {
                "case": "standard_runner_parallel_processes",
                "source_file": SOURCE_FILE,
                "test": "test_standard_runner_parallel_processes",
                "elapsed_seconds": elapsed_seconds,
                "pids": sorted({cast(int, record["pid"]) for record in records}),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def test_standard_runner_parallel_processes(tmp_path: Path) -> None:
    _run_standard_runner_parallel_processes(tmp_path)


def _run_standard_runner_bulk_parallel_processes(tmp_path: Path) -> None:
    records_dir = tmp_path / "bulk_parallel_records"
    cases: list[dict[str, object]] = [
        {"case_id": index, "sleep_seconds": 0.08, "value": index}
        for index in range(24)
    ]
    scheduler = StandardScheduler(
        resource_capacity=StandardResourceCapacity(max_workers=4),
        cases=list(cases),
    )
    runner = StandardRunner(scheduler)
    worker = StandardWorker(RecordingTask(records_dir))

    started_at = time.perf_counter()
    runner.run(worker)
    elapsed_seconds = time.perf_counter() - started_at
    records = _read_json_records(records_dir)

    assert len(records) == 24
    assert sorted(_case_id(record) for record in records) == list(range(24))
    assert len({cast(int, record["pid"]) for record in records}) == 4
    assert _has_parallel_overlap(records)
    assert elapsed_seconds < 0.70

    print(
        json.dumps(
            {
                "case": "standard_runner_bulk_parallel_processes",
                "source_file": SOURCE_FILE,
                "test": "test_standard_runner_bulk_parallel_processes",
                "elapsed_seconds": elapsed_seconds,
                "num_records": len(records),
                "pids": sorted({cast(int, record["pid"]) for record in records}),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def test_standard_runner_bulk_parallel_processes(tmp_path: Path) -> None:
    _run_standard_runner_bulk_parallel_processes(tmp_path)


def _run_standard_runner_context_switch_and_failure(tmp_path: Path) -> None:
    records_dir = tmp_path / "context_records"
    cases: list[dict[str, object]] = [
        {"case_id": 0, "sleep_seconds": 0.05, "value": 3, "task_key": "double"},
        {"case_id": 1, "sleep_seconds": 0.05, "value": 7, "task_key": "fail"},
        {"case_id": 2, "sleep_seconds": 0.05, "value": 5, "task_key": "increment"},
    ]
    scheduler = RecordingScheduler(
        resource_capacity=StandardResourceCapacity(max_workers=2),
        cases=list(cases),
    )
    runner = StandardRunner(scheduler)
    worker = StandardWorker(RecordingTask(records_dir))

    runner.run(worker)
    records = _read_json_records(records_dir)
    records_by_case_id = {_case_id(record): record for record in records}
    completions_by_case_id = dict(sorted(scheduler.completions))

    assert records_by_case_id[0]["task_key"] == "double"
    assert records_by_case_id[0]["result_value"] == 6
    assert records_by_case_id[2]["task_key"] == "increment"
    assert records_by_case_id[2]["result_value"] == 6
    assert 1 not in records_by_case_id
    assert completions_by_case_id == {
        0: SUCCESS_EXIT_CODE,
        1: WORKER_PROTOCOL_ERROR_EXIT_CODE,
        2: SUCCESS_EXIT_CODE,
    }

    print(
        json.dumps(
            {
                "case": "standard_runner_context_switch_and_failure",
                "source_file": SOURCE_FILE,
                "test": "test_standard_runner_context_switch_and_failure",
                "completions": completions_by_case_id,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def test_standard_runner_context_switch_and_failure(tmp_path: Path) -> None:
    _run_standard_runner_context_switch_and_failure(tmp_path)


def _run_all_tests() -> None:
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        _run_standard_runner_parallel_processes(tmp_path)
        _run_standard_runner_bulk_parallel_processes(tmp_path)
        _run_standard_runner_context_switch_and_failure(tmp_path)


if __name__ == "__main__":
    _run_all_tests()
