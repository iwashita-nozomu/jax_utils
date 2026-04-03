from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys
import time
from typing import cast

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from experiment_runner.context_utils import apply_environment_variables
from experiment_runner.execution_result import ExecutionResult, FailureKind
from experiment_runner.protocols import DispatchDecision, TaskContext
from experiment_runner.runner import (
    StandardCompletion,
    StandardResourceCapacity,
    StandardRunner,
    StandardScheduler,
    StandardWorker,
)


_IMPORT_SENSITIVE_TOKEN_SNAPSHOT: str | None = None


def _write_json(path: Path, record: dict[str, object], /) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=True, sort_keys=True), encoding="utf-8")


def _read_json_records(directory: Path, /) -> list[dict[str, object]]:
    if not directory.exists():
        return []
    return [
        cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(directory.glob("*.json"))
    ]


def _capture_import_sensitive_token() -> str:
    global _IMPORT_SENSITIVE_TOKEN_SNAPSHOT
    if _IMPORT_SENSITIVE_TOKEN_SNAPSHOT is None:
        _IMPORT_SENSITIVE_TOKEN_SNAPSHOT = os.environ.get("VISIBLE_TOKEN", "")
    return _IMPORT_SENSITIVE_TOKEN_SNAPSHOT


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


def _midpoints(records: list[dict[str, object]], /) -> list[float]:
    times = sorted(
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
        (times[index] + times[index + 1]) / 2.0
        for index in range(len(times) - 1)
    ]


@dataclass(frozen=True)
class _ImportSensitiveTask:
    records_dir: Path

    def __call__(self, case: dict[str, object], context: TaskContext) -> dict[str, object]:
        apply_environment_variables(context)
        started_at = time.time()
        snapshot = _capture_import_sensitive_token()
        finished_at = time.time()
        case_id = int(case["case_id"])
        record = {
            "case_id": case_id,
            "pid": os.getpid(),
            "visible_token_current": os.environ.get("VISIBLE_TOKEN", ""),
            "visible_token_snapshot": snapshot,
            "started_at": started_at,
            "finished_at": finished_at,
        }
        _write_json(self.records_dir / f"case_{case_id}.json", record)
        return record


@dataclass(frozen=True)
class _SleepRecordingTask:
    records_dir: Path

    def __call__(self, case: dict[str, object], context: TaskContext) -> dict[str, object]:
        del context
        case_id = int(case["case_id"])
        sleep_seconds = float(case["sleep_seconds"])
        started_at = time.time()
        time.sleep(sleep_seconds)
        finished_at = time.time()
        record = {
            "case_id": case_id,
            "pid": os.getpid(),
            "started_at": started_at,
            "finished_at": finished_at,
        }
        _write_json(self.records_dir / f"case_{case_id}.json", record)
        return record


def _noop_task(case: int, context: TaskContext) -> None:
    del case, context


def _raise_runtime_error(case: int, context: TaskContext) -> None:
    del case, context
    raise RuntimeError("boom from task")


def _hang_forever(case: int, context: TaskContext) -> None:
    del case, context
    time.sleep(30.0)


@dataclass(frozen=True)
class _AbruptExitWorker:
    task: object = _noop_task

    def __call__(self, case: int, context: TaskContext) -> int:
        del case, context
        os._exit(17)

    def resource_estimate(self, case: int) -> object:
        del case
        raise ValueError("resource_estimate is not used in this test")


@dataclass
class _SkipOddCaseController:
    skipped_case_ids: list[int] = field(default_factory=list)
    updates: list[tuple[int, str]] = field(default_factory=list)

    def should_skip(
        self,
        case: dict[str, object],
        context: TaskContext,
    ) -> DispatchDecision:
        del context
        case_id = int(case["case_id"])
        if case_id % 2 == 1:
            self.skipped_case_ids.append(case_id)
            return DispatchDecision.skip("skip odd case_id before spawn")
        return DispatchDecision.run()

    def update(
        self,
        case: dict[str, object],
        context: TaskContext,
        result: ExecutionResult,
    ) -> None:
        del context
        self.updates.append((int(case["case_id"]), result.status))


@dataclass
class _SkipAfterFailureController:
    skip_remaining: bool = False
    updates: list[tuple[int, str]] = field(default_factory=list)

    def should_skip(
        self,
        case: dict[str, object],
        context: TaskContext,
    ) -> DispatchDecision:
        del context
        if self.skip_remaining and int(case["case_id"]) >= 1:
            return DispatchDecision.skip("skip after prior failure")
        return DispatchDecision.run()

    def update(
        self,
        case: dict[str, object],
        context: TaskContext,
        result: ExecutionResult,
    ) -> None:
        del context
        self.updates.append((int(case["case_id"]), result.status))
        if result.status == "failed":
            self.skip_remaining = True


def _fail_case_zero(case: dict[str, object], context: TaskContext) -> None:
    del context
    if int(case["case_id"]) == 0:
        raise RuntimeError("case zero failed")


def test_standard_runner_isolates_import_sensitive_environment_per_case(
    tmp_path: Path,
) -> None:
    records_dir = tmp_path / "import_sensitive_records"
    cases = [
        {"case_id": 0, "visible_token": "gpu-0"},
        {"case_id": 1, "visible_token": "gpu-1"},
    ]

    scheduler = StandardScheduler(
        resource_capacity=StandardResourceCapacity(max_workers=1),
        cases=cases,
        context_builder=lambda case: {
            "environment_variables": {
                "VISIBLE_TOKEN": str(case["visible_token"]),
            }
        },
    )
    runner = StandardRunner(scheduler)
    worker = StandardWorker(_ImportSensitiveTask(records_dir))

    runner.run(worker)
    records = _read_json_records(records_dir)

    assert [record["visible_token_current"] for record in records] == ["gpu-0", "gpu-1"]
    assert [record["visible_token_snapshot"] for record in records] == ["gpu-0", "gpu-1"]


def test_standard_runner_respects_max_workers_with_fresh_processes(
    tmp_path: Path,
) -> None:
    records_dir = tmp_path / "sleep_records"
    cases = [
        {"case_id": 0, "sleep_seconds": 0.25},
        {"case_id": 1, "sleep_seconds": 0.25},
        {"case_id": 2, "sleep_seconds": 0.25},
        {"case_id": 3, "sleep_seconds": 0.25},
    ]

    scheduler = StandardScheduler(
        resource_capacity=StandardResourceCapacity(max_workers=2),
        cases=cases,
    )
    runner = StandardRunner(scheduler)
    worker = StandardWorker(_SleepRecordingTask(records_dir))

    started_at = time.perf_counter()
    runner.run(worker)
    elapsed_seconds = time.perf_counter() - started_at
    records = _read_json_records(records_dir)

    assert [record["case_id"] for record in records] == [0, 1, 2, 3]
    assert elapsed_seconds < 1.2

    for midpoint in _midpoints(records):
        active_records = _active_records_at(records, midpoint)
        assert len(active_records) <= 2


def test_standard_runner_progress_callback_uses_scheduler_total_case_count() -> None:
    progress_updates: list[tuple[int, int, int]] = []
    scheduler = StandardScheduler(
        resource_capacity=StandardResourceCapacity(max_workers=1),
        cases=[0, 1, 2],
    )
    scheduler.completions.append(
        StandardCompletion(case=-1, context={}, exit_code=0)
    )
    runner = StandardRunner(
        scheduler,
        progress_callback=lambda completed, total, elapsed, running: progress_updates.append(
            (completed, total, running)
        ),
    )
    worker = StandardWorker(_noop_task)

    runner.run(worker)

    assert progress_updates
    assert all(total == 4 for _, total, _ in progress_updates)


def test_standard_runner_can_skip_case_before_spawn(tmp_path: Path) -> None:
    records_dir = tmp_path / "skip_records"
    cases = [
        {"case_id": 0, "sleep_seconds": 0.01},
        {"case_id": 1, "sleep_seconds": 0.01},
        {"case_id": 2, "sleep_seconds": 0.01},
    ]
    skip_controller = _SkipOddCaseController()

    scheduler = StandardScheduler(
        resource_capacity=StandardResourceCapacity(max_workers=2),
        cases=cases,
        skip_controller=skip_controller,
    )
    runner = StandardRunner(scheduler)
    worker = StandardWorker(_SleepRecordingTask(records_dir))

    runner.run(worker)
    records = _read_json_records(records_dir)

    assert [record["case_id"] for record in records] == [0, 2]
    assert len(scheduler.completions) == 3

    skipped_completion = next(
        completion
        for completion in scheduler.completions
        if completion.result.status == "skipped"
    )
    assert skipped_completion.case["case_id"] == 1
    assert skipped_completion.result.status == "skipped"
    assert skipped_completion.result.message == "skip odd case_id before spawn"
    assert skipped_completion.result.source == "runner"
    assert skipped_completion.exit_code == 0
    assert skip_controller.skipped_case_ids == [1]
    assert sorted(skip_controller.updates) == [(0, "ok"), (1, "skipped"), (2, "ok")]


def test_standard_scheduler_skip_controller_updates_from_result() -> None:
    skip_controller = _SkipAfterFailureController()
    scheduler = StandardScheduler(
        resource_capacity=StandardResourceCapacity(max_workers=1),
        cases=[
            {"case_id": 0},
            {"case_id": 1},
        ],
        skip_controller=skip_controller,
    )
    runner = StandardRunner(scheduler)
    worker = StandardWorker(_fail_case_zero)

    runner.run(worker)

    assert len(scheduler.completions) == 2
    assert scheduler.completions[0].result.status == "failed"
    assert scheduler.completions[1].result.status == "skipped"
    assert scheduler.completions[1].result.message == "skip after prior failure"
    assert skip_controller.updates == [(0, "failed"), (1, "skipped")]


def test_standard_runner_captures_python_exception_as_structured_result() -> None:
    scheduler = StandardScheduler(
        resource_capacity=StandardResourceCapacity(max_workers=1),
        cases=[0],
    )
    runner = StandardRunner(scheduler)
    worker = StandardWorker(_raise_runtime_error)

    runner.run(worker)

    assert len(scheduler.completions) == 1
    completion = scheduler.completions[0]
    assert completion.exit_code == 1
    assert completion.result.status == "failed"
    assert completion.result.failure_kind == FailureKind.PYTHON_EXCEPTION.value
    assert "boom from task" in completion.result.message
    assert completion.result.traceback is not None


def test_standard_runner_synthesizes_parent_result_for_abrupt_child_exit() -> None:
    scheduler = StandardScheduler(
        resource_capacity=StandardResourceCapacity(max_workers=1),
        cases=[0],
    )
    runner = StandardRunner(scheduler)

    runner.run(_AbruptExitWorker())

    assert len(scheduler.completions) == 1
    completion = scheduler.completions[0]
    assert completion.result.status == "failed"
    assert completion.result.failure_kind == FailureKind.PROCESS_EXIT.value
    assert completion.result.raw_exit_code == 17
    assert completion.result.source == "parent"


def test_standard_runner_times_out_hanging_child_process() -> None:
    scheduler = StandardScheduler(
        resource_capacity=StandardResourceCapacity(max_workers=1),
        cases=[0],
    )
    runner = StandardRunner(
        scheduler,
        case_timeout_seconds=0.2,
        termination_grace_seconds=0.2,
    )
    worker = StandardWorker(_hang_forever)

    started_at = time.perf_counter()
    runner.run(worker)
    elapsed_seconds = time.perf_counter() - started_at

    assert elapsed_seconds < 5.0
    assert len(scheduler.completions) == 1
    completion = scheduler.completions[0]
    assert completion.result.status == "failed"
    assert completion.result.failure_kind == FailureKind.TIMEOUT.value
    assert completion.result.source == "parent"
