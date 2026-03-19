from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Mapping

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from jax_util.experiment_runner.runner import (
    SUCCESS_EXIT_CODE,
    StandardResourceCapacity,
    StandardScheduler,
)
from jax_util.experiment_runner.protocols import WORKER_PROTOCOL_ERROR_EXIT_CODE


SOURCE_FILE = Path(__file__).name


def _case_id(case: Mapping[str, object], /) -> int:
    case_id = case.get("case_id")
    if not isinstance(case_id, int):
        raise TypeError("case_id must be int.")
    return case_id


def _context_from_case(case: Mapping[str, object], /) -> dict[str, str]:
    task_key = case.get("task_key")
    if isinstance(task_key, str):
        return {"task_key": task_key}
    return {}


def _run_standard_scheduler_uses_empty_context_by_default() -> None:
    cases: list[dict[str, object]] = [
        {"case_id": 0},
        {"case_id": 1},
    ]
    scheduler = StandardScheduler(
        resource_capacity=StandardResourceCapacity(max_workers=2),
        cases=cases,
    )

    first_job = scheduler.next_case()
    second_job = scheduler.next_case()
    third_job = scheduler.next_case()

    assert cases == [{"case_id": 0}, {"case_id": 1}]
    assert first_job == ({"case_id": 0}, {})
    assert second_job == ({"case_id": 1}, {})
    assert third_job is None
    assert scheduler.is_completed()

    print(
        json.dumps(
            {
                "case": "standard_scheduler_default_context",
                "source_file": SOURCE_FILE,
                "test": "test_standard_scheduler_uses_empty_context_by_default",
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def test_standard_scheduler_uses_empty_context_by_default() -> None:
    _run_standard_scheduler_uses_empty_context_by_default()


def _run_standard_scheduler_records_completions_with_context_builder() -> None:
    cases: list[dict[str, object]] = [
        {"case_id": 0, "task_key": "double"},
        {"case_id": 1, "task_key": "fail"},
    ]
    scheduler = StandardScheduler(
        resource_capacity=StandardResourceCapacity(max_workers=2),
        cases=cases,
        context_builder=_context_from_case,
    )

    first_job = scheduler.next_case()
    second_job = scheduler.next_case()

    assert first_job is not None
    assert second_job is not None

    first_case, first_context = first_job
    second_case, second_context = second_job

    scheduler.on_finish(first_case, first_context, SUCCESS_EXIT_CODE)
    first_context["task_key"] = "mutated-after-finish"
    scheduler.on_finish(second_case, second_context, WORKER_PROTOCOL_ERROR_EXIT_CODE)

    assert scheduler.is_completed()
    assert [_case_id(completion.case) for completion in scheduler.completions] == [0, 1]
    assert [completion.context["task_key"] for completion in scheduler.completions] == [
        "double",
        "fail",
    ]
    assert [completion.exit_code for completion in scheduler.completions] == [
        SUCCESS_EXIT_CODE,
        WORKER_PROTOCOL_ERROR_EXIT_CODE,
    ]

    print(
        json.dumps(
            {
                "case": "standard_scheduler_records_completions",
                "source_file": SOURCE_FILE,
                "test": "test_standard_scheduler_records_completions_with_context_builder",
                "exit_codes": [completion.exit_code for completion in scheduler.completions],
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def test_standard_scheduler_records_completions_with_context_builder() -> None:
    _run_standard_scheduler_records_completions_with_context_builder()


def _run_standard_resource_capacity_rejects_non_positive_max_workers() -> None:
    try:
        StandardResourceCapacity(max_workers=0)
    except ValueError as exc:
        assert str(exc) == "max_workers must be positive."
    else:
        raise AssertionError("ValueError was not raised.")

    print(
        json.dumps(
            {
                "case": "standard_resource_capacity_validation",
                "source_file": SOURCE_FILE,
                "test": "test_standard_resource_capacity_rejects_non_positive_max_workers",
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def test_standard_resource_capacity_rejects_non_positive_max_workers() -> None:
    _run_standard_resource_capacity_rejects_non_positive_max_workers()


def _run_all_tests() -> None:
    _run_standard_scheduler_uses_empty_context_by_default()
    _run_standard_scheduler_records_completions_with_context_builder()
    _run_standard_resource_capacity_rejects_non_positive_max_workers()


if __name__ == "__main__":
    _run_all_tests()
