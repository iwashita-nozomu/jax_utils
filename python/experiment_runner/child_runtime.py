"""Child-process helpers for runner-managed worker execution."""

from __future__ import annotations

import traceback
from .execution_result import (
    ExecutionResult,
    FailureKind,
    build_failure_result,
)
from .protocols import TaskContext

def execute_worker_in_child(
    worker: object,
    case: object,
    context: TaskContext,
    /,
) -> ExecutionResult:
    """Execute one worker call and normalize the outcome into `ExecutionResult`."""
    try:
        result = worker(case, context)  # type: ignore[misc]
    except Exception as exc:
        return build_failure_result(
            failure_kind=FailureKind.PYTHON_EXCEPTION,
            message=str(exc),
            raw_exit_code=1,
            traceback=traceback.format_exc(),
            source="child",
        )

    if isinstance(result, ExecutionResult):
        return result

    return build_failure_result(
        failure_kind=FailureKind.PROTOCOL_ERROR,
        message=(
            "Worker must return experiment_runner.execution_result.ExecutionResult."
        ),
        raw_exit_code=1,
        source="child",
    )
