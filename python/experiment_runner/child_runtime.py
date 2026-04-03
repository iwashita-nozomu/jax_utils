"""Child-process helpers for runner-managed worker execution."""

from __future__ import annotations

import traceback
from .execution_result import (
    ExecutionResult,
    FailureKind,
    build_failure_result,
    build_success_result,
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
        worker_exit_code = int(worker(case, context))  # type: ignore[misc]
    except Exception as exc:
        return build_failure_result(
            failure_kind=FailureKind.PYTHON_EXCEPTION,
            message=str(exc),
            worker_exit_code=1,
            raw_exit_code=1,
            traceback=traceback.format_exc(),
            source="child",
        )

    if worker_exit_code == 0:
        return build_success_result(worker_exit_code)

    return build_failure_result(
        failure_kind=FailureKind.WORKER_EXIT_CODE,
        message=f"Worker returned non-success exit code {worker_exit_code}.",
        worker_exit_code=worker_exit_code,
        raw_exit_code=worker_exit_code,
        source="child",
    )
