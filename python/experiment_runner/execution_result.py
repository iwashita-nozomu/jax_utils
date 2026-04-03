"""Structured execution results for experiment-runner child processes."""

from __future__ import annotations

import signal
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class FailureKind(str, Enum):
    """Normalized failure categories for runner-managed child processes."""

    WORKER_EXIT_CODE = "worker_exit_code"
    PYTHON_EXCEPTION = "python_exception"
    PROCESS_EXIT = "process_exit"
    PROCESS_SIGNAL = "process_signal"
    TIMEOUT = "timeout"
    NO_COMPLETION = "no_completion"


@dataclass(frozen=True)
class ExecutionResult:
    """Structured result returned for one runner-managed child process."""

    status: str
    failure_kind: str | None = None
    message: str = ""
    raw_exit_code: int | None = None
    worker_exit_code: int | None = None
    signal_name: str | None = None
    stdout: str | None = None
    stderr: str | None = None
    traceback: str | None = None
    source: str = "child"

    @property
    def exit_code(self) -> int:
        """Return a backward-compatible integer exit code."""
        if self.worker_exit_code is not None:
            return int(self.worker_exit_code)
        return 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly dictionary representation."""
        return asdict(self)


def signal_name_from_exit_code(raw_exit_code: int | None, /) -> str | None:
    """Resolve a signal name when a process exits due to a signal."""
    if raw_exit_code is None or raw_exit_code >= 0:
        return None
    signal_number = -raw_exit_code
    try:
        return signal.Signals(signal_number).name
    except ValueError:
        return f"SIG{signal_number}"


def build_success_result(worker_exit_code: int = 0, /) -> ExecutionResult:
    """Build a successful execution result."""
    return ExecutionResult(
        status="ok",
        worker_exit_code=int(worker_exit_code),
        raw_exit_code=0,
        source="child",
    )


def build_skipped_result(
    message: str = "",
    /,
    *,
    source: str = "runner",
) -> ExecutionResult:
    """Build a skipped execution result without starting a child process."""
    return ExecutionResult(
        status="skipped",
        message=message,
        worker_exit_code=0,
        source=source,
    )


def build_failure_result(
    *,
    failure_kind: FailureKind | str,
    message: str,
    raw_exit_code: int | None = None,
    worker_exit_code: int | None = None,
    signal_name: str | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    traceback: str | None = None,
    source: str = "child",
) -> ExecutionResult:
    """Build a failed execution result."""
    resolved_failure_kind = (
        failure_kind.value if isinstance(failure_kind, FailureKind) else str(failure_kind)
    )
    resolved_signal_name = (
        signal_name if signal_name is not None else signal_name_from_exit_code(raw_exit_code)
    )
    return ExecutionResult(
        status="failed",
        failure_kind=resolved_failure_kind,
        message=message,
        raw_exit_code=raw_exit_code,
        worker_exit_code=worker_exit_code,
        signal_name=resolved_signal_name,
        stdout=stdout,
        stderr=stderr,
        traceback=traceback,
        source=source,
    )


def build_parent_exit_result(
    raw_exit_code: int | None,
    *,
    message: str | None = None,
    source: str = "parent",
) -> ExecutionResult:
    """Build a parent-side failure result from a raw child process exit code."""
    if raw_exit_code is None:
        resolved_kind = FailureKind.NO_COMPLETION
        resolved_message = "Child process ended without completion data."
    elif raw_exit_code < 0:
        resolved_kind = FailureKind.PROCESS_SIGNAL
        signal_name = signal_name_from_exit_code(raw_exit_code)
        resolved_message = (
            message
            if message is not None
            else f"Child process terminated by signal {signal_name or raw_exit_code}."
        )
        return build_failure_result(
            failure_kind=resolved_kind,
            message=resolved_message,
            raw_exit_code=raw_exit_code,
            source=source,
        )
    elif raw_exit_code == 0:
        resolved_kind = FailureKind.NO_COMPLETION
        resolved_message = (
            message
            if message is not None
            else "Child process exited cleanly but returned no completion data."
        )
    else:
        resolved_kind = FailureKind.PROCESS_EXIT
        resolved_message = (
            message
            if message is not None
            else f"Child process exited with return code {raw_exit_code}."
        )

    return build_failure_result(
        failure_kind=resolved_kind,
        message=resolved_message,
        raw_exit_code=raw_exit_code,
        source=source,
    )


def coerce_execution_result(result: ExecutionResult | int, /) -> ExecutionResult:
    """Coerce a legacy integer exit code into an `ExecutionResult`."""
    if isinstance(result, ExecutionResult):
        return result
    if int(result) == 0:
        return build_success_result(int(result))
    return build_failure_result(
        failure_kind=FailureKind.WORKER_EXIT_CODE,
        message=f"Worker returned non-success exit code {int(result)}.",
        worker_exit_code=int(result),
        raw_exit_code=int(result),
        source="child",
    )
