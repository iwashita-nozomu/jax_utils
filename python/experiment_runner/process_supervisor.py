"""Process-lifecycle helpers for runner-managed child processes."""

from __future__ import annotations

from typing import Any


def terminate_then_kill_process(
    process: Any,
    /,
    *,
    grace_seconds: float,
) -> None:
    """Try `terminate()` first and escalate to `kill()` if needed."""
    if not process.is_alive():
        process.join(timeout=0.0)
        return

    process.terminate()
    process.join(timeout=grace_seconds)
    if not process.is_alive():
        return

    kill = getattr(process, "kill", None)
    if callable(kill):
        kill()
    else:
        process.terminate()
    process.join(timeout=grace_seconds)
