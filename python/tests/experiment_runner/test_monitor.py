from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from experiment_runner.monitor import RuntimeMonitor
from experiment_runner.subprocess_scheduler import (
    CHILD_COMPLETE_PREFIX,
    WorkerSlot,
    run_cases_with_subprocess_scheduler,
)


def _read_json(url: str, /) -> object:
    with urlopen(url, timeout=5.0) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_text(url: str, /) -> str:
    with urlopen(url, timeout=5.0) as response:
        return response.read().decode("utf-8")


def _wait_for_url(url: str, /) -> None:
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            _read_text(url)
            return
        except URLError:
            time.sleep(0.02)
    raise AssertionError(f"monitor endpoint did not become ready: {url}")


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


def test_runtime_monitor_serves_http_snapshot_history_and_events() -> None:
    monitor = RuntimeMonitor.for_run(
        bind_host="127.0.0.1",
        port=0,
        sample_interval_seconds=0.05,
        history_limit=8,
        event_limit=8,
        enable_http=True,
        host_metrics_collector=lambda: {
            "cpu_percent": 37.5,
            "memory_total_bytes": 1024,
            "memory_available_bytes": 256,
            "memory_used_bytes": 768,
        },
        gpu_metrics_collector=lambda: [
            {
                "gpu_id": 0,
                "uuid": "GPU-test",
                "utilization_gpu_percent": 88.0,
                "utilization_memory_percent": 50.0,
                "memory_total_bytes": 4096,
                "memory_used_bytes": 2048,
                "temperature_c": 65.0,
                "power_watts": 120.5,
            }
        ],
    )
    monitor.update_runner_state(
        pending_cases=3,
        running_cases=1,
        completed_cases=5,
        max_workers=2,
    )
    monitor.register_worker(
        case_id=42,
        worker_label="gpu-0-w0",
        pid=12345,
        gpu_ids=(0,),
    )
    monitor.start()
    try:
        _wait_for_url(f"{monitor.local_url}/healthz")
        monitor.collect_once()

        snapshot = _read_json(f"{monitor.local_url}/api/v1/snapshot")
        assert isinstance(snapshot, dict)
        assert snapshot["monitor"]["mode"] == "run"
        assert snapshot["monitor"]["sample_interval_seconds"] == 0.05
        assert snapshot["runner"]["pending_cases"] == 3
        assert snapshot["runner"]["running_cases"] == 1
        assert snapshot["host"]["memory_used_bytes"] == 768
        assert snapshot["gpus"][0]["gpu_id"] == 0
        assert snapshot["workers"][0]["case_id"] == 42
        assert snapshot["workers"][0]["worker_label"] == "gpu-0-w0"

        history = _read_json(f"{monitor.local_url}/api/v1/history?limit=1")
        assert isinstance(history, list)
        assert len(history) == 1

        events = _read_json(f"{monitor.local_url}/api/v1/events")
        assert isinstance(events, list)
        assert events[0]["event"] == "case_started"

        html = _read_text(f"{monitor.local_url}/")
        assert "Experiment Runner Monitor" in html
        assert "/api/v1/snapshot" in html

        health = _read_json(f"{monitor.local_url}/healthz")
        assert health == {
            "ok": True,
            "port": monitor.port,
            "mode": "run",
            "sample_interval_seconds": 0.05,
        }

        monitor.complete_worker(pid=12345, result={"status": "ok"})
        monitor.update_runner_state(
            pending_cases=0,
            running_cases=0,
            completed_cases=6,
            max_workers=2,
        )
        monitor.collect_once()

        final_snapshot = _read_json(f"{monitor.local_url}/api/v1/snapshot")
        assert final_snapshot["runner"]["completed_cases"] == 6
        assert final_snapshot["workers"] == []

        final_events = _read_json(f"{monitor.local_url}/api/v1/events")
        assert [event["event"] for event in final_events][-2:] == [
            "case_started",
            "case_finished",
        ]
    finally:
        monitor.stop()


def test_subprocess_scheduler_updates_runtime_monitor_state(tmp_path: Path) -> None:
    child_script = _write_script(
        tmp_path / "success_child.py",
        f"""
        import json
        import sys
        import time

        time.sleep(0.1)
        print({CHILD_COMPLETE_PREFIX!r} + json.dumps({{
            "status": "ok",
            "case_id": int(sys.argv[1]),
            "worker_label": sys.argv[2],
        }}), flush=True)
        """,
    )
    monitor = RuntimeMonitor.for_run(
        enable_http=False,
        sample_interval_seconds=0.02,
        history_limit=16,
        event_limit=16,
        host_metrics_collector=lambda: {
            "cpu_percent": 10.0,
            "memory_total_bytes": 2048,
            "memory_available_bytes": 1024,
            "memory_used_bytes": 1024,
        },
        gpu_metrics_collector=lambda: [],
    )
    monitor.start()
    try:
        worker_slots = [
            WorkerSlot(
                worker_label="cpu-0",
                gpu_index=None,
                gpu_slot=0,
                cpu_affinity=(),
            )
        ]
        results = run_cases_with_subprocess_scheduler(
            [{"case_id": 1}, {"case_id": 2}],
            worker_slots,
            timeout_seconds=5,
            build_child_command=lambda case, worker_slot: [
                sys.executable,
                str(child_script),
                str(case["case_id"]),
                worker_slot.worker_label,
            ],
            build_parent_failure_result=_failure_result,
            fallback_jsonl_output_path=None,
            cwd=tmp_path,
            poll_interval_seconds=0.01,
            monitor=monitor,
        )
        monitor.collect_once()

        assert results == [
            {"status": "ok", "case_id": 1, "worker_label": "cpu-0"},
            {"status": "ok", "case_id": 2, "worker_label": "cpu-0"},
        ]

        snapshot = monitor.snapshot()
        assert snapshot["runner"] == {
            "pending_cases": 0,
            "running_cases": 0,
            "completed_cases": 2,
            "max_workers": 1,
        }
        assert snapshot["workers"] == []

        events = monitor.events()
        assert [event["event"] for event in events] == [
            "case_started",
            "case_finished",
            "case_started",
            "case_finished",
        ]
        assert [event["case_id"] for event in events] == [1, 1, 2, 2]
    finally:
        monitor.stop()


def test_daemon_monitor_uses_lighter_defaults_and_allows_interval_updates() -> None:
    monitor = RuntimeMonitor.for_daemon(
        enable_http=False,
        host_metrics_collector=lambda: {
            "cpu_percent": 5.0,
            "memory_total_bytes": 4096,
            "memory_available_bytes": 3072,
            "memory_used_bytes": 1024,
        },
        gpu_metrics_collector=lambda: [],
    )
    assert monitor.mode == "daemon"
    assert monitor.sample_interval_seconds == 2.0

    monitor.register_worker(
        case_id=7,
        worker_label="cpu-0",
        pid=22222,
        gpu_ids=(),
    )
    monitor.complete_worker(pid=22222, result={"status": "ok"})
    monitor.collect_once()

    snapshot = monitor.snapshot()
    assert snapshot["monitor"]["mode"] == "daemon"
    assert snapshot["workers"] == []
    assert monitor.events() == []

    monitor.complete_worker(
        pid=33333,
        result={"status": "failed", "failure_kind": "timeout"},
    )
    timeout_events = monitor.events()
    assert len(timeout_events) == 1
    assert timeout_events[0]["event"] == "case_timeout"

    monitor.set_sample_interval_seconds(0.25)
    assert monitor.sample_interval_seconds == 0.25
