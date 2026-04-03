from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from experiment_runner.monitor import RuntimeMonitor
from experiment_runner.protocols import TaskContext
from experiment_runner.resource_scheduler import (
    FullResourceCapacity,
    FullResourceEstimate,
    StandardFullResourceScheduler,
)
from experiment_runner.runner import (
    StandardRunner,
    StandardWorker,
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


def _sleep_task(case: dict[str, object], context: TaskContext) -> None:
    del context
    time.sleep(float(case["sleep_seconds"]))


def _zero_estimate(case: object) -> FullResourceEstimate:
    del case
    return FullResourceEstimate()


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


def test_standard_runner_updates_runtime_monitor_state() -> None:
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
        scheduler = StandardFullResourceScheduler(
            resource_capacity=FullResourceCapacity(max_workers=1),
            cases=[
                {"case_id": 1, "sleep_seconds": 0.1},
                {"case_id": 2, "sleep_seconds": 0.1},
            ],
            estimate_builder=_zero_estimate,
            context_builder=lambda case: {
                "runner_metadata": {
                    "worker_label": f"cpu-{int(case['case_id'])}",
                    "gpu_ids": [],
                }
            },
        )
        runner = StandardRunner(
            scheduler,
            monitor=monitor,
        )
        runner.run(StandardWorker(_sleep_task))
        monitor.collect_once()

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
