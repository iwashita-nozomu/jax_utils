"""Lightweight real-time monitor for experiment runners.

This module provides a small in-process monitor service that keeps runtime
snapshots in memory and can optionally expose both a tiny HTML dashboard and
JSON endpoints over HTTP.
"""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
import time
from typing import Callable, Literal, Mapping, Sequence, cast
from urllib.parse import parse_qs, urlparse


HostMetricsCollector = Callable[[], dict[str, object]]
GPUMetricsCollector = Callable[[], list[dict[str, object]]]

__all__ = [
    "RuntimeMonitorConfig",
    "RuntimeMonitor",
]


@dataclass(frozen=True)
class RuntimeMonitorConfig:
    """Configuration for the lightweight runtime monitor."""

    mode: Literal["run", "daemon"] = "run"
    bind_host: str = "127.0.0.1"
    port: int = 8765
    sample_interval_seconds: float = 1.0
    history_limit: int = 300
    event_limit: int = 1000
    enable_http: bool = True
    record_case_start_events: bool = True
    record_case_finish_events: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"run", "daemon"}:
            raise ValueError("mode must be 'run' or 'daemon'.")
        if not self.bind_host:
            raise ValueError("bind_host must not be empty.")
        if self.port < 0:
            raise ValueError("port must be non-negative.")
        if self.sample_interval_seconds <= 0.0:
            raise ValueError("sample_interval_seconds must be positive.")
        if self.history_limit < 1:
            raise ValueError("history_limit must be positive.")
        if self.event_limit < 1:
            raise ValueError("event_limit must be positive.")

    @classmethod
    def for_run(
        cls,
        *,
        bind_host: str = "127.0.0.1",
        port: int = 8765,
        sample_interval_seconds: float = 1.0,
        history_limit: int = 300,
        event_limit: int = 1000,
        enable_http: bool = True,
    ) -> RuntimeMonitorConfig:
        """Return the fuller in-run monitor configuration."""
        return cls(
            mode="run",
            bind_host=bind_host,
            port=port,
            sample_interval_seconds=sample_interval_seconds,
            history_limit=history_limit,
            event_limit=event_limit,
            enable_http=enable_http,
            record_case_start_events=True,
            record_case_finish_events=True,
        )

    @classmethod
    def for_daemon(
        cls,
        *,
        bind_host: str = "127.0.0.1",
        port: int = 8765,
        sample_interval_seconds: float = 2.0,
        history_limit: int = 60,
        event_limit: int = 120,
        enable_http: bool = True,
    ) -> RuntimeMonitorConfig:
        """Return the lighter daemon-style monitor configuration."""
        return cls(
            mode="daemon",
            bind_host=bind_host,
            port=port,
            sample_interval_seconds=sample_interval_seconds,
            history_limit=history_limit,
            event_limit=event_limit,
            enable_http=enable_http,
            record_case_start_events=False,
            record_case_finish_events=False,
        )


@dataclass(frozen=True)
class _CPUCounters:
    total: int
    idle: int


@dataclass(frozen=True)
class _ActiveWorker:
    case_id: object | None
    worker_label: str
    pid: int
    state: str
    gpu_ids: tuple[int, ...]
    started_at: str
    started_at_epoch: float


class _MonitorHTTPServer(ThreadingHTTPServer):
    """HTTP server wrapper that exposes the owning monitor."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        monitor: RuntimeMonitor,
    ) -> None:
        self.monitor = monitor
        super().__init__(server_address, _MonitorRequestHandler)


class _MonitorRequestHandler(BaseHTTPRequestHandler):
    """Serve the runtime monitor HTML and JSON endpoints."""

    def do_GET(self) -> None:  # noqa: N802
        monitor = cast(_MonitorHTTPServer, self.server).monitor
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._write_html(monitor.render_dashboard_html())
            return
        if parsed.path == "/api/v1/snapshot":
            self._write_json(monitor.snapshot())
            return
        if parsed.path == "/api/v1/history":
            query = parse_qs(parsed.query)
            limit = _parse_limit(query.get("limit", []))
            self._write_json(monitor.history(limit=limit))
            return
        if parsed.path == "/api/v1/events":
            query = parse_qs(parsed.query)
            limit = _parse_limit(query.get("limit", []))
            self._write_json(monitor.events(limit=limit))
            return
        if parsed.path == "/healthz":
            self._write_json(
                {
                    "ok": monitor.is_running,
                    "port": monitor.port,
                    "mode": monitor.mode,
                    "sample_interval_seconds": monitor.sample_interval_seconds,
                }
            )
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _write_json(self, payload: object, /) -> None:
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _write_html(self, body: str, /) -> None:
        encoded = body.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _parse_limit(values: Sequence[str], /) -> int | None:
    if not values:
        return None
    try:
        limit = int(values[0])
    except ValueError:
        return None
    return limit if limit > 0 else None


def _iso_utc_timestamp(epoch_seconds: float, /) -> str:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat()


def _read_cpu_counters(proc_root: Path, /) -> _CPUCounters | None:
    stat_path = proc_root / "stat"
    try:
        first_line = stat_path.read_text(encoding="utf-8").splitlines()[0]
    except (FileNotFoundError, IndexError, OSError):
        return None

    tokens = first_line.split()
    if len(tokens) < 6 or tokens[0] != "cpu":
        return None

    try:
        values = [int(token) for token in tokens[1:]]
    except ValueError:
        return None

    total = sum(values)
    idle = values[3]
    if len(values) > 4:
        idle += values[4]
    return _CPUCounters(total=total, idle=idle)


def _read_meminfo(proc_root: Path, /) -> tuple[int, int]:
    meminfo_path = proc_root / "meminfo"
    total_bytes = 0
    available_bytes = 0
    try:
        lines = meminfo_path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError):
        return total_bytes, available_bytes

    for line in lines:
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        fields = raw_value.strip().split()
        if not fields:
            continue
        try:
            value = int(fields[0]) * 1024
        except ValueError:
            continue
        if key == "MemTotal":
            total_bytes = value
        elif key == "MemAvailable":
            available_bytes = value
    return total_bytes, available_bytes


def _coerce_float(value: str, /) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    lowered = stripped.lower()
    if lowered in {"n/a", "[not supported]", "not supported"}:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _default_gpu_metrics_collector() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.total,memory.used,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []

    if completed.returncode != 0:
        return []

    gpus: list[dict[str, object]] = []
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 8:
            continue
        try:
            gpu_id = int(parts[0])
        except ValueError:
            continue

        total_mib = _coerce_float(parts[4])
        used_mib = _coerce_float(parts[5])
        memory_total_bytes = int(total_mib * 1024 * 1024) if total_mib is not None else 0
        memory_used_bytes = int(used_mib * 1024 * 1024) if used_mib is not None else 0
        utilization_memory_percent = None
        if memory_total_bytes > 0:
            utilization_memory_percent = round(
                (memory_used_bytes / memory_total_bytes) * 100.0,
                2,
            )

        gpus.append(
            {
                "gpu_id": gpu_id,
                "uuid": parts[1],
                "utilization_gpu_percent": _coerce_float(parts[2]),
                "utilization_memory_percent": utilization_memory_percent,
                "memory_total_bytes": memory_total_bytes,
                "memory_used_bytes": memory_used_bytes,
                "temperature_c": _coerce_float(parts[6]),
                "power_watts": _coerce_float(parts[7]),
            }
        )
    return gpus


class RuntimeMonitor:
    """Keep lightweight runtime snapshots and optionally expose them over HTTP."""

    @classmethod
    def for_run(
        cls,
        *,
        bind_host: str = "127.0.0.1",
        port: int = 8765,
        sample_interval_seconds: float = 1.0,
        history_limit: int = 300,
        event_limit: int = 1000,
        enable_http: bool = True,
        host_metrics_collector: HostMetricsCollector | None = None,
        gpu_metrics_collector: GPUMetricsCollector | None = None,
        proc_root: Path | None = None,
        time_source: Callable[[], float] | None = None,
    ) -> RuntimeMonitor:
        """Create a run-scoped monitor with richer event retention."""
        return cls(
            config=RuntimeMonitorConfig.for_run(
                bind_host=bind_host,
                port=port,
                sample_interval_seconds=sample_interval_seconds,
                history_limit=history_limit,
                event_limit=event_limit,
                enable_http=enable_http,
            ),
            host_metrics_collector=host_metrics_collector,
            gpu_metrics_collector=gpu_metrics_collector,
            proc_root=proc_root,
            time_source=time_source,
        )

    @classmethod
    def for_daemon(
        cls,
        *,
        bind_host: str = "127.0.0.1",
        port: int = 8765,
        sample_interval_seconds: float = 2.0,
        history_limit: int = 60,
        event_limit: int = 120,
        enable_http: bool = True,
        host_metrics_collector: HostMetricsCollector | None = None,
        gpu_metrics_collector: GPUMetricsCollector | None = None,
        proc_root: Path | None = None,
        time_source: Callable[[], float] | None = None,
    ) -> RuntimeMonitor:
        """Create a lighter daemon-style monitor for always-on use."""
        return cls(
            config=RuntimeMonitorConfig.for_daemon(
                bind_host=bind_host,
                port=port,
                sample_interval_seconds=sample_interval_seconds,
                history_limit=history_limit,
                event_limit=event_limit,
                enable_http=enable_http,
            ),
            host_metrics_collector=host_metrics_collector,
            gpu_metrics_collector=gpu_metrics_collector,
            proc_root=proc_root,
            time_source=time_source,
        )

    def __init__(
        self,
        config: RuntimeMonitorConfig | None = None,
        *,
        host_metrics_collector: HostMetricsCollector | None = None,
        gpu_metrics_collector: GPUMetricsCollector | None = None,
        proc_root: Path | None = None,
        time_source: Callable[[], float] | None = None,
    ) -> None:
        self._config = RuntimeMonitorConfig() if config is None else config
        self._host_metrics_collector = host_metrics_collector
        self._gpu_metrics_collector = gpu_metrics_collector
        self._proc_root = Path("/proc") if proc_root is None else proc_root
        self._time_source = time.time if time_source is None else time_source

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._collector_thread: threading.Thread | None = None
        self._server_thread: threading.Thread | None = None
        self._server: _MonitorHTTPServer | None = None
        self._sample_interval_seconds = self._config.sample_interval_seconds

        self._history: deque[dict[str, object]] = deque(maxlen=self._config.history_limit)
        self._events: deque[dict[str, object]] = deque(maxlen=self._config.event_limit)
        self._runner_state: dict[str, int] = {
            "pending_cases": 0,
            "running_cases": 0,
            "completed_cases": 0,
            "max_workers": 0,
        }
        self._workers: dict[int, _ActiveWorker] = {}
        self._latest_snapshot: dict[str, object] | None = None
        self._cached_host_metrics: dict[str, object] = {
            "cpu_percent": 0.0,
            "memory_total_bytes": 0,
            "memory_available_bytes": 0,
            "memory_used_bytes": 0,
        }
        self._cached_gpu_metrics: list[dict[str, object]] = []
        self._previous_cpu_counters: _CPUCounters | None = None

    @property
    def port(self) -> int:
        with self._lock:
            if self._server is not None:
                return int(self._server.server_port)
        return self._config.port

    @property
    def local_url(self) -> str:
        host = self._config.bind_host
        if host == "0.0.0.0":
            host = "127.0.0.1"
        return f"http://{host}:{self.port}"

    @property
    def mode(self) -> Literal["run", "daemon"]:
        return self._config.mode

    @property
    def sample_interval_seconds(self) -> float:
        with self._lock:
            return self._sample_interval_seconds

    @property
    def is_running(self) -> bool:
        collector = self._collector_thread
        return collector is not None and collector.is_alive()

    def set_sample_interval_seconds(self, sample_interval_seconds: float, /) -> None:
        """Update the background collection interval for subsequent polls."""
        if sample_interval_seconds <= 0.0:
            raise ValueError("sample_interval_seconds must be positive.")
        with self._lock:
            self._sample_interval_seconds = float(sample_interval_seconds)

    def start(self) -> None:
        """Start background collection and, when enabled, the HTTP surface."""
        with self._lock:
            if self.is_running:
                return
            self._stop_event.clear()

        self.collect_once()

        if self._config.enable_http:
            server = _MonitorHTTPServer(
                (self._config.bind_host, self._config.port),
                self,
            )
            with self._lock:
                self._server = server
            self._server_thread = threading.Thread(
                target=server.serve_forever,
                name="experiment-runner-monitor-http",
                daemon=True,
            )
            self._server_thread.start()

        self._collector_thread = threading.Thread(
            target=self._collector_loop,
            name="experiment-runner-monitor-collector",
            daemon=True,
        )
        self._collector_thread.start()

    def stop(self) -> None:
        """Stop background collection and shut down the HTTP server."""
        self._stop_event.set()

        collector_thread = self._collector_thread
        if collector_thread is not None:
            collector_thread.join(timeout=5.0)
        self._collector_thread = None

        server = self._server
        if server is not None:
            server.shutdown()
            server.server_close()
        server_thread = self._server_thread
        if server_thread is not None:
            server_thread.join(timeout=5.0)
        self._server = None
        self._server_thread = None

    def update_runner_state(
        self,
        *,
        pending_cases: int,
        running_cases: int,
        completed_cases: int,
        max_workers: int,
    ) -> None:
        """Update high-level runner counts shown in snapshots."""
        with self._lock:
            self._runner_state = {
                "pending_cases": int(pending_cases),
                "running_cases": int(running_cases),
                "completed_cases": int(completed_cases),
                "max_workers": int(max_workers),
            }

    def register_worker(
        self,
        *,
        case_id: object | None,
        worker_label: str,
        pid: int,
        gpu_ids: Sequence[int],
    ) -> None:
        """Register a running worker process."""
        now = self._time_source()
        with self._lock:
            self._workers[pid] = _ActiveWorker(
                case_id=case_id,
                worker_label=worker_label,
                pid=pid,
                state="running",
                gpu_ids=tuple(int(gpu_id) for gpu_id in gpu_ids),
                started_at=_iso_utc_timestamp(now),
                started_at_epoch=now,
            )
            if self._config.record_case_start_events:
                self._events.append(
                    {
                        "timestamp": _iso_utc_timestamp(now),
                        "event": "case_started",
                        "case_id": case_id,
                        "worker_label": worker_label,
                        "pid": pid,
                        "gpu_ids": [int(gpu_id) for gpu_id in gpu_ids],
                    }
                )

    def complete_worker(
        self,
        *,
        pid: int,
        result: Mapping[str, object],
    ) -> None:
        """Mark a worker as finished and emit a terminal event."""
        now = self._time_source()
        with self._lock:
            worker = self._workers.pop(pid, None)
            failure_kind = result.get("failure_kind")
            event_name = "case_finished"
            if failure_kind == "timeout":
                event_name = "case_timeout"
            elif failure_kind in {
                "worker_terminated",
                "process_exit",
                "process_signal",
                "no_completion",
            }:
                event_name = "worker_terminated"

            should_record_event = (
                event_name != "case_finished"
                or self._config.record_case_finish_events
            )
            if should_record_event:
                self._events.append(
                    {
                        "timestamp": _iso_utc_timestamp(now),
                        "event": event_name,
                        "case_id": None if worker is None else worker.case_id,
                        "worker_label": None if worker is None else worker.worker_label,
                        "pid": pid,
                        "status": result.get("status", "unknown"),
                        "failure_kind": failure_kind,
                    }
                )

    def collect_once(self) -> dict[str, object]:
        """Collect one fresh snapshot and store it in the in-memory history."""
        timestamp_epoch = self._time_source()
        host_metrics = self._collect_host_metrics()
        gpu_metrics = self._collect_gpu_metrics()

        with self._lock:
            self._cached_host_metrics = host_metrics
            self._cached_gpu_metrics = gpu_metrics
            snapshot = self._build_snapshot_locked(timestamp_epoch)
            self._latest_snapshot = snapshot
            self._history.append(snapshot)
            return deepcopy(snapshot)

    def snapshot(self) -> dict[str, object]:
        """Return the latest known snapshot."""
        with self._lock:
            if self._latest_snapshot is None:
                return deepcopy(self._build_snapshot_locked(self._time_source()))
            return deepcopy(self._latest_snapshot)

    def history(self, *, limit: int | None = None) -> list[dict[str, object]]:
        """Return recent runtime snapshots."""
        with self._lock:
            values = list(self._history)
        if limit is not None:
            values = values[-limit:]
        return deepcopy(values)

    def events(self, *, limit: int | None = None) -> list[dict[str, object]]:
        """Return recent monitor events."""
        with self._lock:
            values = list(self._events)
        if limit is not None:
            values = values[-limit:]
        return deepcopy(values)

    def render_dashboard_html(self) -> str:
        """Render the tiny HTML dashboard served by the monitor."""
        return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Experiment Runner Monitor</title>
  <style>
    :root { color-scheme: light; }
    body { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin: 24px; background: #f5f7fb; color: #142033; }
    h1 { margin-bottom: 8px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
    .card { background: white; border: 1px solid #d8dfec; border-radius: 12px; padding: 16px; box-shadow: 0 8px 24px rgba(20, 32, 51, 0.06); }
    pre { white-space: pre-wrap; word-break: break-word; margin: 0; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; border-bottom: 1px solid #e5eaf3; padding: 6px 4px; }
    .muted { color: #53627c; }
  </style>
</head>
<body>
  <h1>Experiment Runner Monitor</h1>
  <p class="muted">Lightweight local dashboard backed by <code>/api/v1/snapshot</code>.</p>
  <div class="grid">
    <section class="card">
      <h2>Runner</h2>
      <pre id="runner">loading...</pre>
    </section>
    <section class="card">
      <h2>Host</h2>
      <pre id="host">loading...</pre>
    </section>
    <section class="card">
      <h2>Workers</h2>
      <div id="workers">loading...</div>
    </section>
    <section class="card">
      <h2>GPUs</h2>
      <div id="gpus">loading...</div>
    </section>
    <section class="card">
      <h2>Events</h2>
      <pre id="events">loading...</pre>
    </section>
  </div>
  <script>
    function renderTable(items, columns) {
      if (!items.length) {
        return "<p>No data</p>";
      }
      const head = "<tr>" + columns.map((column) => `<th>${column.label}</th>`).join("") + "</tr>";
      const rows = items.map((item) => (
        "<tr>" + columns.map((column) => `<td>${item[column.key] ?? ""}</td>`).join("") + "</tr>"
      )).join("");
      return `<table><thead>${head}</thead><tbody>${rows}</tbody></table>`;
    }

    async function refresh() {
      const [snapshotResponse, eventsResponse] = await Promise.all([
        fetch("/api/v1/snapshot"),
        fetch("/api/v1/events?limit=10"),
      ]);
      const snapshot = await snapshotResponse.json();
      const events = await eventsResponse.json();
      document.getElementById("runner").textContent = JSON.stringify(snapshot.runner, null, 2);
      document.getElementById("host").textContent = JSON.stringify(snapshot.host, null, 2);
      document.getElementById("workers").innerHTML = renderTable(snapshot.workers, [
        { key: "case_id", label: "case_id" },
        { key: "worker_label", label: "worker" },
        { key: "pid", label: "pid" },
        { key: "elapsed_seconds", label: "elapsed_s" },
      ]);
      document.getElementById("gpus").innerHTML = renderTable(snapshot.gpus, [
        { key: "gpu_id", label: "gpu" },
        { key: "utilization_gpu_percent", label: "util%" },
        { key: "memory_used_bytes", label: "mem_used" },
        { key: "temperature_c", label: "temp_c" },
      ]);
      document.getElementById("events").textContent = JSON.stringify(events, null, 2);
    }

    refresh().catch((error) => {
      document.body.insertAdjacentHTML("beforeend", `<pre>${String(error)}</pre>`);
    });
    setInterval(() => { refresh().catch(() => {}); }, 1000);
  </script>
</body>
</html>"""

    def _collector_loop(self) -> None:
        while True:
            interval = self.sample_interval_seconds
            if self._stop_event.wait(interval):
                break
            self.collect_once()

    def _collect_host_metrics(self) -> dict[str, object]:
        if self._host_metrics_collector is not None:
            try:
                return dict(self._host_metrics_collector())
            except Exception as exc:  # pragma: no cover - defensive
                self._record_collector_error("host_metrics", exc)
                return deepcopy(self._cached_host_metrics)

        current = _read_cpu_counters(self._proc_root)
        total_bytes, available_bytes = _read_meminfo(self._proc_root)
        cpu_percent = 0.0
        if current is not None and self._previous_cpu_counters is not None:
            delta_total = current.total - self._previous_cpu_counters.total
            delta_idle = current.idle - self._previous_cpu_counters.idle
            if delta_total > 0:
                cpu_percent = max(
                    0.0,
                    min(100.0, 100.0 * (delta_total - delta_idle) / delta_total),
                )
        self._previous_cpu_counters = current
        used_bytes = max(total_bytes - available_bytes, 0)
        return {
            "cpu_percent": round(cpu_percent, 2),
            "memory_total_bytes": total_bytes,
            "memory_available_bytes": available_bytes,
            "memory_used_bytes": used_bytes,
        }

    def _collect_gpu_metrics(self) -> list[dict[str, object]]:
        collector = _default_gpu_metrics_collector
        if self._gpu_metrics_collector is not None:
            collector = self._gpu_metrics_collector
        try:
            return [dict(item) for item in collector()]
        except Exception as exc:  # pragma: no cover - defensive
            self._record_collector_error("gpu_metrics", exc)
            return deepcopy(self._cached_gpu_metrics)

    def _record_collector_error(self, collector_name: str, exc: Exception, /) -> None:
        now = self._time_source()
        with self._lock:
            self._events.append(
                {
                    "timestamp": _iso_utc_timestamp(now),
                    "event": "collector_error",
                    "collector": collector_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    def _build_snapshot_locked(self, timestamp_epoch: float, /) -> dict[str, object]:
        workers: list[dict[str, object]] = []
        for worker in sorted(
            self._workers.values(),
            key=lambda item: (item.worker_label, item.pid),
        ):
            workers.append(
                {
                    "case_id": worker.case_id,
                    "worker_label": worker.worker_label,
                    "pid": worker.pid,
                    "state": worker.state,
                    "gpu_ids": list(worker.gpu_ids),
                    "started_at": worker.started_at,
                    "elapsed_seconds": round(timestamp_epoch - worker.started_at_epoch, 3),
                }
            )

        return {
            "timestamp": _iso_utc_timestamp(timestamp_epoch),
            "monitor": {
                "mode": self.mode,
                "sample_interval_seconds": self.sample_interval_seconds,
                "history_limit": self._config.history_limit,
                "event_limit": self._config.event_limit,
            },
            "runner": dict(self._runner_state),
            "host": dict(self._cached_host_metrics),
            "gpus": deepcopy(self._cached_gpu_metrics),
            "workers": workers,
        }
