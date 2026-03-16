from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


CHILD_COMPLETE_PREFIX = "EXPERIMENT_RUNNER_COMPLETE "


@dataclass(frozen=True)
class WorkerSlot:
    worker_label: str
    gpu_index: int | None
    gpu_slot: int
    cpu_affinity: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class _RunningChild:
    case: Mapping[str, object]
    worker_slot: WorkerSlot
    process: subprocess.Popen[str]
    started_at: float


def _mapping_int(payload: Mapping[str, object], key: str, /) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise TypeError(f"{key} must be int.")
    return value


def _mapping_str(payload: Mapping[str, object], key: str, /) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be str.")
    return value


def worker_slot_from_mapping(payload: Mapping[str, object], /) -> WorkerSlot:
    gpu_index_value = payload.get("gpu_index")
    cpu_affinity_value = payload.get("cpu_affinity")
    cpu_affinity = tuple(int(cpu) for cpu in cpu_affinity_value) if isinstance(cpu_affinity_value, list) else ()
    return WorkerSlot(
        worker_label=_mapping_str(payload, "worker_label"),
        gpu_index=int(gpu_index_value) if isinstance(gpu_index_value, int) else None,
        gpu_slot=_mapping_int(payload, "gpu_slot"),
        cpu_affinity=cpu_affinity,
    )


def _available_cpu_indices() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    count = os.cpu_count() or 1
    return list(range(count))


def _partition_cpu_indices(num_workers: int, /) -> list[list[int]]:
    cpu_indices = _available_cpu_indices()
    if num_workers < 1:
        raise ValueError("num_workers must be positive.")
    if not cpu_indices:
        return [[] for _ in range(num_workers)]

    base = len(cpu_indices) // num_workers
    extra = len(cpu_indices) % num_workers
    groups: list[list[int]] = []
    start = 0
    for worker_index in range(num_workers):
        group_size = base + (1 if worker_index < extra else 0)
        if group_size <= 0:
            groups.append([cpu_indices[worker_index % len(cpu_indices)]])
            continue
        stop = start + group_size
        groups.append(cpu_indices[start:stop])
        start = stop
    return groups


def build_worker_slots(
    platform: str,
    gpu_indices: Sequence[int],
    workers_per_gpu: int,
    /,
) -> list[WorkerSlot]:
    if platform == "gpu":
        if workers_per_gpu < 1:
            raise ValueError("workers_per_gpu must be positive.")
        total_workers = len(gpu_indices) * workers_per_gpu
        cpu_groups = _partition_cpu_indices(total_workers) if total_workers > 0 else []
        worker_slots: list[WorkerSlot] = []
        worker_index = 0
        for gpu_index in gpu_indices:
            for gpu_slot in range(workers_per_gpu):
                cpu_affinity = tuple(cpu_groups[worker_index] if worker_index < len(cpu_groups) else [])
                worker_slots.append(
                    WorkerSlot(
                        worker_label=f"gpu-{gpu_index}-w{gpu_slot}",
                        gpu_index=int(gpu_index),
                        gpu_slot=gpu_slot,
                        cpu_affinity=cpu_affinity,
                    )
                )
                worker_index += 1
        return worker_slots

    return [
        WorkerSlot(
            worker_label="cpu-0",
            gpu_index=None,
            gpu_slot=0,
            cpu_affinity=tuple(_available_cpu_indices()),
        )
    ]


def json_compatible(value: object, /) -> object:
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        array_value = np.asarray(value)
        if array_value.ndim == 0:
            return array_value.item()
        return array_value.tolist()
    return value


def append_jsonl_record(output_path: Path, record: Mapping[str, object], /) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(json.dumps(json_compatible(dict(record)), ensure_ascii=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def apply_worker_environment(
    *,
    platform: str,
    worker_slot: WorkerSlot,
    disable_gpu_preallocation: bool,
) -> None:
    if disable_gpu_preallocation:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    if worker_slot.cpu_affinity and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(worker_slot.cpu_affinity))

    if platform == "gpu":
        os.environ.pop("JAX_PLATFORMS", None)
        if worker_slot.gpu_index is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_slot.gpu_index)
            os.environ["SMOLYAK_GPU_INDEX"] = str(worker_slot.gpu_index)
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["SMOLYAK_GPU_INDEX"] = "cpu"

    os.environ["EXPERIMENT_RUNNER_WORKER_LABEL"] = worker_slot.worker_label
    os.environ["EXPERIMENT_RUNNER_GPU_SLOT"] = str(worker_slot.gpu_slot)
    os.environ["EXPERIMENT_RUNNER_CPU_AFFINITY"] = ",".join(str(cpu) for cpu in worker_slot.cpu_affinity)


def _extract_completion_record(stdout: str, /) -> dict[str, object] | None:
    completion: dict[str, object] | None = None
    for line in stdout.splitlines():
        if not line.startswith(CHILD_COMPLETE_PREFIX):
            continue
        payload = line[len(CHILD_COMPLETE_PREFIX):]
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            completion = parsed
    return completion


def _terminate_process(process: subprocess.Popen[str], /) -> tuple[str, str]:
    if process.poll() is None:
        process.terminate()
        try:
            return process.communicate(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
    return process.communicate()


def run_cases_with_subprocess_scheduler(
    cases: Sequence[Mapping[str, object]],
    worker_slots: Sequence[WorkerSlot],
    /,
    *,
    timeout_seconds: int,
    build_child_command: Callable[[Mapping[str, object], WorkerSlot], Sequence[str]],
    build_parent_failure_result: Callable[[Mapping[str, object], WorkerSlot, str, str, str], dict[str, object]],
    fallback_jsonl_output_path: Path | None,
    cwd: Path,
    on_case_started: Callable[[Mapping[str, object], WorkerSlot], None] | None = None,
    on_case_finished: Callable[[Mapping[str, object], WorkerSlot, Mapping[str, object]], None] | None = None,
    poll_interval_seconds: float = 0.05,
) -> list[dict[str, object]]:
    if not worker_slots:
        raise ValueError("worker_slots must not be empty.")

    pending_cases = deque(cases)
    free_slots = deque(worker_slots)
    running_children: dict[int, _RunningChild] = {}
    all_results: list[dict[str, object]] = []

    while pending_cases or running_children:
        while pending_cases and free_slots:
            case = pending_cases.popleft()
            worker_slot = free_slots.popleft()
            if on_case_started is not None:
                on_case_started(case, worker_slot)

            command = list(build_child_command(case, worker_slot))
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            running_children[process.pid] = _RunningChild(
                case=case,
                worker_slot=worker_slot,
                process=process,
                started_at=time.monotonic(),
            )

        finished_pids: list[int] = []
        now = time.monotonic()
        for pid, child in running_children.items():
            timeout_elapsed = now - child.started_at
            if timeout_elapsed > timeout_seconds:
                stdout_text, stderr_text = _terminate_process(child.process)
                result = build_parent_failure_result(
                    child.case,
                    child.worker_slot,
                    "timeout",
                    f"TimeoutError: case exceeded timeout_seconds={timeout_seconds}.",
                    (stdout_text + "\n" + stderr_text).strip(),
                )
                if fallback_jsonl_output_path is not None:
                    append_jsonl_record(fallback_jsonl_output_path, result)
                all_results.append(result)
                if on_case_finished is not None:
                    on_case_finished(child.case, child.worker_slot, result)
                free_slots.append(child.worker_slot)
                finished_pids.append(pid)
                continue

            returncode = child.process.poll()
            if returncode is None:
                continue

            stdout_text, stderr_text = child.process.communicate()
            completion = _extract_completion_record(stdout_text)
            if completion is None:
                error = f"Child exited without completion record (returncode={returncode})."
                traceback_text = (stdout_text + "\n" + stderr_text).strip()
                result = build_parent_failure_result(
                    child.case,
                    child.worker_slot,
                    "worker_terminated",
                    error,
                    traceback_text,
                )
                if fallback_jsonl_output_path is not None:
                    append_jsonl_record(fallback_jsonl_output_path, result)
            else:
                result = completion

            all_results.append(result)
            if on_case_finished is not None:
                on_case_finished(child.case, child.worker_slot, result)
            free_slots.append(child.worker_slot)
            finished_pids.append(pid)

        for pid in finished_pids:
            running_children.pop(pid, None)

        if running_children and not finished_pids:
            time.sleep(poll_interval_seconds)

    return all_results
