from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Mapping, cast

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import pytest

from experiment_runner.context_utils import apply_environment_variables
from experiment_runner.protocols import TaskContext
from experiment_runner.resource_scheduler import (
    GPUDeviceCapacity,
    GPUEnvironmentConfig,
    FullResourceCapacity,
    FullResourceEstimate,
    StandardFullResourceScheduler,
    detect_gpu_devices,
    detect_host_memory_bytes,
)
from experiment_runner.runner import StandardRunner, StandardWorker


SOURCE_FILE = Path(__file__).name
_GPU_ALLOCATOR_ENV_NAMES = (
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "XLA_PYTHON_CLIENT_ALLOCATOR",
    "TF_GPU_ALLOCATOR",
    "XLA_PYTHON_CLIENT_USE_CUDA_HOST_ALLOCATOR",
)


def _discover_gpu_indices() -> list[int]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    return [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]


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


def _result_int(result: Mapping[str, object], key: str, /) -> int:
    value = result.get(key)
    if not isinstance(value, int):
        raise TypeError(f"{key} must be int.")
    return value


def _result_float(result: Mapping[str, object], key: str, /) -> float:
    value = result.get(key)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be float.")
    return float(value)


def _result_str(result: Mapping[str, object], key: str, /) -> str:
    value = result.get(key)
    if not isinstance(value, str):
        raise TypeError(f"{key} must be str.")
    return value


def _result_int_list(result: Mapping[str, object], key: str, /) -> list[int]:
    value = result.get(key)
    if not isinstance(value, list):
        raise TypeError(f"{key} must be list.")
    if not all(isinstance(item, int) for item in value):
        raise TypeError(f"{key} must contain ints.")
    return cast(list[int], value)


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


def _skip_if_gpu_backend_unavailable(results: list[Mapping[str, object]]) -> None:
    failed_results = [result for result in results if result.get("status") != "ok"]
    if not failed_results:
        return

    unavailable_markers = (
        "CUDA_ERROR_OUT_OF_MEMORY",
        "Unable to initialize backend 'cuda'",
        "no supported devices found for platform CUDA",
    )
    if all(
        isinstance(result.get("traceback"), str)
        and any(marker in cast(str, result["traceback"]) for marker in unavailable_markers)
        for result in failed_results
    ):
        pytest.skip("GPU backend was visible but unavailable for fresh child workers.")


def _clear_allocator_environment() -> None:
    for env_name in _GPU_ALLOCATOR_ENV_NAMES:
        os.environ.pop(env_name, None)


def _string_or_none(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("value must be str | None.")
    return value


def _resource_estimate(case: Mapping[str, object], /) -> FullResourceEstimate:
    return FullResourceEstimate(
        host_memory_bytes=int(case.get("host_memory_bytes", 0)),
        gpu_count=int(case.get("gpu_count", 0)),
        gpu_memory_bytes=int(case.get("gpu_memory_bytes", 0)),
        gpu_slots=int(case.get("gpu_slots", 1)),
    )


@dataclass(frozen=True)
class _GpuAllocatorProfile:
    name: str
    disable_gpu_preallocation: bool = False
    gpu_environment_config: GPUEnvironmentConfig | None = None


@dataclass(frozen=True)
class _HeavyGpuRunnerTask:
    records_dir: Path

    def __call__(self, case: dict[str, object], context: TaskContext) -> dict[str, object]:
        env_vars = cast(dict[str, str], context.get("environment_variables", {}))
        assigned_gpu_id = int(env_vars.get("gpu_id", "-1"))
        _clear_allocator_environment()
        apply_environment_variables(context)

        started_at = time.time()
        try:
            import jax
            import jax.numpy as jnp

            target_device = jax.devices("gpu")[0]
            matrix_size = int(case.get("matrix_size", 1024))
            min_work_seconds = float(case.get("min_work_seconds", 2.0))

            with jax.default_device(target_device):
                lhs = jnp.ones((matrix_size, matrix_size), dtype=jnp.float32)
                rhs = jnp.eye(matrix_size, dtype=jnp.float32)
                acc = lhs

            work_started = time.perf_counter()
            iterations = 0
            while time.perf_counter() - work_started < min_work_seconds:
                acc = jnp.tanh(acc @ rhs)
                jax.block_until_ready(acc)
                iterations += 1
            finished_at = time.time()

            record = {
                "status": "ok",
                "case_id": int(case["case_id"]),
                "assigned_gpu_id": assigned_gpu_id,
                "cuda_visible_devices": env_vars.get("CUDA_VISIBLE_DEVICES", ""),
                "gpu_ids": env_vars.get("gpu_ids", ""),
                "pid": os.getpid(),
                "backend": jax.default_backend(),
                "visible_gpu_ids": [int(device.id) for device in jax.devices("gpu")],
                "gpu_device_count": len(jax.devices("gpu")),
                "xla_preallocate": _string_or_none(
                    os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
                ),
                "xla_mem_fraction": _string_or_none(
                    os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION")
                ),
                "xla_client_allocator": _string_or_none(
                    os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR")
                ),
                "tf_gpu_allocator": _string_or_none(os.environ.get("TF_GPU_ALLOCATOR")),
                "xla_use_cuda_host_allocator": _string_or_none(
                    os.environ.get("XLA_PYTHON_CLIENT_USE_CUDA_HOST_ALLOCATOR")
                ),
                "matrix_size": matrix_size,
                "iterations": iterations,
                "work_seconds": finished_at - started_at,
                "started_at": started_at,
                "finished_at": finished_at,
                "checksum": float(jax.device_get(acc[0, 0])),
            }
            _write_json(self.records_dir / f"case_{record['case_id']}.json", record)
            return record
        except Exception as exc:
            finished_at = time.time()
            record = {
                "status": "failed",
                "case_id": int(case["case_id"]),
                "assigned_gpu_id": assigned_gpu_id,
                "cuda_visible_devices": env_vars.get("CUDA_VISIBLE_DEVICES", ""),
                "gpu_ids": env_vars.get("gpu_ids", ""),
                "pid": os.getpid(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "started_at": started_at,
                "finished_at": finished_at,
            }
            _write_json(self.records_dir / f"case_{record['case_id']}.json", record)
            raise


def _build_profile_summaries(
    profile: _GpuAllocatorProfile,
    cases: list[dict[str, object]],
    records: list[dict[str, object]],
    elapsed_seconds: float,
    expected_slots_per_gpu: int,
) -> dict[str, object]:
    case_ids = [int(case["case_id"]) for case in cases]
    records_by_case_id = {_result_int(record, "case_id"): record for record in records}
    missing_case_ids = [
        case_id for case_id in case_ids if case_id not in records_by_case_id
    ]
    successful_records = [
        record for record in records if _result_str(record, "status") == "ok"
    ]
    failed_records = [
        record for record in records if _result_str(record, "status") != "ok"
    ]

    peak_active_workers = 0
    peak_workers_per_gpu = 0
    for midpoint in _midpoints(successful_records):
        active_records = _active_records_at(successful_records, midpoint)
        peak_active_workers = max(peak_active_workers, len(active_records))
        workers_by_gpu: dict[int, int] = {}
        for record in active_records:
            gpu_id = _result_int(record, "assigned_gpu_id")
            workers_by_gpu[gpu_id] = workers_by_gpu.get(gpu_id, 0) + 1
        if workers_by_gpu:
            peak_workers_per_gpu = max(peak_workers_per_gpu, max(workers_by_gpu.values()))
        assert all(count <= expected_slots_per_gpu for count in workers_by_gpu.values())

    for record in successful_records:
        assert _result_int(record, "gpu_device_count") == 1
        assert _result_int_list(record, "visible_gpu_ids") == [0]

    return {
        "profile": profile.name,
        "disable_gpu_preallocation": profile.disable_gpu_preallocation,
        "elapsed_seconds": elapsed_seconds,
        "total_cases": len(cases),
        "success_count": len(successful_records),
        "failure_count": len(failed_records),
        "missing_case_ids": missing_case_ids,
        "peak_active_workers": peak_active_workers,
        "peak_workers_per_gpu": peak_workers_per_gpu,
        "successful_allocator_envs": [
            {
                "case_id": _result_int(record, "case_id"),
                "assigned_gpu_id": _result_int(record, "assigned_gpu_id"),
                "xla_preallocate": record.get("xla_preallocate"),
                "xla_mem_fraction": record.get("xla_mem_fraction"),
                "xla_client_allocator": record.get("xla_client_allocator"),
                "tf_gpu_allocator": record.get("tf_gpu_allocator"),
                "xla_use_cuda_host_allocator": record.get(
                    "xla_use_cuda_host_allocator"
                ),
            }
            for record in successful_records
        ],
        "failed_cases": [
            {
                "case_id": _result_int(record, "case_id"),
                "assigned_gpu_id": _result_int(record, "assigned_gpu_id"),
                "error": record.get("error"),
            }
            for record in failed_records
        ],
    }


def _allocator_profiles() -> list[_GpuAllocatorProfile]:
    return [
        _GpuAllocatorProfile(name="jax_default"),
        _GpuAllocatorProfile(
            name="mem_fraction_0_4",
            gpu_environment_config=GPUEnvironmentConfig(memory_fraction=0.4),
        ),
        _GpuAllocatorProfile(
            name="preallocate_off",
            disable_gpu_preallocation=True,
        ),
        _GpuAllocatorProfile(
            name="platform_allocator",
            gpu_environment_config=GPUEnvironmentConfig(xla_client_allocator="platform"),
        ),
        _GpuAllocatorProfile(
            name="cuda_malloc_async",
            gpu_environment_config=GPUEnvironmentConfig(
                tf_gpu_allocator="cuda_malloc_async"
            ),
        ),
        _GpuAllocatorProfile(
            name="cuda_host_allocator_off",
            gpu_environment_config=GPUEnvironmentConfig(
                use_cuda_host_allocator=False
            ),
        ),
        _GpuAllocatorProfile(
            name="combined_safe",
            gpu_environment_config=GPUEnvironmentConfig(
                disable_preallocation=True,
                memory_fraction=0.4,
                xla_client_allocator="platform",
                tf_gpu_allocator="cuda_malloc_async",
                use_cuda_host_allocator=False,
            ),
        ),
    ]


@pytest.mark.skipif(
    os.environ.get("RUN_HEAVY_TESTS") != "1",
    reason="set RUN_HEAVY_TESTS=1 to run heavy GPU worker tests",
)
def test_standard_runner_compares_gpu_allocator_profiles_for_shared_gpu_slots(
    tmp_path: Path,
) -> None:
    gpu_indices = _discover_gpu_indices()
    if not gpu_indices:
        pytest.skip("requires at least one GPU")

    selected_gpu_indices = gpu_indices[: min(2, len(gpu_indices))]
    all_gpu_devices = {gpu_device.gpu_id: gpu_device for gpu_device in detect_gpu_devices()}
    selected_gpu_devices = tuple(
        all_gpu_devices[gpu_id]
        for gpu_id in selected_gpu_indices
        if gpu_id in all_gpu_devices
    )
    if not selected_gpu_devices:
        pytest.skip("detect_gpu_devices() did not resolve selected GPUs")

    shared_gpu_devices = tuple(
        GPUDeviceCapacity(
            gpu_id=gpu_device.gpu_id,
            memory_bytes=gpu_device.memory_bytes,
            max_slots=2,
        )
        for gpu_device in selected_gpu_devices
    )
    profiles = _allocator_profiles()
    profile_summaries: list[dict[str, object]] = []

    for profile in profiles:
        records_dir = tmp_path / profile.name / "records"
        cases = [
            {
                "case_id": case_id,
                "matrix_size": 1024,
                "min_work_seconds": 1.5,
                "host_memory_bytes": 256 * 1024 * 1024,
                "gpu_count": 1,
                "gpu_memory_bytes": 512 * 1024 * 1024,
                "gpu_slots": 1,
            }
            for case_id in range(len(shared_gpu_devices) * 2)
        ]
        scheduler = StandardFullResourceScheduler(
            resource_capacity=FullResourceCapacity(
                max_workers=len(shared_gpu_devices) * 2,
                host_memory_bytes=detect_host_memory_bytes(),
                gpu_devices=shared_gpu_devices,
            ),
            cases=cases,
            estimate_builder=_resource_estimate,
            disable_gpu_preallocation=profile.disable_gpu_preallocation,
            gpu_environment_config=profile.gpu_environment_config,
        )
        runner = StandardRunner(scheduler)
        worker = StandardWorker(_HeavyGpuRunnerTask(records_dir))

        started_at = time.perf_counter()
        runner.run(worker)
        elapsed_seconds = time.perf_counter() - started_at
        records = _read_json_records(records_dir)
        if records:
            _skip_if_gpu_backend_unavailable(records)
            assert {
                _result_int(record, "assigned_gpu_id")
                for record in records
                if record.get("status") == "ok"
            }.issubset(set(selected_gpu_indices))
            assert all(
                _result_float(record, "work_seconds") >= 1.0
                for record in records
                if record.get("status") == "ok"
            )
        profile_summaries.append(
            _build_profile_summaries(
                profile=profile,
                cases=cases,
                records=records,
                elapsed_seconds=elapsed_seconds,
                expected_slots_per_gpu=2,
            )
        )

    successful_profile_count = sum(
        1
        for summary in profile_summaries
        if int(summary["failure_count"]) == 0 and not cast(list[int], summary["missing_case_ids"])
    )
    assert successful_profile_count >= 1

    summary = {
        "case": "experiment_runner_gpu_allocator_profile_comparison",
        "source_file": SOURCE_FILE,
        "test": "test_standard_runner_compares_gpu_allocator_profiles_for_shared_gpu_slots",
        "selected_gpu_indices": selected_gpu_indices,
        "profiles": profile_summaries,
    }
    results_path = Path(
        os.environ.get(
            "GPU_ALLOCATOR_SWEEP_RESULTS_PATH",
            str(tmp_path / "gpu_allocator_profile_summary.json"),
        )
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, sort_keys=True))
