# Results branch: results/functional-smolyak-scaling
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
from numpy.typing import NDArray


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = WORKSPACE_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from experiment_runner.execution_result import ExecutionResult, FailureKind
from experiment_runner.monitor import RuntimeMonitor
from experiment_runner.protocols import SkipController, TaskContext
from experiment_runner.resource_scheduler import (
    FullResourceCapacity,
    FullResourceEstimate,
    StandardFullResourceScheduler,
    detect_gpu_devices,
)
from experiment_runner.result_io import (
    append_jsonl_record,
    json_compatible,
    read_jsonl_records,
)
from experiment_runner.runner import StandardRunner, StandardWorker
from jax_util.xla_env import build_cpu_env, build_env_for_profile


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_BRANCH_NAME = "results/functional-smolyak-scaling"
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_NUM_REPEATS = 5
DEFAULT_NUM_ACCURACY_PROBLEMS = 1000
DEFAULT_COEFF_START = -0.55
DEFAULT_COEFF_STOP = 0.65
DEFAULT_WORKERS_PER_GPU = 1
DEFAULT_CHUNK_SIZE = 16384
SUPPORTED_FLOAT_DTYPES = ("float16", "bfloat16", "float32", "float64")
SUPPORTED_INTEGRATION_METHODS = ("smolyak", "monte_carlo")
SUPPORTED_EXECUTION_VARIANTS = ("single", "vmap")
GPU_PREALLOCATION_DISABLED = True


class _CasePhaseError(RuntimeError):
    def __init__(self, phase: str, cause: BaseException):
        super().__init__(str(cause))
        self.phase = phase


@dataclass
class _TimeoutFrontierSkipController(SkipController[dict[str, object]]):
    timed_out_levels: dict[tuple[int, str, str, str], int] = field(default_factory=dict)

    def _key(self, case: Mapping[str, object], /) -> tuple[int, str, str, str]:
        return (
            _case_int(case, "dimension"),
            str(case["dtype_name"]),
            _case_method(case),
            _case_variant(case),
        )

    def should_skip(self, case: dict[str, object], context: TaskContext) -> str | None:
        del context
        key = self._key(case)
        frontier_level = self.timed_out_levels.get(key)
        if frontier_level is None:
            return None
        current_level = _case_int(case, "level")
        if current_level >= frontier_level:
            return (
                "skip due to prior timeout frontier at "
                f"dimension={key[0]} dtype={key[1]} method={key[2]} "
                f"variant={key[3]} level={frontier_level}"
            )
        return None

    def update(
        self,
        case: dict[str, object],
        context: TaskContext,
        result: ExecutionResult,
    ) -> None:
        del context
        if result.status != "failed":
            return
        if result.failure_kind != FailureKind.TIMEOUT.value:
            return
        key = self._key(case)
        current_level = _case_int(case, "level")
        previous = self.timed_out_levels.get(key)
        if previous is None or current_level < previous:
            self.timed_out_levels[key] = current_level


def _parse_optional_bool_flag(value: str | None, /) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean flag: {value}")


def _parse_integer_range(spec: str, /) -> list[int]:
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("Range must be start:end or start:end:step.")
    start = int(parts[0])
    stop = int(parts[1])
    step = int(parts[2]) if len(parts) == 3 else 1
    if step <= 0:
        raise ValueError("Range step must be positive.")
    if start > stop:
        raise ValueError("Range start must be <= stop.")
    return list(range(start, stop + 1, step))


def _parse_gpu_indices(spec: str, /) -> list[int]:
    values = [item.strip() for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("GPU index list must not be empty.")
    return [int(value) for value in values]


def _parse_dtype_names(spec: str, /) -> list[str]:
    if spec.strip().lower() == "all":
        return list(SUPPORTED_FLOAT_DTYPES)
    aliases = {
        "f16": "float16",
        "bf16": "bfloat16",
        "f32": "float32",
        "f64": "float64",
    }
    normalized: list[str] = []
    for raw_name in spec.split(","):
        name = raw_name.strip().lower()
        if not name:
            continue
        canonical = aliases.get(name, name)
        if canonical not in SUPPORTED_FLOAT_DTYPES:
            raise ValueError(f"Unsupported dtype: {raw_name}")
        if canonical not in normalized:
            normalized.append(canonical)
    if not normalized:
        raise ValueError("dtype list must not be empty.")
    return normalized


def _parse_execution_variants(spec: str, /) -> list[str]:
    if spec.strip().lower() == "all":
        return list(SUPPORTED_EXECUTION_VARIANTS)
    normalized: list[str] = []
    for raw_name in spec.split(","):
        name = raw_name.strip().lower()
        if not name:
            continue
        if name not in SUPPORTED_EXECUTION_VARIANTS:
            raise ValueError(f"Unsupported execution variant: {raw_name}")
        if name not in normalized:
            normalized.append(name)
    if not normalized:
        raise ValueError("execution variant list must not be empty.")
    return normalized


def _parse_integration_methods(spec: str, /) -> list[str]:
    if spec.strip().lower() == "all":
        return list(SUPPORTED_INTEGRATION_METHODS)
    aliases = {"mc": "monte_carlo"}
    normalized: list[str] = []
    for raw_name in spec.split(","):
        name = raw_name.strip().lower()
        if not name:
            continue
        canonical = aliases.get(name, name)
        if canonical not in SUPPORTED_INTEGRATION_METHODS:
            raise ValueError(f"Unsupported integration method: {raw_name}")
        if canonical not in normalized:
            normalized.append(canonical)
    if not normalized:
        raise ValueError("integration method list must not be empty.")
    return normalized


def _discover_gpu_indices() -> list[int]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return []
    return [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]


def _git_stdout(args: list[str], /) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    stdout = completed.stdout.strip()
    return stdout if stdout else None


def _experiment_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        "results_branch": RESULTS_BRANCH_NAME,
        "worktree_path": str(WORKSPACE_ROOT),
        "script_path": str(Path(__file__).resolve()),
        "gpu_preallocation_disabled": GPU_PREALLOCATION_DISABLED,
    }
    branch = _git_stdout(["branch", "--show-current"])
    commit = _git_stdout(["rev-parse", "HEAD"])
    if branch is not None:
        metadata["git_branch"] = branch
    if commit is not None:
        metadata["git_commit"] = commit
    return metadata


def _xla_config_from_run_config(run_config: Mapping[str, object], /) -> dict[str, object]:
    return {
        "memory_fraction": run_config.get("xla_memory_fraction"),
        "allocator": run_config.get("xla_allocator"),
        "tf_gpu_allocator": run_config.get("xla_tf_gpu_allocator"),
        "use_cuda_host_allocator": run_config.get("xla_use_cuda_host_allocator"),
        "memory_scheduler": run_config.get("xla_memory_scheduler"),
        "while_loop_double_buffering": run_config.get(
            "xla_gpu_enable_while_loop_double_buffering"
        ),
        "latency_hiding_scheduler_rerun": run_config.get(
            "xla_latency_hiding_scheduler_rerun"
        ),
        "jax_compiler_enable_remat_pass": run_config.get(
            "jax_compiler_enable_remat_pass"
        ),
    }


def _build_cases(
    dimensions: list[int],
    levels: list[int],
    dtype_names: list[str],
    integration_methods: list[str],
    execution_variants: list[str],
    /,
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    case_id = 0
    for dimension in dimensions:
        for level in levels:
            same_budget_num_points = _same_budget_num_points(dimension, level)
            for dtype_name in dtype_names:
                for integration_method_index, integration_method in enumerate(integration_methods):
                    for execution_variant_index, execution_variant in enumerate(execution_variants):
                        cases.append({
                            "case_id": case_id,
                            "dimension": dimension,
                            "level": level,
                            "dtype_name": dtype_name,
                            "integration_method": integration_method,
                            "integration_method_index": integration_method_index,
                            "execution_variant": execution_variant,
                            "execution_variant_index": execution_variant_index,
                            "mode": f"{integration_method}_{execution_variant}",
                            "budget_kind": "same_budget",
                            "budget_value": same_budget_num_points,
                            "budget_reference": "smolyak_num_points",
                            "same_budget_num_points": same_budget_num_points,
                        })
                        case_id += 1
    return cases


def _case_int(case: Mapping[str, object], key: str, /) -> int:
    value = case[key]
    if not isinstance(value, int):
        raise TypeError(f"{key} must be int.")
    return value


def _result_int(result: Mapping[str, object], key: str, /) -> int:
    value = result[key]
    if not isinstance(value, int):
        raise TypeError(f"{key} must be int.")
    return value


def _result_float(result: Mapping[str, object], key: str, /) -> float:
    value = result[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be float.")
    return float(value)


def _config_int(config: Mapping[str, object], key: str, /) -> int:
    value = config[key]
    if not isinstance(value, int):
        raise TypeError(f"{key} must be int.")
    return value


def _config_float(config: Mapping[str, object], key: str, /) -> float:
    value = config[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be float.")
    return float(value)


def _config_str(config: Mapping[str, object], key: str, /) -> str:
    value = config[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be str.")
    return value


def _case_variant(case: Mapping[str, object], /) -> str:
    value = case["execution_variant"]
    if not isinstance(value, str):
        raise TypeError("execution_variant must be str.")
    return value


def _case_method(case: Mapping[str, object], /) -> str:
    value = case["integration_method"]
    if not isinstance(value, str):
        raise TypeError("integration_method must be str.")
    return value


@lru_cache(maxsize=None)
def _smolyak_rule_lengths(level: int, /) -> tuple[int, ...]:
    if level < 1:
        raise ValueError("level must be positive.")
    rule_lengths: list[int] = []
    for current_level in range(1, level + 1):
        if current_level == 1:
            rule_lengths.append(1)
        else:
            rule_lengths.append((1 << (current_level - 1)) + 1)
    return tuple(rule_lengths)


@lru_cache(maxsize=None)
def _same_budget_num_points(dimension: int, level: int, /) -> int:
    from jax_util.functional.smolyak import (
        _count_evaluation_points,
        _term_generation_weights_numpy,
    )

    rule_lengths_np = np.asarray(_smolyak_rule_lengths(level), dtype=np.int64)
    generation_weights_np = np.asarray(
        _generation_weights_for_dimension(dimension),
        dtype=np.int64,
    )
    return int(_count_evaluation_points(level, rule_lengths_np, generation_weights_np))


@lru_cache(maxsize=None)
def _generation_weights_for_dimension(dimension: int, /) -> tuple[int, ...]:
    from jax_util.functional.smolyak import _term_generation_weights_numpy

    generation_weights_np = _term_generation_weights_numpy(dimension, None)
    return tuple(int(value) for value in generation_weights_np.tolist())


def _build_accuracy_coefficients(
    dimension: int,
    coeff_start: float,
    coeff_stop: float,
    num_accuracy_problems: int,
    /,
) -> NDArray[np.float64]:
    base = np.linspace(coeff_start, coeff_stop, dimension, dtype=np.float64)
    scales = np.linspace(-1.0, 1.0, num_accuracy_problems, dtype=np.float64)
    return scales[:, None] * base[None, :]


def _build_monte_carlo_integrator(
    dimension: int,
    num_samples: int,
    runtime_dtype: Any,
    seed: int,
    chunk_size: int,
    /,
) -> Any:
    import jax
    import jax.numpy as jnp

    from jax_util.functional import MonteCarloIntegrator

    def typed_uniform_cube_samples(
        key: jax.Array,
        sample_dimension: int,
        sample_count: int,
        /,
    ) -> tuple[jax.Array, jax.Array]:
        next_key, sample_key = jax.random.split(key)
        samples = jax.random.uniform(
            sample_key,
            shape=(sample_dimension, sample_count),
            minval=jnp.asarray(-0.5, dtype=runtime_dtype),
            maxval=jnp.asarray(0.5, dtype=runtime_dtype),
            dtype=runtime_dtype,
        )
        return next_key, samples

    integrator = MonteCarloIntegrator(
        dimension=dimension,
        num_samples=num_samples,
        key=jax.random.PRNGKey(seed),
        chunk_size=chunk_size,
        sampler=typed_uniform_cube_samples,
    )
    return integrator


def _analytic_box_exponential_integrals(coefficients: NDArray[np.float64], /) -> NDArray[np.float64]:
    factors = np.ones_like(coefficients, dtype=np.float64)
    nonzero_mask = np.abs(coefficients) > 1.0e-14
    factors[nonzero_mask] = (
        2.0 * np.sinh(0.5 * coefficients[nonzero_mask])
    ) / coefficients[nonzero_mask]
    return np.prod(factors, axis=-1)


def _compact_memory_stats(device: object, /) -> dict[str, int] | None:
    memory_stats = getattr(device, "memory_stats", None)
    if memory_stats is None:
        return None
    stats = memory_stats()
    if stats is None:
        return None
    keys = (
        "bytes_in_use",
        "peak_bytes_in_use",
        "bytes_reserved",
        "peak_bytes_reserved",
        "bytes_limit",
        "pool_bytes",
        "peak_pool_bytes",
    )
    compact: dict[str, int] = {}
    for key in keys:
        value = stats.get(key)
        if isinstance(value, int):
            compact[key] = int(value)
    return compact


def _array_nbytes(value: Any, /) -> int:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None or dtype is None:
        raise TypeError("value must expose shape and dtype.")
    return int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dtype).itemsize)


def _benchmark_compiled(compiled: Any, *args: Any, warm_repeats: int) -> dict[str, float]:
    first_start = time.perf_counter()
    first_value = compiled(*args)
    if hasattr(first_value, "block_until_ready"):
        first_value.block_until_ready()
    first_stop = time.perf_counter()
    first_ms = (first_stop - first_start) * 1000.0

    warm_times_ms: list[float] = []
    for _ in range(warm_repeats):
        start = time.perf_counter()
        value = compiled(*args)
        if hasattr(value, "block_until_ready"):
            value.block_until_ready()
        stop = time.perf_counter()
        warm_times_ms.append((stop - start) * 1000.0)

    warm_runtime_ms = float(np.mean(warm_times_ms)) if warm_times_ms else 0.0
    return {
        "first_call_ms": first_ms,
        "warm_runtime_ms": warm_runtime_ms,
        "compile_ms": max(0.0, first_ms - warm_runtime_ms),
    }


def _jsonl_path_for_output(output_path: Path, /) -> Path:
    return output_path.with_suffix(".jsonl")


def _failure_kind_from_exception(exc: BaseException, traceback_text: str, /) -> str:
    lower = traceback_text.lower()
    if isinstance(exc, MemoryError):
        return "host_oom"
    if "xla_gpu_host_bfc" in lower or "pinned host" in lower:
        return "host_oom"
    if "resource_exhausted" in lower or "out of memory" in lower:
        return "oom"
    return FailureKind.PYTHON_EXCEPTION.value


def _runner_metadata(context: Mapping[str, object], /) -> dict[str, object]:
    metadata = context.get("runner_metadata", {})
    if not isinstance(metadata, dict):
        return {}
    return {str(key): value for key, value in metadata.items()}


def _worker_label_from_context(context: Mapping[str, object], /) -> str:
    metadata = _runner_metadata(context)
    worker_label = metadata.get("worker_label")
    if isinstance(worker_label, str) and worker_label:
        return worker_label
    return "worker"


def _gpu_slot_from_context(context: Mapping[str, object], /) -> int | None:
    metadata = _runner_metadata(context)
    gpu_slot = metadata.get("gpu_slot")
    if gpu_slot is None:
        return None
    return int(gpu_slot)


def _cpu_affinity_from_context(context: Mapping[str, object], /) -> list[int] | None:
    metadata = _runner_metadata(context)
    cpu_affinity = metadata.get("cpu_affinity")
    if not isinstance(cpu_affinity, list):
        return None
    return [int(cpu) for cpu in cpu_affinity]


def _assigned_gpu_index_from_context(context: Mapping[str, object], /) -> int | str | None:
    metadata = _runner_metadata(context)
    gpu_ids = metadata.get("gpu_ids")
    if isinstance(gpu_ids, list) and gpu_ids:
        return int(gpu_ids[0])
    env_vars = context.get("environment_variables", {})
    if isinstance(env_vars, dict):
        gpu_id = env_vars.get("gpu_id")
        if isinstance(gpu_id, str) and gpu_id:
            return int(gpu_id)
        if env_vars.get("JAX_PLATFORMS") == "cpu":
            return "cpu"
    return None


def _attach_runner_metadata(
    result: dict[str, object],
    context: Mapping[str, object],
    /,
) -> dict[str, object]:
    result.setdefault("worker_label", _worker_label_from_context(context))
    result.setdefault("assigned_gpu_index", _assigned_gpu_index_from_context(context))
    result["gpu_slot"] = _gpu_slot_from_context(context)
    result["cpu_affinity"] = _cpu_affinity_from_context(context)
    return result


def _base_environment_variables(run_config: Mapping[str, object], /) -> dict[str, str]:
    env_vars = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    platform = _config_str(run_config, "platform")
    if platform == "cpu":
        env_vars.update(build_cpu_env())
        return env_vars

    env_vars.update(
        build_env_for_profile(
            "gpu",
            jax_platform_name="cuda",
            disable_preallocation=False,
            memory_fraction=(
                float(run_config["xla_memory_fraction"])
                if run_config.get("xla_memory_fraction") is not None
                else None
            ),
            allocator=(
                str(run_config["xla_allocator"])
                if run_config.get("xla_allocator") is not None
                else None
            ),
            tf_gpu_allocator=(
                str(run_config["xla_tf_gpu_allocator"])
                if run_config.get("xla_tf_gpu_allocator") is not None
                else None
            ),
            use_cuda_host_allocator=(
                bool(run_config["xla_use_cuda_host_allocator"])
                if run_config.get("xla_use_cuda_host_allocator") is not None
                else None
            ),
            xla_memory_scheduler=(
                str(run_config["xla_memory_scheduler"])
                if run_config.get("xla_memory_scheduler") is not None
                else None
            ),
            xla_gpu_enable_while_loop_double_buffering=(
                bool(run_config["xla_gpu_enable_while_loop_double_buffering"])
                if run_config.get("xla_gpu_enable_while_loop_double_buffering") is not None
                else None
            ),
            xla_latency_hiding_scheduler_rerun=(
                int(run_config["xla_latency_hiding_scheduler_rerun"])
                if run_config.get("xla_latency_hiding_scheduler_rerun") is not None
                else None
            ),
            jax_compiler_enable_remat_pass=(
                bool(run_config["jax_compiler_enable_remat_pass"])
                if run_config.get("jax_compiler_enable_remat_pass") is not None
                else None
            ),
        )
    )
    return env_vars


def _resource_capacity_for_run(run_config: Mapping[str, object], /) -> FullResourceCapacity:
    platform = _config_str(run_config, "platform")
    if platform == "cpu":
        return FullResourceCapacity.from_system(max_workers=1, gpu_devices=())

    gpu_indices_value = run_config.get("gpu_indices")
    gpu_indices = (
        [int(gpu_index) for gpu_index in gpu_indices_value]
        if isinstance(gpu_indices_value, list)
        else []
    )
    if not gpu_indices:
        raise RuntimeError("gpu_indices must be resolved before building GPU capacity.")

    workers_per_gpu = _config_int(run_config, "workers_per_gpu")
    visible_devices = ",".join(str(gpu_index) for gpu_index in gpu_indices)
    gpu_devices = detect_gpu_devices(
        environ={"CUDA_VISIBLE_DEVICES": visible_devices},
        max_slots=workers_per_gpu,
    )
    return FullResourceCapacity.from_system(
        max_workers=len(gpu_devices) * workers_per_gpu,
        gpu_devices=gpu_devices,
    )


def _resource_estimate_for_case(
    case: Mapping[str, object],
    run_config: Mapping[str, object],
) -> FullResourceEstimate:
    del case
    if _config_str(run_config, "platform") == "gpu":
        return FullResourceEstimate(gpu_count=1, gpu_slots=1)
    return FullResourceEstimate()


def _context_builder(
    case: Mapping[str, object],
    run_config: Mapping[str, object],
    jsonl_output_path: Path,
) -> dict[str, object]:
    del case
    return {
        "run_config": dict(run_config),
        "jsonl_output_path": str(jsonl_output_path),
        "environment_variables": _base_environment_variables(run_config),
    }


def _build_failure_result_record(
    case: Mapping[str, object],
    context: Mapping[str, object],
    *,
    failure_kind: str | None,
    error: str,
    traceback_text: str,
    failure_source: str,
    runner_failure_kind: str | None,
    failure_phase: str | None = None,
    device_memory_stats: dict[str, int] | None = None,
    extra_fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "status": "failed",
        "failure_kind": failure_kind,
        "runner_failure_kind": runner_failure_kind,
        "failure_source": failure_source,
        "failure_phase": failure_phase,
        "case_id": _case_int(case, "case_id"),
        "dimension": _case_int(case, "dimension"),
        "level": _case_int(case, "level"),
        "dtype_name": str(case["dtype_name"]),
        "integration_method": _case_method(case),
        "integration_method_index": int(case["integration_method_index"]),
        "execution_variant": _case_variant(case),
        "execution_variant_index": int(case["execution_variant_index"]),
        "mode": str(case["mode"]),
        "budget_kind": str(case["budget_kind"]),
        "budget_value": _case_int(case, "budget_value"),
        "budget_reference": str(case["budget_reference"]),
        "same_budget_num_points": _case_int(case, "same_budget_num_points"),
        "error": error,
        "traceback": traceback_text[-4000:],
    }
    if device_memory_stats is not None:
        result["device_memory_stats"] = device_memory_stats
    if extra_fields is not None:
        result.update(extra_fields)
    return _attach_runner_metadata(result, context)


def _parent_completion_record(
    case: Mapping[str, object],
    context: Mapping[str, object],
    result: ExecutionResult,
    /,
) -> dict[str, object]:
    status = "skipped" if result.status == "skipped" else "failed"
    record: dict[str, object] = {
        "status": status,
        "failure_kind": result.failure_kind,
        "runner_failure_kind": result.failure_kind,
        "failure_source": result.source,
        "failure_phase": None,
        "case_id": _case_int(case, "case_id"),
        "dimension": _case_int(case, "dimension"),
        "level": _case_int(case, "level"),
        "dtype_name": str(case["dtype_name"]),
        "integration_method": _case_method(case),
        "integration_method_index": int(case["integration_method_index"]),
        "execution_variant": _case_variant(case),
        "execution_variant_index": int(case["execution_variant_index"]),
        "mode": str(case["mode"]),
        "budget_kind": str(case["budget_kind"]),
        "budget_value": _case_int(case, "budget_value"),
        "budget_reference": str(case["budget_reference"]),
        "same_budget_num_points": _case_int(case, "same_budget_num_points"),
        "error": result.message,
        "traceback": (result.traceback or "")[-4000:],
        "raw_exit_code": result.raw_exit_code,
        "signal_name": result.signal_name,
        "result_source": result.source,
    }
    return _attach_runner_metadata(record, context)


def _case_label(case: Mapping[str, object], /) -> str:
    return (
        f"case={_case_int(case, 'case_id')} "
        f"mode={case['mode']} "
        f"d={_case_int(case, 'dimension')} "
        f"l={_case_int(case, 'level')} "
        f"dtype={case['dtype_name']}"
    )


def _run_single_case(case: Mapping[str, object], run_config: Mapping[str, object], /) -> dict[str, object]:
    phase = "jax_import"
    try:
        import equinox as eqx
        import jax
        import jax.numpy as jnp

        from jax_util.functional import integrate, initialize_smolyak_integrator

        class ExponentialIntegrand(eqx.Module):
            coeffs: jax.Array

            def __call__(self, x: jax.Array) -> jax.Array:
                return jnp.asarray([jnp.exp(jnp.dot(self.coeffs, x))], dtype=self.coeffs.dtype)

        platform = _config_str(run_config, "platform")
        if platform == "gpu":
            target_device = jax.devices("gpu")[0]
            init_device = target_device
        else:
            cpu_device = jax.devices("cpu")[0]
            target_device = cpu_device
            init_device = cpu_device
        runtime_dtype = getattr(jnp, str(case["dtype_name"]))
        integration_method = _case_method(case)
        execution_variant = _case_variant(case)
        accuracy_coefficients = _build_accuracy_coefficients(
            _case_int(case, "dimension"),
            _config_float(run_config, "coeff_start"),
            _config_float(run_config, "coeff_stop"),
            _config_int(run_config, "num_accuracy_problems"),
        )

        memory_checkpoints: dict[str, dict[str, int] | None] = {}

        phase = "integrator_init"
        t0 = time.perf_counter()
        if integration_method == "smolyak":
            with jax.default_device(init_device):
                integrator = initialize_smolyak_integrator(
                    dimension=_case_int(case, "dimension"),
                    level=_case_int(case, "level"),
                    dtype=runtime_dtype,
                    chunk_size=_config_int(run_config, "chunk_size"),
                )
            jax.block_until_ready(integrator.rule_nodes)
            jax.block_until_ready(integrator.rule_weights)
            jax.block_until_ready(integrator.rule_offsets)
            jax.block_until_ready(integrator.rule_lengths)
            jax.block_until_ready(integrator.generation_weights)
            storage_bytes = int(integrator.storage_bytes)
            rule_nodes_dtype = str(integrator.rule_nodes.dtype)
            rule_weights_dtype = str(integrator.rule_weights.dtype)
            num_terms = int(integrator.num_terms)
            num_points = int(integrator.num_evaluation_points)
            num_evaluation_points = int(integrator.num_evaluation_points)
            num_samples = None
            chunk_size = int(integrator.chunk_size)
            sampling_seconds: float | None = None
        elif integration_method == "monte_carlo":
            with jax.default_device(init_device):
                integrator = _build_monte_carlo_integrator(
                    _case_int(case, "dimension"),
                    _case_int(case, "same_budget_num_points"),
                    runtime_dtype,
                    _case_int(case, "case_id"),
                    _config_int(run_config, "chunk_size"),
                )
            jax.block_until_ready(integrator.key)
            storage_bytes = _array_nbytes(integrator.key)
            rule_nodes_dtype = None
            rule_weights_dtype = None
            num_terms = None
            num_points = int(integrator.num_samples)
            num_evaluation_points = int(integrator.num_samples)
            num_samples = int(integrator.num_samples)
            chunk_size = int(integrator.chunk_size)
            sampling_seconds = 0.0
        else:
            raise ValueError(f"Unsupported integration method: {integration_method}")
        t1 = time.perf_counter()
        if integration_method == "monte_carlo":
            sampling_seconds = t1 - t0
        memory_checkpoints["after_init"] = _compact_memory_stats(target_device)

        phase = "device_transfer"
        if platform != "gpu":
            integrator = jax.device_put(integrator, target_device)
        coeffs = jax.device_put(jnp.asarray(accuracy_coefficients[-1], dtype=runtime_dtype), target_device)
        accuracy_coeffs = jax.device_put(jnp.asarray(accuracy_coefficients, dtype=runtime_dtype), target_device)
        if integration_method == "smolyak":
            jax.block_until_ready(integrator.rule_nodes)
            jax.block_until_ready(integrator.rule_weights)
            jax.block_until_ready(integrator.rule_offsets)
            jax.block_until_ready(integrator.rule_lengths)
            jax.block_until_ready(integrator.generation_weights)
        else:
            jax.block_until_ready(integrator.key)
        jax.block_until_ready(coeffs)
        jax.block_until_ready(accuracy_coeffs)
        t2 = time.perf_counter()
        memory_checkpoints["after_transfer"] = _compact_memory_stats(target_device)

        def single_integral(current_integrator: Any, current_coeffs: jax.Array) -> jax.Array:
            return integrate(ExponentialIntegrand(current_coeffs), current_integrator)[0]

        def batched_accuracy_integrals(current_integrator: Any, coeff_matrix: jax.Array) -> jax.Array:
            batched_integrands = ExponentialIntegrand(coeff_matrix)
            return jax.vmap(
                lambda current_f: integrate(current_f, current_integrator)[0],
                in_axes=0,
                out_axes=0,
            )(batched_integrands)

        @eqx.filter_jit
        def compiled_single(current_integrator: Any, current_coeffs: jax.Array) -> jax.Array:
            return single_integral(current_integrator, current_coeffs)

        @eqx.filter_jit
        def compiled_batch(current_integrator: Any, coeff_matrix: jax.Array) -> jax.Array:
            return batched_accuracy_integrals(current_integrator, coeff_matrix)

        expected_values = _analytic_box_exponential_integrals(accuracy_coefficients)
        dense_integrand_matrix_upper_bound_bytes = (
            int(num_evaluation_points)
            * _config_int(run_config, "num_accuracy_problems")
            * int(np.dtype(runtime_dtype).itemsize)
        )

        phase = "benchmark"
        benchmark_start = time.perf_counter()
        if execution_variant == "single":
            benchmark_stats = _benchmark_compiled(
                compiled_single,
                integrator,
                coeffs,
                warm_repeats=_config_int(run_config, "num_repeats"),
            )
            benchmark_end = time.perf_counter()
            memory_checkpoints["after_benchmark"] = _compact_memory_stats(target_device)

            phase = "execute"
            execute_start = time.perf_counter()
            measured_value = compiled_single(integrator, coeffs)
            measured_value = jax.block_until_ready(measured_value)
            execute_end = time.perf_counter()
            memory_checkpoints["after_execute"] = _compact_memory_stats(target_device)

            phase = "host_copy"
            host_copy_start = time.perf_counter()
            actual_scalar = float(np.asarray(measured_value))
            host_copy_end = time.perf_counter()
            memory_checkpoints["after_host_copy"] = _compact_memory_stats(target_device)

            expected_scalar = float(expected_values[-1])
            abs_error_scalar = abs(actual_scalar - expected_scalar)
            measurement_problem_count = 1
            measured_device_nbytes = _array_nbytes(measured_value)
            mean_abs_err = abs_error_scalar
            var_abs_err = 0.0
            max_abs_err = abs_error_scalar
            actual_value_field = actual_scalar
            expected_value_field = expected_scalar
        elif execution_variant == "vmap":
            benchmark_stats = _benchmark_compiled(
                compiled_batch,
                integrator,
                accuracy_coeffs,
                warm_repeats=_config_int(run_config, "num_repeats"),
            )
            benchmark_end = time.perf_counter()
            memory_checkpoints["after_benchmark"] = _compact_memory_stats(target_device)

            phase = "execute"
            execute_start = time.perf_counter()
            measured_values = compiled_batch(integrator, accuracy_coeffs)
            measured_values = jax.block_until_ready(measured_values)
            execute_end = time.perf_counter()
            memory_checkpoints["after_execute"] = _compact_memory_stats(target_device)

            phase = "host_copy"
            host_copy_start = time.perf_counter()
            actual_values = np.asarray(measured_values, dtype=np.float64)
            host_copy_end = time.perf_counter()
            memory_checkpoints["after_host_copy"] = _compact_memory_stats(target_device)

            abs_errors = np.abs(actual_values - expected_values)
            measurement_problem_count = _config_int(run_config, "num_accuracy_problems")
            measured_device_nbytes = _array_nbytes(measured_values)
            mean_abs_err = float(np.mean(abs_errors))
            var_abs_err = float(np.var(abs_errors))
            max_abs_err = float(np.max(abs_errors))
            actual_value_field = float(actual_values[-1])
            expected_value_field = float(expected_values[-1])
        else:
            raise ValueError(f"Unsupported execution variant: {execution_variant}")

        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        benchmark_seconds = benchmark_end - benchmark_start
        execute_seconds = execute_end - execute_start
        host_copy_seconds = host_copy_end - host_copy_start

        return {
            "status": "ok",
            "case_id": _case_int(case, "case_id"),
            "dimension": _case_int(case, "dimension"),
            "level": _case_int(case, "level"),
            "dtype_name": str(case["dtype_name"]),
            "integration_method": integration_method,
            "integration_method_index": int(case["integration_method_index"]),
            "execution_variant": execution_variant,
            "execution_variant_index": int(case["execution_variant_index"]),
            "mode": str(case["mode"]),
            "budget_kind": str(case["budget_kind"]),
            "budget_value": _case_int(case, "budget_value"),
            "budget_reference": str(case["budget_reference"]),
            "same_budget_num_points": _case_int(case, "same_budget_num_points"),
            "backend": jax.default_backend(),
            "device_kind": target_device.device_kind,
            "visible_device_id": int(target_device.id),
            "num_terms": num_terms,
            "num_points": num_points,
            "num_evaluation_points": num_evaluation_points,
            "storage_bytes": storage_bytes,
            "rule_nodes_dtype": rule_nodes_dtype,
            "rule_weights_dtype": rule_weights_dtype,
            "chunk_size": chunk_size,
            "num_samples": num_samples,
            "expected": expected_value_field,
            "actual": actual_value_field,
            "num_accuracy_problems": _config_int(run_config, "num_accuracy_problems"),
            "measurement_problem_count": measurement_problem_count,
            "problem_batch_size": measurement_problem_count,
            "coeff_inputs_device_nbytes": _array_nbytes(
                accuracy_coeffs if execution_variant == "vmap" else coeffs
            ),
            "measured_values_device_nbytes": measured_device_nbytes,
            "dense_integrand_matrix_upper_bound_bytes": dense_integrand_matrix_upper_bound_bytes,
            "mean_abs_err": mean_abs_err,
            "var_abs_err": var_abs_err,
            "max_abs_err": max_abs_err,
            "num_repeats": _config_int(run_config, "num_repeats"),
            "vmap_batch_size": measurement_problem_count if execution_variant == "vmap" else None,
            "first_call_ms": benchmark_stats["first_call_ms"],
            "compile_ms": benchmark_stats["compile_ms"],
            "warm_runtime_ms": benchmark_stats["warm_runtime_ms"],
            "throughput_integrals_per_second": (
                measurement_problem_count * 1000.0 / benchmark_stats["warm_runtime_ms"]
                if benchmark_stats["warm_runtime_ms"] > 0.0
                else None
            ),
            "integrator_init_seconds": t1 - t0,
            "sampling_seconds": sampling_seconds,
            "device_transfer_seconds": t2 - t1,
            "lowering_seconds": benchmark_stats["compile_ms"] / 1000.0,
            "first_execute_seconds": benchmark_stats["first_call_ms"] / 1000.0,
            "warm_execute_seconds": benchmark_stats["warm_runtime_ms"] / 1000.0,
            "integration_seconds": benchmark_stats["warm_runtime_ms"] / 1000.0,
            "timing_probe_seconds": benchmark_seconds,
            "warmup_seconds": benchmark_stats["first_call_ms"] / 1000.0,
            "measured_runtime_seconds": benchmark_stats["warm_runtime_ms"] / 1000.0,
            "execute_seconds": execute_seconds,
            "host_copy_seconds": host_copy_seconds,
            "avg_integral_seconds": (
                benchmark_stats["warm_runtime_ms"] / 1000.0 / measurement_problem_count
            ),
            "process_rss_mb": rss_mb,
            "device_memory_stats": _compact_memory_stats(target_device),
            "memory_checkpoints": memory_checkpoints,
            "failure_phase": None,
        }
    except Exception as exc:
        raise _CasePhaseError(phase, exc) from exc


def _run_case_task(case: Mapping[str, object], context: TaskContext) -> None:
    run_config = cast(Mapping[str, object], context["run_config"])
    jsonl_output_path = Path(str(context["jsonl_output_path"]))
    try:
        result = _run_single_case(case, run_config)
    except Exception as exc:
        original_exc = exc.__cause__ if isinstance(exc, _CasePhaseError) and exc.__cause__ is not None else exc
        failure_phase = exc.phase if isinstance(exc, _CasePhaseError) else None
        traceback_text = traceback.format_exc(limit=8)
        message = "".join(traceback.format_exception_only(type(original_exc), original_exc)).strip()
        failure_kind = _failure_kind_from_exception(original_exc, traceback_text)
        device_memory_stats: dict[str, int] | None = None
        try:
            import jax
            platform = _config_str(run_config, "platform")
            target_device = jax.devices("gpu")[0] if platform == "gpu" else jax.devices("cpu")[0]
            device_memory_stats = _compact_memory_stats(target_device)
        except Exception:
            device_memory_stats = None
        append_jsonl_record(
            jsonl_output_path,
            _build_failure_result_record(
                case,
                context,
                failure_kind=failure_kind,
                runner_failure_kind=FailureKind.PYTHON_EXCEPTION.value,
                failure_source="child",
                failure_phase=failure_phase,
                error=message,
                traceback_text=traceback_text,
                device_memory_stats=device_memory_stats,
            ),
        )
        raise
    append_jsonl_record(jsonl_output_path, _attach_runner_metadata(result, context))


def _log_case_started(
    case: Mapping[str, object],
    context: Mapping[str, object],
    pid: int,
    /,
) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    print(
        f"[{started_at}] start {_worker_label_from_context(context)} {_case_label(case)} pid={pid}",
        flush=True,
    )


def _log_case_finished(
    case: Mapping[str, object],
    context: Mapping[str, object],
    result: ExecutionResult,
    pid: int | None,
    /,
) -> None:
    finished_at = datetime.now(timezone.utc).isoformat()
    failure_kind = result.failure_kind if result.failure_kind is not None else "-"
    print(
        f"[{finished_at}] done  {_worker_label_from_context(context)} {_case_label(case)} "
        f"pid={pid} status={result.status} failure_kind={failure_kind}",
        flush=True,
    )


def _handle_case_finished(
    jsonl_output_path: Path,
    case: Mapping[str, object],
    context: Mapping[str, object],
    result: ExecutionResult,
    pid: int | None,
    /,
) -> None:
    if result.source == "parent" or result.status == "skipped":
        append_jsonl_record(
            jsonl_output_path,
            _parent_completion_record(case, context, result),
        )
    _log_case_finished(case, context, result, pid)


def _build_runtime_monitor(run_config: Mapping[str, object], /) -> RuntimeMonitor | None:
    port = run_config.get("monitor_port")
    if not isinstance(port, int) or port <= 0:
        return None
    return RuntimeMonitor.for_run(
        bind_host=_config_str(run_config, "monitor_bind_host"),
        port=port,
        sample_interval_seconds=_config_float(run_config, "monitor_sample_interval_seconds"),
        enable_http=bool(run_config.get("monitor_enable_http", True)),
    )


def _run_cases_with_runner(
    cases: list[dict[str, object]],
    run_config: Mapping[str, object],
    jsonl_output_path: Path,
    /,
) -> list[dict[str, object]]:
    worker = StandardWorker(
        _run_case_task,
        resource_estimator=cast(
            Any,
            partial(_resource_estimate_for_case, run_config=run_config),
        ),
    )
    skip_controller = _TimeoutFrontierSkipController()
    scheduler = StandardFullResourceScheduler.from_worker(
        cases=cases,
        worker=worker,
        context_builder=partial(
            _context_builder,
            run_config=run_config,
            jsonl_output_path=jsonl_output_path,
        ),
        skip_controller=skip_controller,
        disable_gpu_preallocation=GPU_PREALLOCATION_DISABLED,
        resource_capacity=_resource_capacity_for_run(run_config),
    )
    monitor = _build_runtime_monitor(run_config)
    if monitor is not None:
        monitor.start()
        if bool(run_config.get("monitor_enable_http", True)):
            print(
                f"[monitor] http://{_config_str(run_config, 'monitor_bind_host')}:{_config_int(run_config, 'monitor_port')}",
                flush=True,
            )
    try:
        runner = StandardRunner(
            scheduler,
            case_timeout_seconds=float(_config_int(run_config, "timeout_seconds")),
            monitor=monitor,
            on_case_started=_log_case_started,
            on_case_finished=partial(_handle_case_finished, jsonl_output_path),
        )
        runner.run(worker)
    finally:
        if monitor is not None:
            monitor.stop()
    results = read_jsonl_records(jsonl_output_path)
    results.sort(key=lambda result: _result_int(result, "case_id"))
    return results


def _summary_by_dtype(
    results: list[dict[str, object]],
    dtype_names: list[str],
    /,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for integration_method in SUPPORTED_INTEGRATION_METHODS:
        for execution_variant in SUPPORTED_EXECUTION_VARIANTS:
            for dtype_name in dtype_names:
                filtered_results = [
                    result
                    for result in results
                    if result.get("dtype_name") == dtype_name
                    and result.get("integration_method") == integration_method
                    and result.get("execution_variant") == execution_variant
                ]
                ok_results = [result for result in filtered_results if result.get("status") == "ok"]
                failed_results = [result for result in filtered_results if result.get("status") == "failed"]
                skipped_results = [result for result in filtered_results if result.get("status") == "skipped"]

                mean_abs_errs = [
                    _result_float(result, "mean_abs_err")
                    for result in ok_results
                    if isinstance(result.get("mean_abs_err"), (int, float))
                ]
                avg_times = [
                    _result_float(result, "avg_integral_seconds")
                    for result in ok_results
                    if isinstance(result.get("avg_integral_seconds"), (int, float))
                ]
                summaries.append({
                    "integration_method": integration_method,
                    "execution_variant": execution_variant,
                    "dtype_name": dtype_name,
                    "num_cases": len(filtered_results),
                    "num_success": len(ok_results),
                    "num_failed": len(failed_results),
                    "num_skipped": len(skipped_results),
                    "max_mean_abs_err": max(mean_abs_errs) if mean_abs_errs else None,
                    "mean_mean_abs_err": float(np.mean(mean_abs_errs)) if mean_abs_errs else None,
                    "mean_avg_integral_seconds": float(np.mean(avg_times)) if avg_times else None,
                    "min_avg_integral_seconds": min(avg_times) if avg_times else None,
                    "max_avg_integral_seconds": max(avg_times) if avg_times else None,
                })
    return summaries


def _frontier_by_dtype_and_level(
    results: list[dict[str, object]],
    dtype_names: list[str],
    levels: list[int],
    /,
) -> list[dict[str, object]]:
    frontier: list[dict[str, object]] = []
    for integration_method in SUPPORTED_INTEGRATION_METHODS:
        for execution_variant in SUPPORTED_EXECUTION_VARIANTS:
            for dtype_name in dtype_names:
                dtype_results = [
                    result
                    for result in results
                    if result.get("dtype_name") == dtype_name
                    and result.get("integration_method") == integration_method
                    and result.get("execution_variant") == execution_variant
                ]
                for level in levels:
                    level_results = [
                        result for result in dtype_results if result.get("level") == level
                    ]
                    success_dimensions = [
                        _result_int(result, "dimension")
                        for result in level_results
                        if result.get("status") == "ok"
                    ]
                    failed_dimensions = [
                        _result_int(result, "dimension")
                        for result in level_results
                        if result.get("status") == "failed"
                    ]
                    skipped_dimensions = [
                        _result_int(result, "dimension")
                        for result in level_results
                        if result.get("status") == "skipped"
                    ]
                    frontier.append({
                        "integration_method": integration_method,
                        "execution_variant": execution_variant,
                        "dtype_name": dtype_name,
                        "level": level,
                        "max_success_dimension": max(success_dimensions) if success_dimensions else None,
                        "min_failure_dimension": min(failed_dimensions) if failed_dimensions else None,
                        "min_skipped_dimension": min(skipped_dimensions) if skipped_dimensions else None,
                        "num_success": len(success_dimensions),
                        "num_failed": len(failed_dimensions),
                        "num_skipped": len(skipped_dimensions),
                    })
    return frontier


def _frontier_by_dtype_and_dimension(
    results: list[dict[str, object]],
    dtype_names: list[str],
    dimensions: list[int],
    /,
) -> list[dict[str, object]]:
    frontier: list[dict[str, object]] = []
    for integration_method in SUPPORTED_INTEGRATION_METHODS:
        for execution_variant in SUPPORTED_EXECUTION_VARIANTS:
            for dtype_name in dtype_names:
                dtype_results = [
                    result
                    for result in results
                    if result.get("dtype_name") == dtype_name
                    and result.get("integration_method") == integration_method
                    and result.get("execution_variant") == execution_variant
                ]
                for dimension in dimensions:
                    dimension_results = [
                        result
                        for result in dtype_results
                        if result.get("dimension") == dimension
                    ]
                    success_levels = [
                        _result_int(result, "level")
                        for result in dimension_results
                        if result.get("status") == "ok"
                    ]
                    failed_levels = [
                        _result_int(result, "level")
                        for result in dimension_results
                        if result.get("status") == "failed"
                    ]
                    skipped_levels = [
                        _result_int(result, "level")
                        for result in dimension_results
                        if result.get("status") == "skipped"
                    ]
                    frontier.append({
                        "integration_method": integration_method,
                        "execution_variant": execution_variant,
                        "dtype_name": dtype_name,
                        "dimension": dimension,
                        "max_success_level": max(success_levels) if success_levels else None,
                        "min_failure_level": min(failed_levels) if failed_levels else None,
                        "min_skipped_level": min(skipped_levels) if skipped_levels else None,
                        "num_success": len(success_levels),
                        "num_failed": len(failed_levels),
                        "num_skipped": len(skipped_levels),
                    })
    return frontier


def run_benchmark(
    dimensions: list[int],
    levels: list[int],
    dtype_names: list[str],
    integration_methods: list[str],
    execution_variants: list[str],
    /,
    *,
    output_path: Path,
    platform: str,
    gpu_indices: list[int],
    workers_per_gpu: int,
    timeout_seconds: int,
    num_repeats: int,
    num_accuracy_problems: int,
    coeff_start: float,
    coeff_stop: float,
    chunk_size: int,
    xla_memory_fraction: float | None,
    xla_allocator: str | None,
    xla_tf_gpu_allocator: str | None,
    xla_use_cuda_host_allocator: bool | None,
    xla_memory_scheduler: str | None,
    xla_gpu_enable_while_loop_double_buffering: bool | None,
    xla_latency_hiding_scheduler_rerun: int | None,
    jax_compiler_enable_remat_pass: bool | None,
    monitor_port: int,
    monitor_bind_host: str,
    monitor_sample_interval_seconds: float,
    monitor_enable_http: bool,
) -> dict[str, object]:
    started_at = datetime.now(timezone.utc)
    jsonl_output_path = _jsonl_path_for_output(output_path)
    jsonl_output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_output_path.write_text("", encoding="utf-8")

    run_config: dict[str, object] = {
        "platform": platform,
        "gpu_indices": gpu_indices,
        "workers_per_gpu": workers_per_gpu,
        "timeout_seconds": timeout_seconds,
        "num_repeats": num_repeats,
        "num_accuracy_problems": num_accuracy_problems,
        "coeff_start": coeff_start,
        "coeff_stop": coeff_stop,
        "chunk_size": chunk_size,
        "xla_memory_fraction": xla_memory_fraction,
        "xla_allocator": xla_allocator,
        "xla_tf_gpu_allocator": xla_tf_gpu_allocator,
        "xla_use_cuda_host_allocator": xla_use_cuda_host_allocator,
        "xla_memory_scheduler": xla_memory_scheduler,
        "xla_gpu_enable_while_loop_double_buffering": xla_gpu_enable_while_loop_double_buffering,
        "xla_latency_hiding_scheduler_rerun": xla_latency_hiding_scheduler_rerun,
        "jax_compiler_enable_remat_pass": jax_compiler_enable_remat_pass,
        "monitor_port": monitor_port,
        "monitor_bind_host": monitor_bind_host,
        "monitor_sample_interval_seconds": monitor_sample_interval_seconds,
        "monitor_enable_http": monitor_enable_http,
    }
    metadata = _experiment_metadata()
    cases = _build_cases(
        dimensions,
        levels,
        dtype_names,
        integration_methods,
        execution_variants,
    )
    results = _run_cases_with_runner(cases, run_config, jsonl_output_path)
    finished_at = datetime.now(timezone.utc)
    num_success = sum(1 for result in results if result.get("status") == "ok")
    num_failed = sum(1 for result in results if result.get("status") == "failed")
    num_skipped = sum(1 for result in results if result.get("status") == "skipped")

    return {
        "experiment": "smolyak_scaling_benchmark",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "run_wall_seconds": (finished_at - started_at).total_seconds(),
        "platform": platform,
        "gpu_indices": gpu_indices if platform == "gpu" else [],
        "workers_per_gpu": workers_per_gpu if platform == "gpu" else 1,
        "dimensions": dimensions,
        "levels": levels,
        "dtype_names": dtype_names,
        "integration_methods": integration_methods,
        "execution_variants": execution_variants,
        "num_cases": len(cases),
        "num_success": num_success,
        "num_failed": num_failed,
        "num_skipped": num_skipped,
        "num_repeats": num_repeats,
        "num_accuracy_problems": num_accuracy_problems,
        "timeout_seconds": timeout_seconds,
        "coeff_start": coeff_start,
        "coeff_stop": coeff_stop,
        "chunk_size": chunk_size,
        "xla_config": _xla_config_from_run_config(run_config),
        "jsonl_output_path": str(jsonl_output_path),
        **metadata,
        "cases": results,
        "summary_by_dtype": _summary_by_dtype(results, dtype_names),
        "frontier_by_dtype_and_level": _frontier_by_dtype_and_level(results, dtype_names, levels),
        "frontier_by_dtype_and_dimension": _frontier_by_dtype_and_dimension(results, dtype_names, dimensions),
    }


def save_results(results: dict[str, object], output_path: Path, /) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(json_compatible(results), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Smolyak and Monte Carlo scaling on dimension/level ranges."
    )
    parser.add_argument("--dimensions", required=True, help="Inclusive integer range start:end[:step] for dimensions.")
    parser.add_argument("--levels", required=True, help="Inclusive integer range start:end[:step] for levels.")
    parser.add_argument("--platform", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--gpu-indices", default=None, help="Comma-separated physical GPU indices. Defaults to all visible GPUs.")
    parser.add_argument("--workers-per-gpu", type=int, default=DEFAULT_WORKERS_PER_GPU)
    parser.add_argument("--dtypes", default="all", help="Comma-separated dtype names or 'all'.")
    parser.add_argument("--integration-methods", default="all", help="Comma-separated integration methods or 'all'.")
    parser.add_argument("--execution-variants", default="all", help="Comma-separated execution variants or 'all'.")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--num-repeats", type=int, default=DEFAULT_NUM_REPEATS)
    parser.add_argument("--num-accuracy-problems", type=int, default=DEFAULT_NUM_ACCURACY_PROBLEMS)
    parser.add_argument("--coeff-start", type=float, default=DEFAULT_COEFF_START)
    parser.add_argument("--coeff-stop", type=float, default=DEFAULT_COEFF_STOP)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--xla-memory-fraction", type=float, default=None)
    parser.add_argument("--xla-allocator", default=None)
    parser.add_argument("--xla-tf-gpu-allocator", default=None)
    parser.add_argument("--xla-use-cuda-host-allocator", default=None)
    parser.add_argument("--xla-memory-scheduler", default=None)
    parser.add_argument("--xla-gpu-enable-while-loop-double-buffering", default=None)
    parser.add_argument("--xla-latency-hiding-scheduler-rerun", type=int, default=None)
    parser.add_argument("--jax-compiler-enable-remat-pass", default=None)
    parser.add_argument("--monitor-port", type=int, default=0)
    parser.add_argument("--monitor-bind-host", default="127.0.0.1")
    parser.add_argument("--monitor-sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--monitor-enable-http", default="true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.num_accuracy_problems < 1:
        raise ValueError("--num-accuracy-problems must be positive.")
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive.")

    dimensions = _parse_integer_range(args.dimensions)
    levels = _parse_integer_range(args.levels)
    dtype_names = _parse_dtype_names(args.dtypes)
    integration_methods = _parse_integration_methods(args.integration_methods)
    execution_variants = _parse_execution_variants(args.execution_variants)

    if args.platform == "gpu":
        gpu_indices = (
            _parse_gpu_indices(args.gpu_indices)
            if args.gpu_indices is not None
            else _discover_gpu_indices()
        )
        if not gpu_indices:
            raise RuntimeError("No GPUs were discovered for gpu platform.")
    else:
        gpu_indices = []

    xla_use_cuda_host_allocator = _parse_optional_bool_flag(args.xla_use_cuda_host_allocator)
    xla_gpu_enable_while_loop_double_buffering = _parse_optional_bool_flag(
        args.xla_gpu_enable_while_loop_double_buffering
    )
    jax_compiler_enable_remat_pass = _parse_optional_bool_flag(
        args.jax_compiler_enable_remat_pass
    )
    monitor_enable_http = _parse_optional_bool_flag(args.monitor_enable_http)
    if monitor_enable_http is None:
        monitor_enable_http = True

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or (
        RESULTS_DIR / f"smolyak_scaling_{args.platform}_{timestamp}.json"
    )
    results = run_benchmark(
        dimensions,
        levels,
        dtype_names,
        integration_methods,
        execution_variants,
        output_path=output_path,
        platform=args.platform,
        gpu_indices=gpu_indices,
        workers_per_gpu=args.workers_per_gpu,
        timeout_seconds=args.timeout_seconds,
        num_repeats=args.num_repeats,
        num_accuracy_problems=args.num_accuracy_problems,
        coeff_start=args.coeff_start,
        coeff_stop=args.coeff_stop,
        chunk_size=args.chunk_size,
        xla_memory_fraction=args.xla_memory_fraction,
        xla_allocator=args.xla_allocator,
        xla_tf_gpu_allocator=args.xla_tf_gpu_allocator,
        xla_use_cuda_host_allocator=xla_use_cuda_host_allocator,
        xla_memory_scheduler=args.xla_memory_scheduler,
        xla_gpu_enable_while_loop_double_buffering=xla_gpu_enable_while_loop_double_buffering,
        xla_latency_hiding_scheduler_rerun=args.xla_latency_hiding_scheduler_rerun,
        jax_compiler_enable_remat_pass=jax_compiler_enable_remat_pass,
        monitor_port=args.monitor_port,
        monitor_bind_host=args.monitor_bind_host,
        monitor_sample_interval_seconds=args.monitor_sample_interval_seconds,
        monitor_enable_http=monitor_enable_http,
    )

    save_results(results, output_path)
    save_results(results, RESULTS_DIR / "latest.json")
    jsonl_output_path = _jsonl_path_for_output(output_path)
    if jsonl_output_path.exists():
        shutil.copyfile(jsonl_output_path, RESULTS_DIR / "latest.jsonl")
    print(output_path)


if __name__ == "__main__":
    main()
