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
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = WORKSPACE_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from experiment_runner import (
    CHILD_COMPLETE_PREFIX,
    FailureKind,
    RuntimeMonitor,
    WorkerSlot,
    append_jsonl_record,
    apply_worker_environment,
    build_worker_slots,
    json_compatible,
    run_cases_with_subprocess_scheduler,
    worker_slot_from_mapping,
)
from jax_util.xla_env import build_env_for_profile
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_NUM_REPEATS = 5
DEFAULT_NUM_ACCURACY_PROBLEMS = 1000
DEFAULT_COEFF_START = -0.55
DEFAULT_COEFF_STOP = 0.65
DEFAULT_WORKERS_PER_GPU = 2
SUPPORTED_FLOAT_DTYPES = ("float16", "bfloat16", "float32", "float64")
SUPPORTED_INTEGRATION_METHODS = ("smolyak", "monte_carlo")
SUPPORTED_EXECUTION_VARIANTS = ("single", "vmap")
RESULTS_BRANCH_NAME = "results/functional-smolyak-scaling"
GPU_PREALLOCATION_DISABLED = True


def _parse_optional_bool_flag(value: str | None, /) -> bool | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean flag: {value}")


# 責務: `start:end[:step]` 形式の整数レンジを inclusive な整数列へ展開する。
def _parse_integer_range(spec: str, /) -> list[int]:
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("Range must be start:end or start:end:step.")

    start = int(parts[0])
    stop = int(parts[1])
    step = int(parts[2]) if len(parts) == 3 else 1

    if step == 0:
        raise ValueError("Range step must be non-zero.")
    if step < 0:
        raise ValueError("Range step must be positive.")
    if start > stop:
        raise ValueError("Range start must be less than or equal to stop.")

    return list(range(start, stop + 1, step))


# 責務: カンマ区切りの GPU index 列を整数列へ変換する。
def _parse_gpu_indices(spec: str, /) -> list[int]:
    values = [item.strip() for item in spec.split(",") if item.strip()]
    if not values:
        raise ValueError("GPU index list must not be empty.")
    return [int(value) for value in values]


# 責務: dtype 指定文字列を正規化して重複のない dtype 名列へ変換する。
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

    aliases = {
        "mc": "monte_carlo",
    }
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


# 責務: `nvidia-smi` から利用可能な GPU index を列挙する。
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

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return [int(line) for line in lines]


# 責務: Git コマンドの標準出力を取り出せるときだけ返す。
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


# 責務: 実験結果へ残す Git と worktree のメタデータを構築する。
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


# 責務: レンジ指定から解析解つきベンチマークケース列を構成する。
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
            for dtype_name in dtype_names:
                same_budget_num_points = _same_budget_num_points(dimension, level)
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
                            "same_budget_num_points": same_budget_num_points,
                        })
                        case_id += 1
    return cases


# 責務: ケース定義から整数値を安全に取り出す。
def _case_int(case: Mapping[str, object], key: str, /) -> int:
    value = case[key]
    if not isinstance(value, int):
        raise TypeError(f"{key} must be int.")
    return value


# 責務: 実行結果から整数値を安全に取り出す。
def _result_int(result: Mapping[str, object], key: str, /) -> int:
    value = result[key]
    if not isinstance(value, int):
        raise TypeError(f"{key} must be int.")
    return value


# 責務: 実行結果から浮動小数値を安全に取り出す。
def _result_float(result: Mapping[str, object], key: str, /) -> float:
    value = result[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be float.")
    return float(value)


# 責務: 実行条件辞書から整数値を安全に取り出す。
def _config_int(config: Mapping[str, object], key: str, /) -> int:
    value = config[key]
    if not isinstance(value, int):
        raise TypeError(f"{key} must be int.")
    return value


# 責務: 実行条件辞書から浮動小数値を安全に取り出す。
def _config_float(config: Mapping[str, object], key: str, /) -> float:
    value = config[key]
    if not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be float.")
    return float(value)


# 責務: 実行条件辞書から文字列値を安全に取り出す。
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
    import jax

    from jax_util.functional.smolyak import _difference_rule_storage_device

    with jax.default_device(jax.devices("cpu")[0]):
        _, _, _, rule_lengths = _difference_rule_storage_device(level)
    return tuple(int(value) for value in np.asarray(rule_lengths, dtype=np.int64).tolist())


@lru_cache(maxsize=None)
def _same_budget_num_points(dimension: int, level: int, /) -> int:
    from jax_util.functional.smolyak import _count_evaluation_points, _term_generation_weights_numpy

    rule_lengths_np = np.asarray(_smolyak_rule_lengths(level), dtype=np.int64)
    generation_weights_np = _term_generation_weights_numpy(dimension, None)
    return int(_count_evaluation_points(level, rule_lengths_np, generation_weights_np))


# 責務: 解析解比較用の係数行列を構築する。
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
    /,
) -> Any:
    import equinox as eqx
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
        sampler=typed_uniform_cube_samples,
    )
    return eqx.tree_at(lambda tree: tree.samples, integrator, jnp.asarray(integrator.samples, dtype=runtime_dtype))


# 責務: exp(a^T x) の解析積分を係数行列ぶんまとめて返す。
def _analytic_box_exponential_integrals(coefficients: NDArray[np.float64], /) -> NDArray[np.float64]:
    factors = np.ones_like(coefficients, dtype=np.float64)
    nonzero_mask = np.abs(coefficients) > 1.0e-14
    factors[nonzero_mask] = (2.0 * np.sinh(0.5 * coefficients[nonzero_mask])) / coefficients[nonzero_mask]
    return np.prod(factors, axis=-1)


# 責務: デバイスメモリ統計のうち JSON 化しやすい整数項だけを抜き出す。
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


# 責務: 出力 JSON に対応する逐次保存用 JSONL パスを返す。
def _jsonl_path_for_output(output_path: Path, /) -> Path:
    return output_path.with_suffix(".jsonl")


# 責務: 例外情報から失敗種別を粗く分類する。
def _failure_kind_from_exception(exc: BaseException, traceback_text: str, /) -> str:
    lower = traceback_text.lower()
    if isinstance(exc, MemoryError):
        return "host_oom"
    if "xla_gpu_host_bfc" in lower or "pinned host" in lower:
        return "host_oom"
    if "resource_exhausted" in lower or "out of memory" in lower:
        return "oom"
    return FailureKind.PYTHON_EXCEPTION.value


def _parent_runner_failure_kind(failure_kind: str, /) -> str:
    if failure_kind == "timeout":
        return FailureKind.TIMEOUT.value
    if failure_kind == "worker_terminated":
        return FailureKind.NO_COMPLETION.value
    return str(failure_kind)


def _build_failure_result_record(
    case: Mapping[str, object],
    worker_slot: WorkerSlot,
    *,
    failure_kind: str,
    error: str,
    traceback_text: str,
    failure_source: str,
    runner_failure_kind: str,
    device_memory_stats: dict[str, int] | None = None,
    extra_fields: Mapping[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "status": "failed",
        "failure_kind": failure_kind,
        "runner_failure_kind": runner_failure_kind,
        "failure_source": failure_source,
        "case_id": _case_int(case, "case_id"),
        "dimension": _case_int(case, "dimension"),
        "level": _case_int(case, "level"),
        "dtype_name": str(case["dtype_name"]),
        "integration_method": _case_method(case),
        "integration_method_index": int(case["integration_method_index"]),
        "execution_variant": _case_variant(case),
        "execution_variant_index": int(case["execution_variant_index"]),
        "assigned_gpu_index": worker_slot.gpu_index,
        "gpu_slot": worker_slot.gpu_slot,
        "cpu_affinity": list(worker_slot.cpu_affinity),
        "worker_label": worker_slot.worker_label,
        "error": error,
        "traceback": traceback_text[-4000:],
    }
    if device_memory_stats is not None:
        result["device_memory_stats"] = device_memory_stats
    if extra_fields is not None:
        result.update(extra_fields)
    return result


# 責務: 親側で補完した失敗結果レコードを構築する。
def _parent_failure_result(
    case: Mapping[str, object],
    worker_slot: WorkerSlot,
    failure_kind: str,
    error: str,
    traceback_text: str,
    /,
) -> dict[str, object]:
    runner_failure_kind = _parent_runner_failure_kind(failure_kind)
    return _build_failure_result_record(
        case,
        worker_slot,
        failure_kind=runner_failure_kind,
        runner_failure_kind=runner_failure_kind,
        failure_source="parent",
        error=error,
        traceback_text=traceback_text,
    )


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


# 責務: ケース識別用の短い文字列を作る。
def _case_label(case: Mapping[str, object], /) -> str:
    return (
        f"case={_case_int(case, 'case_id')} "
        f"method={case['integration_method_index']}:{case['integration_method']} "
        f"variant={case['execution_variant_index']}:{case['execution_variant']} "
        f"d={_case_int(case, 'dimension')} "
        f"l={_case_int(case, 'level')} "
        f"dtype={case['dtype_name']}"
    )


# 責務: 単一ケースのベンチマークを実行して JSON 互換結果へまとめる。
def _run_single_case(case: Mapping[str, object], run_config: Mapping[str, object], /) -> dict[str, object]:
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    from jax_util.functional import initialize_smolyak_integrator, integrate

    class ExponentialIntegrand(eqx.Module):
        coeffs: jax.Array

        def __call__(self, x: jax.Array) -> jax.Array:
            return jnp.asarray([jnp.exp(jnp.dot(self.coeffs, x))], dtype=self.coeffs.dtype)

    cpu_device = jax.devices("cpu")[0]
    platform = _config_str(run_config, "platform")
    target_device = jax.devices("gpu")[0] if platform == "gpu" else cpu_device
    init_device = target_device if platform == "gpu" else cpu_device
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
    t0 = time.perf_counter()
    with jax.default_device(init_device):
        if integration_method == "smolyak":
            integrator = initialize_smolyak_integrator(
                dimension=_case_int(case, "dimension"),
                level=_case_int(case, "level"),
                dtype=runtime_dtype,
                chunk_size=_config_int(run_config, "chunk_size"),
            )
        elif integration_method == "monte_carlo":
            integrator = _build_monte_carlo_integrator(
                _case_int(case, "dimension"),
                _case_int(case, "same_budget_num_points"),
                runtime_dtype,
                _case_int(case, "case_id"),
            )
        else:
            raise ValueError(f"Unsupported integration_method: {integration_method}")
    if integration_method == "smolyak":
        jax.block_until_ready(integrator.rule_nodes)
        jax.block_until_ready(integrator.rule_weights)
        jax.block_until_ready(integrator.rule_offsets)
        jax.block_until_ready(integrator.rule_lengths)
        jax.block_until_ready(integrator.generation_weights)
        storage_bytes = integrator.storage_bytes
        rule_nodes_dtype = str(integrator.rule_nodes.dtype)
        rule_weights_dtype = str(integrator.rule_weights.dtype)
        num_terms = int(integrator.num_terms)
        num_points = int(integrator.num_evaluation_points)
        num_evaluation_points = int(integrator.num_evaluation_points)
    else:
        jax.block_until_ready(integrator.samples)
        jax.block_until_ready(integrator.key)
        storage_bytes = _array_nbytes(integrator.samples) + _array_nbytes(integrator.key)
        rule_nodes_dtype = None
        rule_weights_dtype = None
        num_terms = None
        num_points = int(integrator.num_samples)
        num_evaluation_points = int(integrator.num_samples)
    t1 = time.perf_counter()
    memory_checkpoints["after_init"] = _compact_memory_stats(target_device)

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
        jax.block_until_ready(integrator.samples)
        jax.block_until_ready(integrator.key)
    jax.block_until_ready(coeffs)
    jax.block_until_ready(accuracy_coeffs)
    t2 = time.perf_counter()
    memory_checkpoints["after_transfer"] = _compact_memory_stats(target_device)

    def single_integral(current_integrator: Any, current_coeffs: jax.Array) -> jax.Array:
        return integrate(ExponentialIntegrand(current_coeffs), current_integrator)[0]

    def batched_accuracy_integrals(current_integrator: Any, coeff_matrix: jax.Array) -> jax.Array:
        def apply_single(coeff_vector: jax.Array) -> jax.Array:
            return single_integral(current_integrator, coeff_vector)

        return jax.vmap(
            apply_single,
            in_axes=0,
            out_axes=0,
        )(coeff_matrix)

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

    t3 = time.perf_counter()
    if execution_variant == "single":
        benchmark_stats = _benchmark_compiled(
            compiled_single,
            integrator,
            coeffs,
            warm_repeats=_config_int(run_config, "num_repeats"),
        )
        memory_checkpoints["after_benchmark"] = _compact_memory_stats(target_device)
        measured_value = compiled_single(integrator, coeffs)
        measured_value = jax.block_until_ready(measured_value)
        t4 = time.perf_counter()
        memory_checkpoints["after_execute"] = _compact_memory_stats(target_device)
        actual_scalar = float(np.asarray(measured_value))
        expected_scalar = float(expected_values[-1])
        abs_error_scalar = abs(actual_scalar - expected_scalar)
        memory_checkpoints["after_host_copy"] = _compact_memory_stats(target_device)
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
        memory_checkpoints["after_benchmark"] = _compact_memory_stats(target_device)
        measured_values = compiled_batch(integrator, accuracy_coeffs)
        measured_values = jax.block_until_ready(measured_values)
        t4 = time.perf_counter()
        memory_checkpoints["after_execute"] = _compact_memory_stats(target_device)
        actual_values = np.asarray(measured_values, dtype=np.float64)
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
        raise ValueError(f"Unsupported execution_variant: {execution_variant}")

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

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
        "backend": jax.default_backend(),
        "device_kind": target_device.device_kind,
        "visible_device_id": int(target_device.id),
        "assigned_gpu_index": os.environ.get("SMOLYAK_GPU_INDEX"),
        "cpu_affinity": sorted(int(cpu) for cpu in os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "num_terms": num_terms,
        "num_points": num_points,
        "num_evaluation_points": num_evaluation_points,
        "storage_bytes": storage_bytes,
        "same_budget_num_points": _case_int(case, "same_budget_num_points"),
        "rule_nodes_dtype": rule_nodes_dtype,
        "rule_weights_dtype": rule_weights_dtype,
        "chunk_size": int(getattr(integrator, "chunk_size", 0)),
        "num_samples": int(getattr(integrator, "num_samples", num_evaluation_points)),
        "expected": expected_value_field,
        "actual": actual_value_field,
        "num_accuracy_problems": _config_int(run_config, "num_accuracy_problems"),
        "measurement_problem_count": measurement_problem_count,
        "coeff_inputs_device_nbytes": _array_nbytes(accuracy_coeffs if execution_variant == "vmap" else coeffs),
        "measured_values_device_nbytes": measured_device_nbytes,
        "dense_integrand_matrix_upper_bound_bytes": dense_integrand_matrix_upper_bound_bytes,
        "mean_abs_err": mean_abs_err,
        "var_abs_err": var_abs_err,
        "max_abs_err": max_abs_err,
        "num_repeats": _config_int(run_config, "num_repeats"),
        "vmap_batch_size": measurement_problem_count,
        "first_call_ms": benchmark_stats["first_call_ms"],
        "compile_ms": benchmark_stats["compile_ms"],
        "warm_runtime_ms": benchmark_stats["warm_runtime_ms"],
        "throughput_integrals_per_second": (
            measurement_problem_count * 1000.0 / benchmark_stats["warm_runtime_ms"]
            if benchmark_stats["warm_runtime_ms"] > 0.0
            else None
        ),
        "integrator_init_seconds": t1 - t0,
        "device_transfer_seconds": t2 - t1,
        "timing_probe_seconds": t4 - t2,
        "warmup_seconds": benchmark_stats["first_call_ms"] / 1000.0,
        "measured_runtime_seconds": benchmark_stats["warm_runtime_ms"] / 1000.0,
        "avg_integral_seconds": benchmark_stats["warm_runtime_ms"] / 1000.0 / measurement_problem_count,
        "process_rss_mb": rss_mb,
        "device_memory_stats": _compact_memory_stats(target_device),
        "memory_checkpoints": memory_checkpoints,
    }


# 責務: 子プロセスで 1 ケースを実行し、失敗も結果として返す。
def _run_case_in_child(
    case: Mapping[str, object],
    run_config: Mapping[str, object],
    worker_slot: WorkerSlot,
    /,
) -> dict[str, object]:
    try:
        result = _run_single_case(case, run_config)
    except Exception as exc:
        device_memory_stats: dict[str, int] | None = None
        try:
            import jax

            if _config_str(run_config, "platform") == "gpu":
                device_memory_stats = _compact_memory_stats(jax.devices("gpu")[0])
        except Exception:
            device_memory_stats = None
        message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        traceback_text = traceback.format_exc(limit=8)
        failure_kind = _failure_kind_from_exception(exc, traceback_text)
        result = _build_failure_result_record(
            case,
            worker_slot,
            failure_kind=failure_kind,
            runner_failure_kind=FailureKind.PYTHON_EXCEPTION.value,
            failure_source="child",
            error=message,
            traceback_text=traceback_text,
            device_memory_stats=device_memory_stats,
        )

    result["worker_label"] = worker_slot.worker_label
    if "assigned_gpu_index" not in result:
        result["assigned_gpu_index"] = worker_slot.gpu_index
    result["gpu_slot"] = worker_slot.gpu_slot
    result["cpu_affinity"] = list(worker_slot.cpu_affinity)
    return result


def _build_child_command(
    case: Mapping[str, object],
    run_config: Mapping[str, object],
    worker_slot: WorkerSlot,
    jsonl_output_path: Path,
    /,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child-case-json",
        json.dumps(json_compatible(case), ensure_ascii=True),
        "--child-run-config-json",
        json.dumps(json_compatible(run_config), ensure_ascii=True),
        "--child-worker-slot-json",
        json.dumps(json_compatible(worker_slot.to_dict()), ensure_ascii=True),
        "--child-jsonl-output",
        str(jsonl_output_path),
    ]


def _log_case_started(case: Mapping[str, object], worker_slot: WorkerSlot, /) -> None:
    started_at = datetime.now(timezone.utc).isoformat()
    print(f"[{started_at}] start {worker_slot.worker_label} {_case_label(case)}", flush=True)


def _log_case_finished(
    case: Mapping[str, object],
    worker_slot: WorkerSlot,
    result: Mapping[str, object],
    /,
) -> None:
    finished_at = datetime.now(timezone.utc).isoformat()
    status = str(result.get("status", "unknown"))
    failure_kind = str(result.get("failure_kind", "-"))
    print(
        f"[{finished_at}] done  {worker_slot.worker_label} {_case_label(case)} status={status} failure_kind={failure_kind}",
        flush=True,
    )


def _run_cases_with_host_scheduler(
    cases: list[dict[str, object]],
    run_config: Mapping[str, object],
    jsonl_output_path: Path,
    /,
) -> list[dict[str, object]]:
    platform = _config_str(run_config, "platform")
    gpu_indices_value = run_config.get("gpu_indices")
    gpu_indices = [int(gpu_index) for gpu_index in gpu_indices_value] if isinstance(gpu_indices_value, list) else []
    worker_slots = build_worker_slots(platform, gpu_indices, _config_int(run_config, "workers_per_gpu"))
    monitor = _build_runtime_monitor(run_config)
    if monitor is not None:
        monitor.start()
        if bool(run_config.get("monitor_enable_http", True)):
            print(
                f"[monitor] http://{_config_str(run_config, 'monitor_bind_host')}:{_config_int(run_config, 'monitor_port')}",
                flush=True,
            )
    try:
        results = run_cases_with_subprocess_scheduler(
            cases,
            worker_slots,
            timeout_seconds=_config_int(run_config, "timeout_seconds"),
            build_child_command=lambda case, worker_slot: _build_child_command(case, run_config, worker_slot, jsonl_output_path),
            build_parent_failure_result=_parent_failure_result,
            fallback_jsonl_output_path=jsonl_output_path,
            cwd=WORKSPACE_ROOT,
            on_case_started=_log_case_started,
            on_case_finished=_log_case_finished,
            monitor=monitor,
        )
    finally:
        if monitor is not None:
            monitor.stop()
    results.sort(key=lambda result: _result_int(result, "case_id"))
    return results


def _child_main(
    case_json: str,
    run_config_json: str,
    worker_slot_json: str,
    jsonl_output_path: Path,
    /,
) -> None:
    case = json.loads(case_json)
    run_config = json.loads(run_config_json)
    worker_slot = worker_slot_from_mapping(json.loads(worker_slot_json))
    apply_worker_environment(
        platform=_config_str(run_config, "platform"),
        worker_slot=worker_slot,
        disable_gpu_preallocation=GPU_PREALLOCATION_DISABLED,
    )
    os.environ.update(
        build_env_for_profile(
            _config_str(run_config, "platform"),
            disable_preallocation=GPU_PREALLOCATION_DISABLED,
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
    result = _run_case_in_child(case, run_config, worker_slot)
    append_jsonl_record(jsonl_output_path, result)
    print(f"{CHILD_COMPLETE_PREFIX}{json.dumps(json_compatible(result), ensure_ascii=True)}", flush=True)


# 責務: dtype ごとの誤差と実行時間を要約する。
def _summary_by_dtype(results: list[dict[str, object]], dtype_names: list[str], /) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
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
                ok_results = [result for result in dtype_results if result.get("status") == "ok"]
                failed_results = [result for result in dtype_results if result.get("status") != "ok"]

                mean_abs_errs = [
                    _result_float(result, "mean_abs_err")
                    for result in ok_results
                    if isinstance(result.get("mean_abs_err"), (int, float))
                ]
                var_abs_errs = [
                    _result_float(result, "var_abs_err")
                    for result in ok_results
                    if isinstance(result.get("var_abs_err"), (int, float))
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
                    "num_cases": len(dtype_results),
                    "num_success": len(ok_results),
                    "num_failure": len(failed_results),
                    "max_mean_abs_err": max(mean_abs_errs) if mean_abs_errs else None,
                    "mean_mean_abs_err": float(np.mean(mean_abs_errs)) if mean_abs_errs else None,
                    "max_var_abs_err": max(var_abs_errs) if var_abs_errs else None,
                    "mean_var_abs_err": float(np.mean(var_abs_errs)) if var_abs_errs else None,
                    "mean_avg_integral_seconds": float(np.mean(avg_times)) if avg_times else None,
                    "min_avg_integral_seconds": min(avg_times) if avg_times else None,
                    "max_avg_integral_seconds": max(avg_times) if avg_times else None,
                })
    return summaries


# 責務: dtype ごとの各レベル成功 frontier を要約する。
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
                    level_results = [result for result in dtype_results if result["level"] == level]
                    success_dimensions = [_result_int(result, "dimension") for result in level_results if result["status"] == "ok"]
                    failure_dimensions = [_result_int(result, "dimension") for result in level_results if result["status"] != "ok"]
                    frontier.append({
                        "integration_method": integration_method,
                        "execution_variant": execution_variant,
                        "dtype_name": dtype_name,
                        "level": level,
                        "max_success_dimension": max(success_dimensions) if success_dimensions else None,
                        "min_failure_dimension": min(failure_dimensions) if failure_dimensions else None,
                        "num_success": len(success_dimensions),
                        "num_failure": len(failure_dimensions),
                    })
    return frontier


# 責務: dtype ごとの各次元成功 frontier を要約する。
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
                    dimension_results = [result for result in dtype_results if result["dimension"] == dimension]
                    success_levels = [_result_int(result, "level") for result in dimension_results if result["status"] == "ok"]
                    failure_levels = [_result_int(result, "level") for result in dimension_results if result["status"] != "ok"]
                    frontier.append({
                        "integration_method": integration_method,
                        "execution_variant": execution_variant,
                        "dtype_name": dtype_name,
                        "dimension": dimension,
                        "max_success_level": max(success_levels) if success_levels else None,
                        "min_failure_level": min(failure_levels) if failure_levels else None,
                        "num_success": len(success_levels),
                        "num_failure": len(failure_levels),
                    })
    return frontier


# 責務: ベンチマーク実行全体を構成して JSON 互換の結果へまとめる。
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
    chunk_size: int,
    num_repeats: int,
    num_accuracy_problems: int,
    coeff_start: float,
    coeff_stop: float,
    xla_memory_fraction: float | None,
    xla_allocator: str | None,
    xla_tf_gpu_allocator: str | None,
    xla_use_cuda_host_allocator: bool | None,
    xla_memory_scheduler: str | None,
    xla_gpu_enable_while_loop_double_buffering: bool | None,
    xla_latency_hiding_scheduler_rerun: int | None,
    jax_compiler_enable_remat_pass: bool | None,
    monitor_port: int | None,
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
        "chunk_size": chunk_size,
        "num_repeats": num_repeats,
        "num_accuracy_problems": num_accuracy_problems,
        "coeff_start": coeff_start,
        "coeff_stop": coeff_stop,
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
    cases = _build_cases(dimensions, levels, dtype_names, integration_methods, execution_variants)
    results = _run_cases_with_host_scheduler(cases, run_config, jsonl_output_path)
    finished_at = datetime.now(timezone.utc)

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
        "num_repeats": num_repeats,
        "num_accuracy_problems": num_accuracy_problems,
        "timeout_seconds": timeout_seconds,
        "chunk_size": chunk_size,
        "coeff_start": coeff_start,
        "coeff_stop": coeff_stop,
        "monitor_port": monitor_port,
        "monitor_bind_host": monitor_bind_host if monitor_port is not None else None,
        "monitor_sample_interval_seconds": (
            monitor_sample_interval_seconds if monitor_port is not None else None
        ),
        "monitor_enable_http": monitor_enable_http if monitor_port is not None else None,
        "jsonl_output_path": str(jsonl_output_path),
        "xla_config": _xla_config_from_run_config(run_config),
        **metadata,
        "cases": results,
        "summary_by_dtype": _summary_by_dtype(results, dtype_names),
        "frontier_by_dtype_and_level": _frontier_by_dtype_and_level(results, dtype_names, levels),
        "frontier_by_dtype_and_dimension": _frontier_by_dtype_and_dimension(results, dtype_names, dimensions),
    }


# 責務: 実験結果を JSON ファイルへ保存する。
def save_results(results: dict[str, object], output_path: Path, /) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_compatible(results), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Smolyak scaling on a dimension/level range.")
    parser.add_argument("--dimensions", help="Inclusive integer range start:end[:step] for dimensions.")
    parser.add_argument("--levels", help="Inclusive integer range start:end[:step] for levels.")
    parser.add_argument("--platform", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--gpu-indices", default=None, help="Comma-separated physical GPU indices. Defaults to all visible GPUs.")
    parser.add_argument("--workers-per-gpu", type=int, default=DEFAULT_WORKERS_PER_GPU)
    parser.add_argument("--dtypes", default="all", help="Comma-separated dtype names or 'all'.")
    parser.add_argument(
        "--integration-methods",
        default="all",
        help="Comma-separated integration methods or 'all'. Supported: smolyak,monte_carlo",
    )
    parser.add_argument(
        "--execution-variants",
        default="all",
        help="Comma-separated execution variants or 'all'. Supported: single,vmap",
    )
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--num-repeats", type=int, default=DEFAULT_NUM_REPEATS)
    parser.add_argument("--num-accuracy-problems", type=int, default=DEFAULT_NUM_ACCURACY_PROBLEMS)
    parser.add_argument("--coeff-start", type=float, default=DEFAULT_COEFF_START)
    parser.add_argument("--coeff-stop", type=float, default=DEFAULT_COEFF_STOP)
    parser.add_argument("--xla-memory-fraction", type=float, default=None)
    parser.add_argument("--xla-allocator", type=str, default=None)
    parser.add_argument("--xla-tf-gpu-allocator", type=str, default=None)
    parser.add_argument("--xla-use-cuda-host-allocator", type=str, default=None)
    parser.add_argument("--xla-memory-scheduler", type=str, default=None)
    parser.add_argument("--xla-gpu-enable-while-loop-double-buffering", type=str, default=None)
    parser.add_argument("--xla-latency-hiding-scheduler-rerun", type=int, default=None)
    parser.add_argument("--jax-compiler-enable-remat-pass", type=str, default=None)
    parser.add_argument("--monitor-port", type=int, default=None, help="Enable runtime monitor HTTP server on this port.")
    parser.add_argument("--monitor-bind-host", type=str, default="127.0.0.1")
    parser.add_argument("--monitor-sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--monitor-enable-http", type=str, default="true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--child-case-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--child-run-config-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--child-worker-slot-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--child-jsonl-output", type=Path, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if (
        args.child_case_json is not None
        and args.child_run_config_json is not None
        and args.child_worker_slot_json is not None
        and args.child_jsonl_output is not None
    ):
        _child_main(
            args.child_case_json,
            args.child_run_config_json,
            args.child_worker_slot_json,
            args.child_jsonl_output,
        )
        return

    if args.dimensions is None or args.levels is None:
        raise ValueError("--dimensions and --levels are required outside child mode.")

    if args.num_accuracy_problems < 1:
        raise ValueError("--num-accuracy-problems must be positive.")

    dimensions = _parse_integer_range(args.dimensions)
    levels = _parse_integer_range(args.levels)
    dtype_names = _parse_dtype_names(args.dtypes)
    integration_methods = _parse_integration_methods(args.integration_methods)
    execution_variants = _parse_execution_variants(args.execution_variants)

    if args.platform == "gpu":
        gpu_indices = _parse_gpu_indices(args.gpu_indices) if args.gpu_indices is not None else _discover_gpu_indices()
        if not gpu_indices:
            raise RuntimeError("No GPUs were discovered for gpu platform.")
    else:
        gpu_indices = []

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = args.output or (RESULTS_DIR / f"smolyak_scaling_{args.platform}_{timestamp}.json")
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
        chunk_size=args.chunk_size,
        num_repeats=args.num_repeats,
        num_accuracy_problems=args.num_accuracy_problems,
        coeff_start=args.coeff_start,
        coeff_stop=args.coeff_stop,
        xla_memory_fraction=args.xla_memory_fraction,
        xla_allocator=args.xla_allocator,
        xla_tf_gpu_allocator=args.xla_tf_gpu_allocator,
        xla_use_cuda_host_allocator=_parse_optional_bool_flag(args.xla_use_cuda_host_allocator),
        xla_memory_scheduler=args.xla_memory_scheduler,
        xla_gpu_enable_while_loop_double_buffering=_parse_optional_bool_flag(
            args.xla_gpu_enable_while_loop_double_buffering
        ),
        xla_latency_hiding_scheduler_rerun=args.xla_latency_hiding_scheduler_rerun,
        jax_compiler_enable_remat_pass=_parse_optional_bool_flag(
            args.jax_compiler_enable_remat_pass
        ),
        monitor_port=args.monitor_port,
        monitor_bind_host=args.monitor_bind_host,
        monitor_sample_interval_seconds=args.monitor_sample_interval_seconds,
        monitor_enable_http=bool(_parse_optional_bool_flag(args.monitor_enable_http)),
    )

    save_results(results, output_path)
    save_results(results, RESULTS_DIR / "latest.json")
    jsonl_output_path = _jsonl_path_for_output(output_path)
    if jsonl_output_path.exists():
        shutil.copyfile(jsonl_output_path, RESULTS_DIR / "latest.jsonl")
    print(output_path)


if __name__ == "__main__":
    main()
