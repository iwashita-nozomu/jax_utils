# Results branch: results/functional-smolyak-scaling
from __future__ import annotations

import argparse
import functools
import json
import os
import resource
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
from numpy.typing import NDArray


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = WORKSPACE_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from experiment_runner.execution_result import ExecutionResult
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
from jax_util.xla_env import build_cpu_env
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_NUM_REPEATS = 100
DEFAULT_NUM_ACCURACY_PROBLEMS = 9
DEFAULT_COEFF_START = -0.55
DEFAULT_COEFF_STOP = 0.65
DEFAULT_WORKERS_PER_GPU = 2
SUPPORTED_FLOAT_DTYPES = ("float16", "bfloat16", "float32", "float64")
RESULTS_BRANCH_NAME = "results/functional-smolyak-scaling"
GPU_PREALLOCATION_DISABLED = True


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


# 責務: レンジ指定から解析解つきベンチマークケース列を構成する。
def _build_cases(
    dimensions: list[int],
    levels: list[int],
    dtype_names: list[str],
    /,
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    case_id = 0
    for dimension in dimensions:
        for level in levels:
            for dtype_name in dtype_names:
                cases.append({
                    "case_id": case_id,
                    "dimension": dimension,
                    "level": level,
                    "dtype_name": dtype_name,
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
    return "error"


# 責務: ケース識別用の短い文字列を作る。
def _case_label(case: Mapping[str, object], /) -> str:
    return (
        f"case={_case_int(case, 'case_id')} "
        f"d={_case_int(case, 'dimension')} "
        f"l={_case_int(case, 'level')} "
        f"dtype={case['dtype_name']}"
    )


# 責務: 単一ケースのベンチマークを実行して JSON 互換結果へまとめる。
def _run_single_case(case: Mapping[str, object], run_config: Mapping[str, object], /) -> dict[str, object]:
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from jax import lax

    from jax_util.functional import initialize_smolyak_integrator

    class ExponentialIntegrand(eqx.Module):
        coeffs: jax.Array

        def __call__(self, x: jax.Array) -> jax.Array:
            return jnp.asarray([jnp.exp(jnp.dot(self.coeffs, x))], dtype=self.coeffs.dtype)

    cpu_device = jax.devices("cpu")[0]
    platform = _config_str(run_config, "platform")
    target_device = jax.devices("gpu")[0] if platform == "gpu" else cpu_device
    runtime_dtype = getattr(jnp, str(case["dtype_name"]))
    accuracy_coefficients = _build_accuracy_coefficients(
        _case_int(case, "dimension"),
        _config_float(run_config, "coeff_start"),
        _config_float(run_config, "coeff_stop"),
        _config_int(run_config, "num_accuracy_problems"),
    )

    t0 = time.perf_counter()
    with jax.default_device(cpu_device):
        integrator = initialize_smolyak_integrator(
            dimension=_case_int(case, "dimension"),
            level=_case_int(case, "level"),
            dtype=runtime_dtype,
        )
    jax.block_until_ready(integrator.rule_nodes)
    jax.block_until_ready(integrator.rule_weights)
    jax.block_until_ready(integrator.rule_offsets)
    jax.block_until_ready(integrator.rule_lengths)
    jax.block_until_ready(integrator.term_levels)
    jax.block_until_ready(integrator.term_num_points)
    t1 = time.perf_counter()

    storage_bytes = integrator.storage_bytes
    integrator = jax.device_put(integrator, target_device)
    coeffs = jax.device_put(jnp.asarray(accuracy_coefficients[-1], dtype=runtime_dtype), target_device)
    accuracy_coeffs = jax.device_put(jnp.asarray(accuracy_coefficients, dtype=runtime_dtype), target_device)
    jax.block_until_ready(integrator.rule_nodes)
    jax.block_until_ready(integrator.rule_weights)
    jax.block_until_ready(integrator.rule_offsets)
    jax.block_until_ready(integrator.rule_lengths)
    jax.block_until_ready(integrator.term_levels)
    jax.block_until_ready(integrator.term_num_points)
    jax.block_until_ready(coeffs)
    jax.block_until_ready(accuracy_coeffs)
    t2 = time.perf_counter()

    def single_integral(current_integrator: Any, current_coeffs: jax.Array) -> jax.Array:
        return current_integrator.integrate(ExponentialIntegrand(current_coeffs))[0]

    def batched_accuracy_integrals(current_integrator: Any, coeff_matrix: jax.Array) -> jax.Array:
        def apply_single(coeff_vector: jax.Array) -> jax.Array:
            return single_integral(current_integrator, coeff_vector)

        return jax.vmap(
            apply_single,
            in_axes=0,
            out_axes=0,
        )(coeff_matrix)

    @eqx.filter_jit
    def repeated_integral(current_integrator: Any, current_coeffs: jax.Array) -> jax.Array:
        def body(_: int, acc: jax.Array) -> jax.Array:
            return acc + single_integral(current_integrator, current_coeffs)

        num_repeats = _config_int(run_config, "num_repeats")
        total = lax.fori_loop(0, num_repeats, body, jnp.asarray(0.0, dtype=current_integrator.rule_nodes.dtype))
        return total / num_repeats

    accuracy_values = batched_accuracy_integrals(integrator, accuracy_coeffs)
    accuracy_values = jax.block_until_ready(accuracy_values)
    warmup_value = repeated_integral(integrator, coeffs)
    warmup_value = jax.block_until_ready(warmup_value)
    t3 = time.perf_counter()

    timed_value = repeated_integral(integrator, coeffs)
    timed_value = jax.block_until_ready(timed_value)
    t4 = time.perf_counter()

    expected_values = _analytic_box_exponential_integrals(accuracy_coefficients)
    actual_values = np.asarray(accuracy_values, dtype=np.float64)
    abs_errors = np.abs(actual_values - expected_values)
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    return {
        "status": "ok",
        "case_id": _case_int(case, "case_id"),
        "dimension": _case_int(case, "dimension"),
        "level": _case_int(case, "level"),
        "dtype_name": str(case["dtype_name"]),
        "backend": jax.default_backend(),
        "device_kind": target_device.device_kind,
        "visible_device_id": int(target_device.id),
        "assigned_gpu_index": os.environ.get("gpu_id", "cpu" if platform == "cpu" else None),
        "worker_label": os.environ.get("EXPERIMENT_RUNNER_WORKER_LABEL"),
        "gpu_slot": (
            int(os.environ["EXPERIMENT_RUNNER_GPU_SLOT"])
            if os.environ.get("EXPERIMENT_RUNNER_GPU_SLOT")
            else None
        ),
        "cpu_affinity": sorted(int(cpu) for cpu in os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "num_terms": int(integrator.num_terms),
        "num_points": int(integrator.num_evaluation_points),
        "num_evaluation_points": int(integrator.num_evaluation_points),
        "storage_bytes": storage_bytes,
        "rule_nodes_dtype": str(integrator.rule_nodes.dtype),
        "rule_weights_dtype": str(integrator.rule_weights.dtype),
        "chunk_size": integrator.chunk_size,
        "expected": float(expected_values[-1]),
        "actual": float(np.asarray(timed_value)),
        "num_accuracy_problems": _config_int(run_config, "num_accuracy_problems"),
        "mean_abs_err": float(np.mean(abs_errors)),
        "var_abs_err": float(np.var(abs_errors)),
        "max_abs_err": float(np.max(abs_errors)),
        "num_repeats": _config_int(run_config, "num_repeats"),
        "integrator_init_seconds": t1 - t0,
        "device_transfer_seconds": t2 - t1,
        "warmup_seconds": t3 - t2,
        "batched_integral_seconds": t4 - t3,
        "avg_integral_seconds": (t4 - t3) / _config_int(run_config, "num_repeats"),
        "process_rss_mb": rss_mb,
        "device_memory_stats": _compact_memory_stats(target_device),
    }


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
    if "worker_label" not in result:
        result["worker_label"] = _worker_label_from_context(context)
    if "assigned_gpu_index" not in result:
        result["assigned_gpu_index"] = _assigned_gpu_index_from_context(context)
    result["gpu_slot"] = _gpu_slot_from_context(context)
    result["cpu_affinity"] = _cpu_affinity_from_context(context)
    return result


def _base_environment_variables(platform: str, /) -> dict[str, str]:
    env_vars = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    if platform == "cpu":
        env_vars.update(build_cpu_env())
    return env_vars


def _resource_capacity_for_run(run_config: Mapping[str, object], /) -> FullResourceCapacity:
    platform = _config_str(run_config, "platform")
    if platform == "cpu":
        return FullResourceCapacity.from_system(
            max_workers=1,
            gpu_devices=(),
        )

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
    /,
) -> dict[str, object]:
    del case
    return {
        "run_config": dict(run_config),
        "jsonl_output_path": str(jsonl_output_path),
        "environment_variables": _base_environment_variables(
            _config_str(run_config, "platform")
        ),
    }


def _build_child_failure_result(
    case: Mapping[str, object],
    context: Mapping[str, object],
    exc: BaseException,
    /,
) -> dict[str, object]:
    message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    traceback_text = traceback.format_exc(limit=8)
    failure_kind = _failure_kind_from_exception(exc, traceback_text)
    return _attach_runner_metadata(
        {
            "status": "failed",
            "failure_kind": failure_kind,
            "case_id": _case_int(case, "case_id"),
            "dimension": _case_int(case, "dimension"),
            "level": _case_int(case, "level"),
            "dtype_name": str(case["dtype_name"]),
            "error": message,
            "traceback": traceback_text[-4000:],
        },
        context,
    )


def _parent_completion_record(
    case: Mapping[str, object],
    context: Mapping[str, object],
    result: ExecutionResult,
    /,
) -> dict[str, object]:
    status = "skipped" if result.status == "skipped" else "failed"
    return _attach_runner_metadata(
        {
            "status": status,
            "failure_kind": result.failure_kind,
            "case_id": _case_int(case, "case_id"),
            "dimension": _case_int(case, "dimension"),
            "level": _case_int(case, "level"),
            "dtype_name": str(case["dtype_name"]),
            "error": result.message,
            "traceback": (result.traceback or "")[-4000:],
            "raw_exit_code": result.raw_exit_code,
            "signal_name": result.signal_name,
            "result_source": result.source,
        },
        context,
    )


def _run_case_task(case: Mapping[str, object], context: dict[str, object]) -> None:
    run_config = cast(Mapping[str, object], context["run_config"])
    jsonl_output_path = Path(str(context["jsonl_output_path"]))
    try:
        result = _run_single_case(case, run_config)
    except Exception as exc:
        append_jsonl_record(jsonl_output_path, _build_child_failure_result(case, context, exc))
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
        f"[{finished_at}] done  {_worker_label_from_context(context)} {_case_label(case)} pid={pid} status={result.status} failure_kind={failure_kind}",
        flush=True,
    )


def _run_cases_with_runner(
    cases: list[dict[str, object]],
    run_config: Mapping[str, object],
    jsonl_output_path: Path,
    /,
) -> list[dict[str, object]]:
    worker = StandardWorker(
        _run_case_task,
        resource_estimator=functools.partial(_resource_estimate_for_case, run_config=run_config),
    )
    scheduler = StandardFullResourceScheduler.from_worker(
        cases=cases,
        worker=worker,
        context_builder=lambda case: _context_builder(case, run_config, jsonl_output_path),
        disable_gpu_preallocation=GPU_PREALLOCATION_DISABLED,
        resource_capacity=_resource_capacity_for_run(run_config),
    )
    runner = StandardRunner(
        scheduler,
        case_timeout_seconds=float(_config_int(run_config, "timeout_seconds")),
        on_case_started=_log_case_started,
        on_case_finished=lambda case, context, result, pid: (
            append_jsonl_record(
                jsonl_output_path,
                _parent_completion_record(case, context, result),
            )
            if result.source == "parent" or result.status == "skipped"
            else None
        )
        or _log_case_finished(case, context, result, pid),
    )
    runner.run(worker)
    results = read_jsonl_records(jsonl_output_path)
    results.sort(key=lambda result: _result_int(result, "case_id"))
    return results


# 責務: dtype ごとの誤差と実行時間を要約する。
def _summary_by_dtype(results: list[dict[str, object]], dtype_names: list[str], /) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for dtype_name in dtype_names:
        dtype_results = [result for result in results if result.get("dtype_name") == dtype_name]
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
    for dtype_name in dtype_names:
        dtype_results = [result for result in results if result.get("dtype_name") == dtype_name]
        for level in levels:
            level_results = [result for result in dtype_results if result["level"] == level]
            success_dimensions = [_result_int(result, "dimension") for result in level_results if result["status"] == "ok"]
            failure_dimensions = [_result_int(result, "dimension") for result in level_results if result["status"] != "ok"]
            frontier.append({
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
    for dtype_name in dtype_names:
        dtype_results = [result for result in results if result.get("dtype_name") == dtype_name]
        for dimension in dimensions:
            dimension_results = [result for result in dtype_results if result["dimension"] == dimension]
            success_levels = [_result_int(result, "level") for result in dimension_results if result["status"] == "ok"]
            failure_levels = [_result_int(result, "level") for result in dimension_results if result["status"] != "ok"]
            frontier.append({
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
    }
    metadata = _experiment_metadata()
    cases = _build_cases(dimensions, levels, dtype_names)
    results = _run_cases_with_runner(cases, run_config, jsonl_output_path)
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
        "num_cases": len(cases),
        "num_repeats": num_repeats,
        "num_accuracy_problems": num_accuracy_problems,
        "timeout_seconds": timeout_seconds,
        "coeff_start": coeff_start,
        "coeff_stop": coeff_stop,
        "jsonl_output_path": str(jsonl_output_path),
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
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--num-repeats", type=int, default=DEFAULT_NUM_REPEATS)
    parser.add_argument("--num-accuracy-problems", type=int, default=DEFAULT_NUM_ACCURACY_PROBLEMS)
    parser.add_argument("--coeff-start", type=float, default=DEFAULT_COEFF_START)
    parser.add_argument("--coeff-stop", type=float, default=DEFAULT_COEFF_STOP)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.dimensions is None or args.levels is None:
        raise ValueError("--dimensions and --levels are required.")

    if args.num_accuracy_problems < 1:
        raise ValueError("--num-accuracy-problems must be positive.")

    dimensions = _parse_integer_range(args.dimensions)
    levels = _parse_integer_range(args.levels)
    dtype_names = _parse_dtype_names(args.dtypes)

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
        output_path=output_path,
        platform=args.platform,
        gpu_indices=gpu_indices,
        workers_per_gpu=args.workers_per_gpu,
        timeout_seconds=args.timeout_seconds,
        num_repeats=args.num_repeats,
        num_accuracy_problems=args.num_accuracy_problems,
        coeff_start=args.coeff_start,
        coeff_stop=args.coeff_stop,
    )

    save_results(results, output_path)
    save_results(results, RESULTS_DIR / "latest.json")
    jsonl_output_path = _jsonl_path_for_output(output_path)
    if jsonl_output_path.exists():
        shutil.copyfile(jsonl_output_path, RESULTS_DIR / "latest.jsonl")
    print(output_path)


if __name__ == "__main__":
    main()
