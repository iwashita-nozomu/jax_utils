#!/usr/bin/env python3
"""Run Smolyak scaling experiments with the experiment runner."""

# results branch note:
# medium / large runs should be recorded on a dedicated results/* branch worktree.

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent
PYTHON_ROOT = WORKSPACE_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from experiment_runner import (  # noqa: E402
    FullResourceCapacity,
    StandardFullResourceScheduler,
    StandardRunner,
    StandardWorker,
    TaskContext,
)
from experiment_runner.result_io import append_jsonl_record, read_jsonl_records  # noqa: E402

from experiments.smolyak_experiment import cases, runner_config  # noqa: E402
from jax_util.xla_env import build_cpu_env, build_gpu_env  # noqa: E402


def _read_jsonl_records(path: Path, /) -> list[dict[str, Any]]:
    return read_jsonl_records(path)


def _numeric_stats(values: list[float], /) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }

    sorted_values = sorted(values)
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "median": sorted_values[len(sorted_values) // 2],
    }


@dataclass(frozen=True)
class SmolyakTask:
    """Task that executes one Smolyak scaling case and persists per-case results."""

    def __call__(self, case: dict[str, Any], context: TaskContext) -> None:
        try:
            t_import_start = time.perf_counter()
            import jax.numpy as jnp
            from jax_util.functional.smolyak import SmolyakIntegrator

            t_import_end = time.perf_counter()

            result = self._run_case(
                case,
                jnp=jnp,
                SmolyakIntegrator=SmolyakIntegrator,
                context=context,
                extra_timers={
                    "t_jax_import_ms": (t_import_end - t_import_start) * 1000.0,
                },
            )
            self._save_result(result, context)
        except Exception as exc:
            error_result = {
                "case_id": case["case_id"],
                "case_params": {
                    "dimension": case["dimension"],
                    "level": case["level"],
                    "dtype": case["dtype"],
                    "trial_index": case["trial_index"],
                },
                "smolyak": {
                    "status": "FAILURE",
                    "init_time_ms": 0.0,
                    "jax_import_time_ms": None,
                    "integrate_time_ms": 0.0,
                    "integrate_first_call_ms": None,
                    "integrate_second_call_ms": None,
                    "compile_time_ms": None,
                    "num_evaluation_points": 0,
                    "storage_bytes": None,
                    "integral_value": 0.0,
                    "analytical_value": 0.0,
                    "absolute_error": 0.0,
                    "relative_error": 0.0,
                    "error": str(exc),
                },
            }
            self._save_result(error_result, context)
            sys.stderr.write(f"[SmolyakTask] case={case['case_id']} error={exc}\n")
            sys.stderr.flush()
            raise

    @staticmethod
    def _run_case(
        case: dict[str, Any],
        *,
        jnp: Any,
        SmolyakIntegrator: Any,
        context: TaskContext,
        extra_timers: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "case_id": case["case_id"],
            "case_params": {
                "dimension": case["dimension"],
                "level": case["level"],
                "dtype": case["dtype"],
                "trial_index": case["trial_index"],
            },
            "smolyak": {
                "status": "SUCCESS",
                "init_time_ms": 0.0,
                "jax_import_time_ms": None,
                "integrate_time_ms": 0.0,
                "integrate_first_call_ms": None,
                "integrate_second_call_ms": None,
                "compile_time_ms": None,
                "num_evaluation_points": 0,
                "storage_bytes": None,
                "integral_value": 0.0,
                "analytical_value": 0.0,
                "absolute_error": 0.0,
                "relative_error": 0.0,
                "error": None,
            },
        }

        if extra_timers is not None and "t_jax_import_ms" in extra_timers:
            result["smolyak"]["jax_import_time_ms"] = float(extra_timers["t_jax_import_ms"])

        try:
            jax_dtype = getattr(jnp, str(case["dtype"]))

            t_init_start = time.perf_counter()
            integrator = SmolyakIntegrator(
                dimension=int(case["dimension"]),
                level=int(case["level"]),
                dtype=jax_dtype,
            )
            t_init_end = time.perf_counter()
            result["smolyak"]["init_time_ms"] = (t_init_end - t_init_start) * 1000.0
            result["smolyak"]["num_evaluation_points"] = int(integrator.num_evaluation_points)
            result["smolyak"]["storage_bytes"] = int(getattr(integrator, "storage_bytes", 0))

            def integrand(x: Any) -> Any:
                return jnp.sum(x**2, axis=-1)

            import equinox as eqx

            jitted_integrate = eqx.filter_jit(integrator.integrate)

            jsonl_path_text = str(context.get("jsonl_path", ""))
            tracing_started = False
            jax_module: Any | None = None
            if jsonl_path_text:
                try:
                    import jax as jax_imported
                    from jax_util.hlo import dump

                    jax_module = jax_imported
                    trace_dir = (
                        Path(jsonl_path_text).parent
                        / f"trace_{case['case_id']}_{int(time.time())}"
                    )
                    trace_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        jax_module.profiler.start_trace(str(trace_dir))
                        tracing_started = True
                    except Exception:
                        tracing_started = False
                    try:
                        dump(
                            lambda: integrator.integrate(integrand),
                            trace_dir / f"hlo_{case['case_id']}.jsonl",
                            tag=str(case["case_id"]),
                        )
                    except Exception:
                        pass
                except Exception:
                    tracing_started = False

            t_first_start = time.perf_counter()
            integral_first = jitted_integrate(integrand)
            t_first_end = time.perf_counter()
            first_ms = (t_first_end - t_first_start) * 1000.0

            try:
                if tracing_started and jax_module is not None:
                    jax_module.profiler.stop_trace()
            except Exception:
                pass

            t_second_start = time.perf_counter()
            integral_second = jitted_integrate(integrand)
            t_second_end = time.perf_counter()
            second_ms = (t_second_end - t_second_start) * 1000.0

            first_value = float(integral_first)
            second_value = float(integral_second)
            analytical_value = float(int(case["dimension"]) / 12.0)
            integral_value = (first_value + second_value) / 2.0
            absolute_error = abs(integral_value - analytical_value)

            result["smolyak"]["integrate_first_call_ms"] = first_ms
            result["smolyak"]["integrate_second_call_ms"] = second_ms
            result["smolyak"]["compile_time_ms"] = max(0.0, first_ms - second_ms)
            result["smolyak"]["integrate_time_ms"] = (first_ms + second_ms) / 2.0
            result["smolyak"]["integral_value"] = integral_value
            result["smolyak"]["analytical_value"] = analytical_value
            result["smolyak"]["absolute_error"] = absolute_error
            result["smolyak"]["relative_error"] = (
                absolute_error / abs(analytical_value) if analytical_value != 0 else 0.0
            )

            try:
                monte_carlo = SmolyakTask._run_monte_carlo(
                    dimension=int(case["dimension"]),
                    seed=case["case_id"],
                    num_samples=int(integrator.num_evaluation_points),
                )
                result["monte_carlo"] = monte_carlo
            except Exception as exc:
                result["monte_carlo"] = {"error": str(exc)}
        except Exception as exc:
            result["smolyak"]["status"] = "FAILURE"
            result["smolyak"]["error"] = str(exc)

        return result

    @staticmethod
    def _save_result(result: dict[str, Any], context: TaskContext) -> None:
        jsonl_path = context.get("jsonl_path")
        if not isinstance(jsonl_path, str) or not jsonl_path:
            return

        append_jsonl_record(Path(jsonl_path), result)

    @staticmethod
    def _run_monte_carlo(
        *,
        dimension: int,
        seed: Any,
        num_samples: int | None = None,
    ) -> dict[str, Any]:
        import numpy as np

        try:
            hash_seed = hash(str(seed)) % (2**31)
        except Exception:
            hash_seed = 42 + dimension

        if num_samples is None:
            num_samples = max(100000, 5000 * dimension)

        rng = np.random.RandomState(hash_seed)
        t_start = time.perf_counter()
        x_samples = rng.uniform(-0.5, 0.5, (num_samples, dimension))
        f_samples = np.sum(x_samples**2, axis=1)
        mc_integral = float(np.mean(f_samples))
        t_end = time.perf_counter()

        analytical_value = dimension / 12.0
        absolute_error = abs(mc_integral - analytical_value)
        relative_error = absolute_error / abs(analytical_value) if analytical_value != 0 else 0.0
        return {
            "num_samples": int(num_samples),
            "time_ms": (t_end - t_start) * 1000.0,
            "integral_value": mc_integral,
            "absolute_error": float(absolute_error),
            "relative_error": float(relative_error),
            "std": float(f_samples.std()),
        }


def _generate_final_results(
    jsonl_path: str | Path,
    config: runner_config.SmolyakExperimentConfig,
    elapsed_seconds: float,
) -> dict[str, Any]:
    records = _read_jsonl_records(Path(jsonl_path))
    smolyak_records = [
        payload
        for payload in (
            record.get("smolyak")
            for record in records
        )
        if isinstance(payload, dict)
    ]
    init_times = [
        float(payload["init_time_ms"])
        for payload in smolyak_records
        if isinstance(payload.get("init_time_ms"), (int, float))
    ]
    integrate_times = [
        float(payload["integrate_time_ms"])
        for payload in smolyak_records
        if isinstance(payload.get("integrate_time_ms"), (int, float))
    ]
    absolute_errors = [
        float(payload["absolute_error"])
        for payload in smolyak_records
        if payload.get("status") == "SUCCESS" and isinstance(payload.get("absolute_error"), (int, float))
    ]
    relative_errors = [
        float(payload["relative_error"])
        for payload in smolyak_records
        if payload.get("status") == "SUCCESS" and isinstance(payload.get("relative_error"), (int, float))
    ]

    cases_by_dtype: dict[str, int] = {}
    for record in records:
        case_params = record.get("case_params", {})
        if not isinstance(case_params, dict):
            continue
        dtype_name = str(case_params.get("dtype", "unknown"))
        cases_by_dtype[dtype_name] = cases_by_dtype.get(dtype_name, 0) + 1

    total_cases = len(records)
    successful_cases = sum(
        1 for payload in smolyak_records if payload.get("status") == "SUCCESS"
    )
    failed_cases = total_cases - successful_cases

    return {
        "condition": config.to_dict(),
        "total_cases": total_cases,
        "successful_cases": successful_cases,
        "failed_cases": failed_cases,
        "success_rate": (successful_cases / total_cases * 100.0) if total_cases > 0 else 0.0,
        "elapsed_seconds": elapsed_seconds,
        "throughput_cases_per_second": (total_cases / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
        "cases_by_dtype": cases_by_dtype,
        "init_time_stats": _numeric_stats(init_times),
        "integrate_time_stats": _numeric_stats(integrate_times),
        "accuracy_stats": {
            "absolute_error": _numeric_stats(absolute_errors),
            "relative_error": _numeric_stats(relative_errors),
        },
    }


def get_experiment_config(size: str, /) -> runner_config.SmolyakExperimentConfig:
    configs = {
        "smoke": runner_config.SmolyakExperimentConfig(
            min_dimension=1,
            max_dimension=3,
            min_level=1,
            max_level=2,
            dtypes=["float32"],
            num_trials=1,
            device="cpu",
        ),
        "small": runner_config.SmolyakExperimentConfig(
            min_dimension=1,
            max_dimension=5,
            min_level=1,
            max_level=5,
            dtypes=["float32"],
            num_trials=2,
            device="cpu",
        ),
        "verified": runner_config.SmolyakExperimentConfig(
            min_dimension=2,
            max_dimension=5,
            min_level=2,
            max_level=3,
            dtypes=["float32"],
            num_trials=1,
            device="cpu",
        ),
        "medium": runner_config.SmolyakExperimentConfig(
            min_dimension=1,
            max_dimension=10,
            min_level=1,
            max_level=10,
            dtypes=["float32", "float64"],
            num_trials=2,
            device="gpu",
        ),
        "large": runner_config.SmolyakExperimentConfig(
            min_dimension=1,
            max_dimension=20,
            min_level=1,
            max_level=20,
            dtypes=["float16", "bfloat16", "float32", "float64"],
            num_trials=3,
            device="gpu",
        ),
    }
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {sorted(configs)}")
    return configs[size]


def main(
    size: str = "smoke",
    *,
    max_cases: int | None = None,
    max_workers: int | None = None,
) -> None:
    config = get_experiment_config(size)
    config.validate()

    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be positive.")

    case_list = cases.generate_cases(config)
    if max_cases is not None:
        if max_cases < 1:
            raise ValueError("max_cases must be positive.")
        case_list = case_list[:max_cases]

    print("=" * 70)
    print(f"Smolyak Experiment - {size.upper()}")
    print("=" * 70)
    print(
        f"Config: dim {config.min_dimension}-{config.max_dimension}, "
        f"level {config.min_level}-{config.max_level}, "
        f"{len(config.dtypes)} dtypes, {len(case_list)} tasks"
    )

    output_dir = SCRIPT_DIR / "results" / size
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = int(time.time())
    jsonl_file = output_dir / f"results_{run_id}.jsonl"
    final_json_file = output_dir / f"final_results_{run_id}.json"

    def context_builder(case: dict[str, Any]) -> TaskContext:
        environment_variables = (
            build_cpu_env()
            if config.device == "cpu"
            else build_gpu_env(disable_preallocation=True)
        )
        return {
            "case_id": case["case_id"],
            "jsonl_path": str(jsonl_file),
            "environment_variables": environment_variables,
        }

    def progress(completed: int, total: int, elapsed: float, running: int) -> None:
        if total <= 0:
            return
        progress_pct = (completed / total) * 100.0
        throughput = completed / elapsed if elapsed > 0 else 0.0
        print(
            f"\r[{completed:4d}/{total:4d}] {progress_pct:5.1f}% | "
            f"Throughput: {throughput:6.2f} cases/s | "
            f"Running: {running:d} | Elapsed: {elapsed:7.1f}s",
            end="",
            flush=True,
        )

    task = SmolyakTask()
    worker = StandardWorker(
        task,
        resource_estimator=cases.estimate_case_resources,
    )
    resource_capacity = FullResourceCapacity.from_system(
        max_workers=max_workers,
        gpu_max_slots=1,
    )
    scheduler = StandardFullResourceScheduler[dict[str, Any]].from_worker(
        cases=case_list,
        worker=worker,
        context_builder=context_builder,
        disable_gpu_preallocation=False,
        resource_capacity=resource_capacity,
    )
    runner = StandardRunner(scheduler, progress_callback=progress)

    t_start = time.time()
    runner.run(worker)
    elapsed_seconds = time.time() - t_start
    print()

    final_results = _generate_final_results(jsonl_file, config, elapsed_seconds)
    with final_json_file.open("w", encoding="utf-8") as handle:
        json.dump(final_results, handle, ensure_ascii=True, indent=2, sort_keys=True)

    print(f"JSONL results: {jsonl_file}")
    print(f"Final JSON: {final_json_file}")
    print("=" * 70)
    print(
        json.dumps(
            {
                "output_jsonl": str(jsonl_file),
                "output_json": str(final_json_file),
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Smolyak scaling experiments.")
    parser.add_argument(
        "--size",
        default="smoke",
        choices=["smoke", "small", "verified", "medium", "large"],
        help="Experiment size preset.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional upper bound on the number of cases to run.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional worker limit. Defaults to auto-detection.",
    )
    args = parser.parse_args()
    main(
        args.size,
        max_cases=args.max_cases,
        max_workers=args.max_workers,
    )
