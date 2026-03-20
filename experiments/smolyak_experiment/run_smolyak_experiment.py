#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smolyak 積分器大規模実験実行スクリプト

次元・レベル・データ型の全組み合わせで Smolyak 積分器の
初期化時間・実行時間を測定し、スケーリング特性を分析する。

StandardFullResourceScheduler を使用して、リソース認識型の
並列実行を行う。ワーカー数・ホストメモリ を追跡しながら
効率的にタスクをスケジューリング。
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

# ワークスペース構成を解決
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
WORKSPACE_ROOT = SCRIPT_DIR
if str(WORKSPACE_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "python"))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from jax_util.functional.smolyak import SmolyakIntegrator
from jax_util.experiment_runner import (
    FullResourceCapacity,
    FullResourceEstimate,
    StandardFullResourceScheduler,
    StandardRunner,
    TaskContext,
)
from jax_util.experiment_runner.protocols import Worker
from experiments.smolyak_experiment import cases, runner_config, results_aggregator


def run_single_case(case: dict[str, Any]) -> dict[str, Any]:
    """
    単一ケースを実行し、結果を返す。
    
    Parameters
    ----------
    case : dict
        {"dimension": int, "level": int, "dtype": str, ...}
        
    Returns
    -------
    dict
        {"case_id": str, "status": "SUCCESS"|"FAILURE", "init_time": float, ...}
    """
    case_id = case["case_id"]
    dimension = case["dimension"]
    level = case["level"]
    dtype = case["dtype"]
    
    result = {
        "case_id": case_id,
        "dimension": dimension,
        "level": level,
        "dtype": dtype,
        "trial_index": case["trial_index"],
        "status": "SUCCESS",
        "init_time_ms": 0.0,
        "integrate_time_ms": 0.0,
        "num_evaluation_points": 0,
        "error": None,
    }
    
    try:
        # JAX dtype に変換
        jax_dtype = getattr(jnp, dtype)
        
        # 初期化時間を測定
        start_time = time.perf_counter()
        integrator = SmolyakIntegrator(
            dimension=dimension,
            level=level,
            dtype=jax_dtype,
        )
        init_time = (time.perf_counter() - start_time) * 1000.0  # ミリ秒
        
        result["init_time_ms"] = init_time
        result["num_evaluation_points"] = int(integrator.num_evaluation_points)
        
        # 簡単な被積分関数：f(x) = sum(x_i^2)
        def integrand(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sum(x**2, axis=-1)
        
        # 積分を実行（2回測定）
        run_times = []
        for _ in range(2):
            start_time = time.perf_counter()
            result_value = integrator.integrate(integrand)
            run_time = (time.perf_counter() - start_time) * 1000.0
            run_times.append(run_time)
        
        result["integrate_time_ms"] = np.mean(run_times)
        
    except Exception as e:
        result["status"] = "FAILURE"
        result["error"] = str(e)
    
    return result


class SmolyakWorker(Worker):
    """
    Smolyak リソース認識ワーカー。
    
    StandardFullResourceScheduler と協働して、リソース見積もりに基づいた
    スケジューリングを行う。
    """
    
    def resource_estimate(self, case: dict[str, Any]) -> FullResourceEstimate:
        """ケースのリソース見積もりを返す。"""
        return cases.estimate_case_resources(case)
    
    def __call__(self, case: dict[str, Any], context: TaskContext) -> None:
        """タスクを実行"""
        result = run_single_case(case)
        context["result"] = result


def run_experiment(args: argparse.Namespace) -> None:
    """
    本実験を実行する。
    
    サイズ（--small / --medium / --large / --default）に応じて
    パラメータを選択し、StandardFullResourceScheduler で並列実行。
    """
    
    # サイズ別の構成
    configs_by_size = {
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
        "medium": runner_config.SmolyakExperimentConfig(
            min_dimension=1,
            max_dimension=10,
            min_level=1,
            max_level=10,
            dtypes=["float32", "float64"],
            num_trials=2,
            device="cpu",
        ),
        "large": runner_config.SmolyakExperimentConfig(
            min_dimension=1,
            max_dimension=20,
            min_level=1,
            max_level=20,
            dtypes=["float16", "bfloat16", "float32", "float64"],
            num_trials=3,
            device="cpu",
        ),
        "full": runner_config.SmolyakExperimentConfig(
            min_dimension=1,
            max_dimension=50,
            min_level=1,
            max_level=50,
            dtypes=["float16", "bfloat16", "float32", "float64"],
            num_trials=3,
            device="cpu",
        ),
    }
    
    # サイズを選択
    size = args.size or "small"
    if size not in configs_by_size:
        print(f"Error: size must be one of {list(configs_by_size.keys())}")
        sys.exit(1)
    
    config = configs_by_size[size]
    config.validate()
    
    print("=" * 70)
    print(f"Smolyak Experiment - {size.upper()}")
    print("=" * 70)
    
    print(f"\nConfig:")
    print(f"  Dimensions: {config.min_dimension}-{config.max_dimension}")
    print(f"  Levels: {config.min_level}-{config.max_level}")
    print(f"  dtypes: {config.dtypes}")
    print(f"  Trials per case: {config.num_trials}")
    print(f"  Total cases: {config.total_cases}")
    print(f"  Total tasks: {config.total_tasks}")
    
    # ケース生成
    case_list = cases.generate_cases(config)
    print(f"\nGenerated {len(case_list)} cases")
    
    # リソース容量を定義
    # ホストメモリ: 16 GB（実システムに応じて調整）
    # ワーカー: 4 並列
    resource_capacity = FullResourceCapacity(
        max_workers=4,
        host_memory_bytes=16 * 1024 * 1024 * 1024,  # 16 GB
        gpu_devices=(),  # CPU only
    )
    
    # ワーカーを作成
    worker = SmolyakWorker()
    
    # コンテキストビルダー
    def context_builder(case: dict[str, Any]) -> TaskContext:
        return {"case_id": case["case_id"]}
    
    # スケジューラを作成
    scheduler = StandardFullResourceScheduler.from_worker(
        resource_capacity=resource_capacity,
        cases=case_list,
        worker=worker,
        context_builder=context_builder,
    )
    
    print(f"\nResource Capacity:")
    print(f"  Max workers: {resource_capacity.max_workers}")
    print(f"  Host memory: {resource_capacity.host_memory_bytes / (1024**3):.1f} GB")
    print(f"  GPU devices: {len(resource_capacity.gpu_devices)}")
    
    # 実行
    print("\nRunning cases (sequential execution with resource awareness):")
    print()
    
    # ワーカーを実行（シーケンシャル）
    # JAX は multiprocessing fork() と非互換のため、シーケンシャル実行で十分
    results = []
    completed_count = 0
    start_time = time.perf_counter()
    
    for case in case_list:
        context = {"case_id": case["case_id"]}
        worker(case, context)
        
        if "result" in context:
            result = context["result"]
            results.append(result)
            completed_count += 1
            
            # 定期的に進捗を表示
            if completed_count % 10 == 0 or completed_count == len(case_list):
                elapsed = time.perf_counter() - start_time
                throughput = completed_count / elapsed if elapsed > 0 else 0
                print(
                    f"  [{completed_count:4d}/{len(case_list):4d}] "
                    f"elapsed={elapsed:7.1f}s  throughput={throughput:5.1f} cases/s"
                )
    
    elapsed_total = time.perf_counter() - start_time
    
    # 結果集計
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    
    success_results = [r for r in results if r["status"] == "SUCCESS"]
    failed_results = [r for r in results if r["status"] != "SUCCESS"]
    
    print(f"\nSuccessful: {len(success_results)} / {len(results)}")
    print(f"Failed: {len(failed_results)} / {len(results)}")
    print(f"Total elapsed time: {elapsed_total:.1f} s ({elapsed_total/60:.1f} m)")
    print(f"Throughput: {len(results)/elapsed_total:.1f} cases/s")
    
    if success_results:
        init_times = [r["init_time_ms"] for r in success_results]
        integrate_times = [r["integrate_time_ms"] for r in success_results]
        
        print(f"\nInitialization time statistics (ms):")
        print(f"  Min: {min(init_times):.2f}")
        print(f"  Max: {max(init_times):.2f}")
        print(f"  Mean: {np.mean(init_times):.2f}")
        print(f"  Std: {np.std(init_times):.2f}")
        
        print(f"\nIntegration time statistics (ms):")
        print(f"  Min: {min(integrate_times):.2f}")
        print(f"  Max: {max(integrate_times):.2f}")
        print(f"  Mean: {np.mean(integrate_times):.2f}")
        
        # Dimension 別の初期化時間
        print(f"\nInitialization time by dimension (ms):")
        by_dim = results_aggregator.aggregate_by_dimension(success_results)
        for dim in sorted(by_dim.keys()):
            times = [r["init_time_ms"] for r in by_dim[dim]]
            print(
                f"  d={dim:2d}: mean={np.mean(times):7.2f}, "
                f"min={min(times):7.2f}, max={max(times):7.2f}"
            )
    
    if failed_results:
        print(f"\nFailed cases (first 10):")
        for r in failed_results[:10]:
            print(f"  {r['case_id']}: {r['error']}")
    
    # 結果を JSON に保存
    output_dir = Path(__file__).parent / "results" / size
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"results_{int(time.time())}.json"
    
    with open(output_file, "w") as f:
        json.dump({
            "config": config.to_dict(),
            "resource_capacity": {
                "max_workers": resource_capacity.max_workers,
                "host_memory_bytes": resource_capacity.host_memory_bytes,
                "gpu_devices": len(resource_capacity.gpu_devices),
            },
            "results": results,
            "summary": {
                "total": len(results),
                "success": len(success_results),
                "failed": len(failed_results),
                "elapsed_seconds": elapsed_total,
                "throughput_cases_per_sec": len(results) / elapsed_total if elapsed_total > 0 else 0,
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smolyak integrator scaling experiment (resource-aware)"
    )
    parser.add_argument(
        "--size",
        choices=["smoke", "small", "medium", "large", "full"],
        help="Experiment size (smoke/small/medium/large/full)",
    )
    
    args = parser.parse_args()
    
    # デフォルトは smoke
    if not args.size:
        args.size = "smoke"
    
    run_experiment(args)

