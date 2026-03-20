#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smolyak 積分器大規模実験実行スクリプト

次元・レベル・データ型の全組み合わせで Smolyak 積分器の
初期化時間・実行時間を測定し、スケーリング特性を分析する。
"""

import argparse
import json
import sys
import time
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
from experiments.smolyak_experiment import cases, runner_config, results_aggregator


def run_single_case(case: dict[str, Any], config: runner_config.SmolyakExperimentConfig) -> dict[str, Any]:
    """
    単一ケースを実行し、結果を返す。
    
    Parameters
    ----------
    case : dict
        {"dimension": int, "level": int, "dtype": str, ...}
    config : SmolyakExperimentConfig
        実験構成
        
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
        result["num_evaluation_points"] = int(integrator.num_evaluation_points)
        
    except Exception as e:
        result["status"] = "FAILURE"
        result["error"] = str(e)
    
    return result


def run_smoke_test() -> None:
    """
    小規模パラメータで smoke test を実行。
    
    d=1-3, level=1-2, float32 のみで実行し、
    基本的な動作確認を行う。
    """
    print("=" * 70)
    print("Smolyak Experiment - SMOKE TEST")
    print("=" * 70)
    
    # 小規模構成
    config = runner_config.SmolyakExperimentConfig(
        min_dimension=1,
        max_dimension=3,
        min_level=1,
        max_level=2,
        dtypes=["float32"],
        num_trials=1,
        device="cpu",
    )
    
    config.validate()
    
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
    
    # 実行
    results = []
    print("\nRunning cases:")
    for i, case in enumerate(case_list, 1):
        result = run_single_case(case, config)
        results.append(result)
        status_str = "✓" if result["status"] == "SUCCESS" else "✗"
        print(
            f"  [{i:2d}/{len(case_list)}] {result['case_id']:20s} "
            f"init={result['init_time_ms']:7.2f}ms  "
            f"integrate={result.get('integrate_time_ms', 0):7.2f}ms  {status_str}"
        )
    
    # 結果集計
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    
    success_results = [r for r in results if r["status"] == "SUCCESS"]
    failed_results = [r for r in results if r["status"] != "SUCCESS"]
    
    print(f"\nSuccessful: {len(success_results)} / {len(results)}")
    print(f"Failed: {len(failed_results)} / {len(results)}")
    
    if success_results:
        init_times = [r["init_time_ms"] for r in success_results]
        print(f"\nInitialization time statistics (ms):")
        print(f"  Min: {min(init_times):.2f}")
        print(f"  Max: {max(init_times):.2f}")
        print(f"  Mean: {np.mean(init_times):.2f}")
        print(f"  Std: {np.std(init_times):.2f}")
        
        # Dimension 別の初期化時間
        print(f"\nInitialization time by dimension (ms):")
        by_dim = results_aggregator.aggregate_by_dimension(success_results)
        for dim in sorted(by_dim.keys()):
            times = [r["init_time_ms"] for r in by_dim[dim]]
            print(f"  d={dim}: mean={np.mean(times):.2f}, min={min(times):.2f}, max={max(times):.2f}")
    
    if failed_results:
        print(f"\nFailed cases:")
        for r in failed_results:
            print(f"  {r['case_id']}: {r['error']}")
    
    # JSON 出力
    output_file = Path(__file__).parent / "smoke_test_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "config": config.to_dict(),
            "results": results,
            "summary": {
                "total": len(results),
                "success": len(success_results),
                "failed": len(failed_results),
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print("=" * 70)


def run_full_experiment() -> None:
    """
    本実験を実行: d=1-50, level=1-50, 4 dtype。
    
    推定実行時間: CPU で約 25 時間。
    """
    print("Full experiment would run d=1-50, level=1-50, 4 dtype")
    print("This is typically run in a separate worktree.")
    print("Use --smoke-only for development.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smolyak integrator scaling experiment"
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run smoke test only (d=1-3, level=1-2)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full experiment (d=1-50, level=1-50)",
    )
    
    args = parser.parse_args()
    
    if args.full:
        run_full_experiment()
    else:
        # Default to smoke test
        run_smoke_test()
