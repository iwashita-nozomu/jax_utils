"""Smolyak 積分器のベンチマーク。

責務:
- SmolyakIntegrator の初期化・実行時間を単一環境で計測
- 次元・レベル・dtype による性能変化を定量化
- CI/CD に組み込み可能な簡潔な JSON output

このベンチマークは、experiment_runner を使った長時間大規模実験とは異なり、
開発環境で素早く性能傾向を掴むためのツール。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from jax_util.functional import (
    Func,
    initialize_smolyak_integrator,
    integrate,
)


def _benchmark_integration_time(
    dimension: int,
    level: int,
    dtype: jnp.dtype | None = None,
    num_trials: int = 1,
) -> dict[str, Any]:
    """Smolyak 積分の初期化・実行時間を計測。
    
    Parameters:
    -----------
    dimension : int
        積分空間の次元
    level : int
        Smolyak level
    dtype : jnp.dtype, optional
        計算精度 (default: float64)
    num_trials : int
        1つのケースを何回実行するか
    
    Returns:
    --------
    dict[str, Any]
        ベンチマーク結果（初期化時間、積分時間、統計）
    """
    if dtype is None:
        dtype = jnp.float64
    
    # ベンチマーク関数: exp(a^T x)
    coefficients = np.ones(dimension, dtype=np.float64)
    def benchmark_func(x):
        return jnp.asarray(
            [jnp.exp(jnp.dot(coefficients.astype(dtype), x))],
            dtype=dtype,
        )
    
    func = Func(benchmark_func)
    
    # 初期化時間を計測
    init_times = []
    for _ in range(num_trials):
        start_init = time.time()
        integrator = initialize_smolyak_integrator(
            dimension=dimension,
            level=level,
            dtype=dtype,
        )
        init_time = time.time() - start_init
        init_times.append(init_time)
    
    # 積分実行時間を計測
    integral_times = []
    for _ in range(num_trials):
        start_integral = time.time()
        _ = integrate(func, integrator)
        integral_time = time.time() - start_integral
        integral_times.append(integral_time)
    
    return {
        "dimension": dimension,
        "level": level,
        "dtype": str(dtype),
        "num_evaluation_points": int(integrator.num_evaluation_points),
        "num_terms": int(integrator.num_terms),
        "init_time": {
            "mean_sec": float(np.mean(init_times)),
            "std_sec": float(np.std(init_times)) if len(init_times) > 1 else 0.0,
            "min_sec": float(np.min(init_times)),
            "max_sec": float(np.max(init_times)),
        },
        "integral_time": {
            "mean_sec": float(np.mean(integral_times)),
            "std_sec": float(np.std(integral_times)) if len(integral_times) > 1 else 0.0,
            "min_sec": float(np.min(integral_times)),
            "max_sec": float(np.max(integral_times)),
        },
    }


def benchmark_initialization_scaling() -> dict[str, Any]:
    """初期化時間の次元スケーリングを計測。
    
    責務: dimension が増えるとき、初期化時間がどのように増加するかを定量化。
    """
    results = []
    
    for dimension in range(1, 9):  # d=1 to 8
        result = _benchmark_integration_time(
            dimension=dimension,
            level=1,  # level を固定
            dtype=jnp.float64,
            num_trials=2,  # 複数回実行して安定性を確認
        )
        results.append(result)
        print(f"✓ d={dimension}: init={result['init_time']['mean_sec']:.4f}s, integral={result['integral_time']['mean_sec']:.4f}s")
    
    return {
        "benchmark": "initialization_scaling",
        "description": "次元ごとの初期化・積分時間のスケーリング",
        "fixed_level": 1,
        "results": results,
    }


def benchmark_level_refinement() -> dict[str, Any]:
    """level 上昇時の実行時間増加（refinement cost）を計測。
    
    責務: level を上げるたびの追加コスト（初期化・積分）を定量化。
    """
    dimension = 3
    results = []
    
    for level in range(1, 6):  # level=1 to 5
        result = _benchmark_integration_time(
            dimension=dimension,
            level=level,
            dtype=jnp.float64,
            num_trials=2,
        )
        results.append(result)
        print(f"✓ level={level}: init={result['init_time']['mean_sec']:.4f}s, n_eval={result['num_evaluation_points']}")
    
    return {
        "benchmark": "level_refinement",
        "description": "Level 上昇時の初期化・積分コスト",
        "fixed_dimension": dimension,
        "results": results,
    }


def benchmark_dtype_comparison() -> dict[str, Any]:
    """異なる dtype での性能比較。
    
    責務: 精度による初期化・実行時間の差を確認。
    """
    dimension = 4
    level = 2
    results = []
    
    for dtype in [jnp.float32, jnp.float64]:
        result = _benchmark_integration_time(
            dimension=dimension,
            level=level,
            dtype=dtype,
            num_trials=2,
        )
        results.append(result)
        print(f"✓ {dtype.__name__}: init={result['init_time']['mean_sec']:.4f}s")
    
    return {
        "benchmark": "dtype_comparison",
        "description": "異なる 精度での性能比較",
        "fixed_dimension": dimension,
        "fixed_level": level,
        "results": results,
    }


def run_all_benchmarks(output_file: str | Path | None = None) -> dict[str, Any]:
    """すべてのベンチマークを実行して結果を返す。
    
    Parameters:
    -----------
    output_file : str | Path, optional
        結果を出力する JSON ファイル (default: None で出力しない)
    
    Returns:
    --------
    dict[str, Any]
        すべてのベンチマーク結果を統合した辞書
    """
    print("=" * 60)
    print("Smolyak Integrator Benchmark Suite")
    print("=" * 60)
    
    print("\n[1/3] Initialization Scaling...")
    scaling_result = benchmark_initialization_scaling()
    
    print("\n[2/3] Level Refinement...")
    refinement_result = benchmark_level_refinement()
    
    print("\n[3/3] Dtype Comparison...")
    dtype_result = benchmark_dtype_comparison()
    
    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmarks": [
            scaling_result,
            refinement_result,
            dtype_result,
        ],
    }
    
    if output_file is not None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    output_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_all_benchmarks(output_file=output_file)
