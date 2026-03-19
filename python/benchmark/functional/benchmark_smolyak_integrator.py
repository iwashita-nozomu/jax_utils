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


def run_light_benchmarks() -> dict[str, Any]:
    """軽量ベンチマーク: 素早く性能傾向を掴む。
    
    実行時間: ~30秒
    対象空間: d=1-8, level=1-5
    用途: デバッグ・開発環境での素早い検証
    """
    print("\n[1/3] Initialization Scaling (Light) d=1-8, level=1...")
    scaling_result = benchmark_initialization_scaling(
        max_dimension=8,
        level=1,
        num_trials=2,
    )
    
    print("\n[2/3] Level Refinement (Light) d=3, level=1-5...")
    refinement_result = benchmark_level_refinement(
        dimension=3,
        max_level=5,
        num_trials=2,
    )
    
    print("\n[3/3] Dtype Comparison (Light) d=4, level=2...")
    dtype_result = benchmark_dtype_comparison(
        dimension=4,
        level=2,
        num_trials=2,
    )
    
    return {
        "benchmarks": [
            scaling_result,
            refinement_result,
            dtype_result,
        ],
    }


def run_heavy_benchmarks() -> dict[str, Any]:
    """中程度ベンチマーク: 詳細な性能分析。
    
    実行時間: ~5-10分
    対象空間: d=1-15, level=1-8
    用途: 主要な性能特性の理解、改善前後の比較
    """
    print("\n[1/3] Initialization Scaling (Heavy) d=1-15, level=1...")
    scaling_result = benchmark_initialization_scaling(
        max_dimension=15,
        level=1,
        num_trials=3,
    )
    
    print("\n[2/3] Level Refinement (Heavy) d=3, level=1-8...")
    refinement_result = benchmark_level_refinement(
        dimension=3,
        max_level=8,
        num_trials=2,
    )
    
    print("\n[3/3] Dtype Comparison (Heavy) d=8, level=3...")
    dtype_result = benchmark_dtype_comparison(
        dimension=8,
        level=3,
        num_trials=2,
    )
    
    return {
        "benchmarks": [
            scaling_result,
            refinement_result,
            dtype_result,
        ],
    }


def run_extreme_benchmarks() -> dict[str, Any]:
    """重量ベンチマーク: 指数的スケーリングの限界確認。
    
    実行時間: ~1時間以上
    対象空間: d=1-20, level=1-10
    用途: 大規模問題での漸近性能、設計限界の特定
    
    注意: 高次元・高レベルでは初期化時間が支配的になり、
          計算時間は exp(d) に従うことが予想される。
    """
    print("\n[1/3] Initialization Scaling (Extreme) d=1-20, level=1...")
    scaling_result = benchmark_initialization_scaling(
        max_dimension=20,
        level=1,
        num_trials=2,
    )
    
    print("\n[2/3] Level Refinement (Extreme) d=4, level=1-10...")
    refinement_result = benchmark_level_refinement(
        dimension=4,
        max_level=10,
        num_trials=1,
    )
    
    print("\n[3/3] Dtype Comparison (Extreme) d=10, level=4...")
    dtype_result = benchmark_dtype_comparison(
        dimension=10,
        level=4,
        num_trials=1,
    )
    
    return {
        "benchmarks": [
            scaling_result,
            refinement_result,
            dtype_result,
        ],
    }


def benchmark_initialization_scaling(
    max_dimension: int = 8,
    level: int = 1,
    num_trials: int = 2,
) -> dict[str, Any]:
    """初期化時間の次元スケーリングを計測。
    
    責務: dimension が増えるとき、初期化時間がどのように増加するかを定量化。
    
    Parameters:
    -----------
    max_dimension : int
        測定する最大次元 (default: 8)
    level : int
        Smolyak level (default: 1)
    num_trials : int
        各ケースの試行回数 (default: 2)
    """
    results = []
    
    for dimension in range(1, max_dimension + 1):
        result = _benchmark_integration_time(
            dimension=dimension,
            level=level,
            dtype=jnp.float64,
            num_trials=num_trials,
        )
        results.append(result)
        print(f"✓ d={dimension}: init={result['init_time']['mean_sec']:.4f}s, integral={result['integral_time']['mean_sec']:.4f}s")
    
    return {
        "benchmark": "initialization_scaling",
        "description": "次元ごとの初期化・積分時間のスケーリング",
        "max_dimension": max_dimension,
        "fixed_level": level,
        "results": results,
    }


def benchmark_level_refinement(
    dimension: int = 3,
    max_level: int = 5,
    num_trials: int = 2,
) -> dict[str, Any]:
    """level 上昇時の実行時間増加（refinement cost）を計測。
    
    責務: level を上げるたびの追加コスト（初期化・積分）を定量化。
    
    Parameters:
    -----------
    dimension : int
        固定する次元 (default: 3)
    max_level : int
        測定する最大 level (default: 5)
    num_trials : int
        各ケースの試行回数 (default: 2)
    """
    results = []
    
    for level in range(1, max_level + 1):
        result = _benchmark_integration_time(
            dimension=dimension,
            level=level,
            dtype=jnp.float64,
            num_trials=num_trials,
        )
        results.append(result)
        print(f"✓ level={level}: init={result['init_time']['mean_sec']:.4f}s, n_eval={result['num_evaluation_points']}")
    
    return {
        "benchmark": "level_refinement",
        "description": "Level 上昇時の初期化・積分コスト",
        "fixed_dimension": dimension,
        "max_level": max_level,
        "results": results,
    }


def benchmark_dtype_comparison(
    dimension: int = 4,
    level: int = 2,
    num_trials: int = 2,
) -> dict[str, Any]:
    """異なる dtype での性能比較。
    
    責務: 精度による初期化・実行時間の差を確認。
    
    Parameters:
    -----------
    dimension : int
        固定する次元 (default: 4)
    level : int
        固定する level (default: 2)
    num_trials : int
        各ケースの試行回数 (default: 2)
    """
    results = []
    
    for dtype in [jnp.float32, jnp.float64]:
        result = _benchmark_integration_time(
            dimension=dimension,
            level=level,
            dtype=dtype,
            num_trials=num_trials,
        )
        results.append(result)
        print(f"✓ {dtype.__name__}: init={result['init_time']['mean_sec']:.4f}s")
    
    return {
        "benchmark": "dtype_comparison",
        "description": "異なる精度での性能比較",
        "fixed_dimension": dimension,
        "fixed_level": level,
        "results": results,
    }


def run_all_benchmarks(
    level: str = "light",
    output_file: str | Path | None = None,
) -> dict[str, Any]:
    """指定したレベルのベンチマークを実行。
    
    Parameters:
    -----------
    level : str
        ベンチマークレベル: "light" (既定), "heavy", "extreme"
    output_file : str | Path, optional
        結果を出力する JSON ファイル (default: None で出力しない)
    
    Returns:
    --------
    dict[str, Any]
        ベンチマーク結果を統合した辞書
    """
    print("=" * 60)
    print("Smolyak Integrator Benchmark Suite")
    print("=" * 60)
    print()
    
    if level.lower() == "light":
        benchmark_result = run_light_benchmarks()
    elif level.lower() == "heavy":
        benchmark_result = run_heavy_benchmarks()
    elif level.lower() == "extreme":
        benchmark_result = run_extreme_benchmarks()
    else:
        raise ValueError(
            f"不明なレベル: {level}. "
            "'light', 'heavy', 'extreme' のいずれかを指定してください。"
        )
    
    all_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_level": level,
        "suite": benchmark_result,
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
    
    # コマンドラインオプション処理
    benchmark_level = "light"
    output_file = None
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg in ("--light", "--heavy", "--extreme"):
            benchmark_level = arg.lstrip("-")
        elif arg == "--output" and i + 2 < len(sys.argv):
            output_file = sys.argv[i + 2]
    
    # 最後の引数が出力ファイルパスの場合
    if len(sys.argv) > 1 and not sys.argv[-1].startswith("--"):
        output_file = sys.argv[-1]
    
    run_all_benchmarks(level=benchmark_level, output_file=output_file)
