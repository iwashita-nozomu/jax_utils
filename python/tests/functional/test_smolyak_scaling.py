from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from jax_util.functional import Func, SmolyakIntegrator, integrate


SOURCE_FILE = Path(__file__).name
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = WORKSPACE_ROOT / "python"
DEFAULT_PROBE_CASES = [
    (24, 4),
    (20, 5),
    (24, 5),
    (28, 5),
    (32, 5),
    (20, 6),
]
DEFAULT_PROBE_TIMEOUT_SECONDS = 45
LARGE_ACCURACY_CASES = [
    (20, 4, 5.0e-6),
    (24, 4, 1.0e-5),
    (16, 5, 1.0e-8),
]


# 責務: points と weights の保持に必要な最終配列サイズを見積もる。
def _storage_bytes(points: jnp.ndarray, weights: jnp.ndarray) -> int:
    itemsize = points.dtype.itemsize
    return int((points.size + weights.size) * itemsize)


# 責務: exp(a^T x) の [-0.5, 0.5]^d 上積分の解析解を返す。
def _analytic_box_exponential_integral(coefficients: NDArray[np.float64]) -> float:
    factors = (2.0 * np.sinh(0.5 * coefficients)) / coefficients
    return float(np.prod(factors))


# 責務: 高次元でも正規化された定数関数積分が保たれることを確認する。
def test_smolyak_high_dimensional_constant_integral_is_normalized() -> None:
    integrator = SmolyakIntegrator(dimension=24, level=4)
    value = integrate(Func(lambda x: jnp.asarray([1.0], dtype=jnp.float64)), integrator)

    print(json.dumps({
        "case": "functional_smolyak_high_dimensional_constant",
        "source_file": SOURCE_FILE,
        "test": "test_smolyak_high_dimensional_constant_integral_is_normalized",
        "dimension": integrator.dimension,
        "level": integrator.level,
        "num_points": int(integrator.points.shape[1]),
        "storage_bytes": _storage_bytes(integrator.points, integrator.weights),
        "value": float(np.asarray(value[0])),
    }))
    assert jnp.allclose(value, jnp.asarray([1.0], dtype=jnp.float64), atol=1.0e-12)


# 責務: 高次元でも対称性により一次モーメントが消えることを確認する。
def test_smolyak_high_dimensional_linear_moment_vanishes() -> None:
    integrator = SmolyakIntegrator(dimension=16, level=5)
    value = integrate(
        Func(lambda x: jnp.asarray([jnp.sum(x)], dtype=jnp.float64)),
        integrator,
    )

    print(json.dumps({
        "case": "functional_smolyak_high_dimensional_linear",
        "source_file": SOURCE_FILE,
        "test": "test_smolyak_high_dimensional_linear_moment_vanishes",
        "dimension": integrator.dimension,
        "level": integrator.level,
        "num_points": int(integrator.points.shape[1]),
        "storage_bytes": _storage_bytes(integrator.points, integrator.weights),
        "value": float(np.asarray(value[0])),
    }))
    assert float(np.asarray(jnp.abs(value[0]))) < 1.0e-12


# 責務: 高次元でも解析解のある指数関数積分を十分な精度で近似できることを確認する。
def test_smolyak_high_dimensional_analytic_exponential_accuracy() -> None:
    coefficients = np.linspace(-0.55, 0.65, 12, dtype=np.float64)
    integrator = SmolyakIntegrator(dimension=12, level=5)
    expected = _analytic_box_exponential_integral(coefficients)
    value = integrate(
        Func(
            lambda x: jnp.asarray(
                [jnp.exp(jnp.dot(jnp.asarray(coefficients), x))],
                dtype=jnp.float64,
            )
        ),
        integrator,
    )
    abs_err = abs(float(np.asarray(value[0])) - expected)

    print(json.dumps({
        "case": "functional_smolyak_high_dimensional_analytic_exponential",
        "source_file": SOURCE_FILE,
        "test": "test_smolyak_high_dimensional_analytic_exponential_accuracy",
        "dimension": integrator.dimension,
        "level": integrator.level,
        "num_points": int(integrator.points.shape[1]),
        "storage_bytes": _storage_bytes(integrator.points, integrator.weights),
        "expected": expected,
        "actual": float(np.asarray(value[0])),
        "abs_err": abs_err,
    }))
    assert abs_err < 1.0e-9


# 責務: より大きな複数ケースでも解析解つき積分の誤差と保持量を確認する。
def test_smolyak_large_analytic_examples() -> None:
    results: list[dict[str, float | int]] = []

    for dimension, level, tolerance in LARGE_ACCURACY_CASES:
        coefficients = np.linspace(-0.55, 0.65, dimension, dtype=np.float64)
        integrator = SmolyakIntegrator(dimension=dimension, level=level)
        expected = _analytic_box_exponential_integral(coefficients)
        value = integrate(
            Func(
                lambda x, coeffs=coefficients: jnp.asarray(
                    [jnp.exp(jnp.dot(jnp.asarray(coeffs), x))],
                    dtype=jnp.float64,
                )
            ),
            integrator,
        )
        actual = float(np.asarray(value[0]))
        abs_err = abs(actual - expected)

        results.append({
            "dimension": dimension,
            "level": level,
            "num_points": int(integrator.points.shape[1]),
            "storage_bytes": _storage_bytes(integrator.points, integrator.weights),
            "expected": expected,
            "actual": actual,
            "abs_err": abs_err,
            "tolerance": tolerance,
        })
        assert abs_err < tolerance

    print(json.dumps({
        "case": "functional_smolyak_large_analytic_examples",
        "source_file": SOURCE_FILE,
        "test": "test_smolyak_large_analytic_examples",
        "results": results,
    }))


# 責務: 1 ケース分の Smolyak 構成と積分を fresh subprocess で評価して結果を返す。
def _probe_case(dimension: int, level: int, timeout_seconds: int, /) -> dict[str, object]:
    code = f"""
from __future__ import annotations
import json
import time
import jax
import jax.numpy as jnp
from jax_util.functional import Func, SmolyakIntegrator, integrate

started = time.perf_counter()
integrator = SmolyakIntegrator(dimension={dimension}, level={level})
value = integrate(Func(lambda x: jnp.asarray([1.0], dtype=jnp.float64)), integrator)
elapsed = time.perf_counter() - started
storage_bytes = int((integrator.points.size + integrator.weights.size) * integrator.points.dtype.itemsize)
print(json.dumps({{
    "status": "ok",
    "backend": jax.default_backend(),
    "dimension": {dimension},
    "level": {level},
    "num_points": int(integrator.points.shape[1]),
    "storage_bytes": storage_bytes,
    "value": float(value[0]),
    "elapsed_seconds": elapsed,
}}))
"""
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PYTHON_ROOT) if not existing_pythonpath else f"{PYTHON_ROOT}:{existing_pythonpath}"

    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=WORKSPACE_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "dimension": dimension,
            "level": level,
            "timeout_seconds": timeout_seconds,
        }

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()

    if completed.returncode == 0 and stdout:
        result = json.loads(stdout.splitlines()[-1])
        return result

    return {
        "status": "failed",
        "dimension": dimension,
        "level": level,
        "returncode": completed.returncode,
        "stdout": stdout[-400:],
        "stderr": stderr[-400:],
    }


# 責務: 構成を増やしながら最初に失敗する高次元ケースを探索する。
def _run_oom_probe(
    cases: list[tuple[int, int]] = DEFAULT_PROBE_CASES,
    timeout_seconds: int = DEFAULT_PROBE_TIMEOUT_SECONDS,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for dimension, level in cases:
        result = _probe_case(dimension, level, timeout_seconds)
        print(json.dumps({
            "case": "functional_smolyak_oom_probe",
            "source_file": SOURCE_FILE,
            **result,
        }))
        results.append(result)
        if result["status"] != "ok":
            break
    return results


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_smolyak_high_dimensional_constant_integral_is_normalized()
    test_smolyak_high_dimensional_linear_moment_vanishes()
    test_smolyak_high_dimensional_analytic_exponential_accuracy()
    test_smolyak_large_analytic_examples()
    if os.environ.get("JAX_UTIL_RUN_OOM_PROBE") == "1":
        _run_oom_probe()


if __name__ == "__main__":
    _run_all_tests()
