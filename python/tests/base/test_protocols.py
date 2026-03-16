from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp

from jax_util.base import DEFAULT_DTYPE
from jax_util.base.protocols import (
    ConstraintedOptimizationProblem,
    LinearOperator,
    Operator,
    OptimizationProblem,
)


SOURCE_FILE = Path(__file__).name


def test_protocols_exist() -> None:
    """プロトコルが import できることを確認します。"""
    assert LinearOperator is not None
    assert Operator is not None
    assert OptimizationProblem is not None
    assert ConstraintedOptimizationProblem is not None
    print(json.dumps({
        "case": "protocols",
        "source_file": SOURCE_FILE,
        "test": "test_protocols_exist",
        "linear_operator": True,
        "operator": True,
        "optimization_problem": True,
        "constrained_optimization_problem": True,
    }))


class _QuadraticProblem:
    def __init__(self) -> None:
        self.objective = lambda x: jnp.sum(x**2)


class _ConstrainedQuadraticProblem:
    def __init__(self) -> None:
        self.objective = lambda x: jnp.sum(x**2)
        self.constraint_eq = lambda x: jnp.asarray([jnp.sum(x)], dtype=DEFAULT_DTYPE)
        self.constraint_ineq = lambda x: jnp.asarray([x[0]], dtype=DEFAULT_DTYPE)


def test_optimization_problem_runtime_protocol() -> None:
    """OptimizationProblem が structural runtime protocol として機能することを確認します。"""
    problem = _QuadraticProblem()
    x = jnp.ones((2,), dtype=DEFAULT_DTYPE)
    assert isinstance(problem, OptimizationProblem)
    assert float(problem.objective(x)) == 2.0


def test_constrained_optimization_problem_runtime_protocol() -> None:
    """ConstraintedOptimizationProblem が structural runtime protocol として機能することを確認します。"""
    problem = _ConstrainedQuadraticProblem()
    x = jnp.ones((2,), dtype=DEFAULT_DTYPE)
    assert isinstance(problem, ConstraintedOptimizationProblem)
    assert problem.constraint_eq(x).shape == (1,)
    assert problem.constraint_ineq(x).shape == (1,)


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_protocols_exist()
    test_optimization_problem_runtime_protocol()
    test_constrained_optimization_problem_runtime_protocol()


if __name__ == "__main__":
    _run_all_tests()
