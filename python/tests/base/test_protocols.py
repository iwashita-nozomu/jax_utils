from __future__ import annotations

import json
from pathlib import Path

from jax_util.base.protocols import (
    ConstrainedOptimizationProblem,
    ConstrainedOptimizationState,
    LinearOperator,
    Operator,
    OptimizationProblem,
    OptimizationState,
)
from jax_util.functional.protocols import FunctionalOptimizationProblem
from jax_util.neuralnetwork.protocols import PyTreeOptimizationProblem
from jax_util.optimizers.protocols import VectorOptimizationProblem


SOURCE_FILE = Path(__file__).name


def test_protocols_exist() -> None:
    """プロトコルが import できることを確認します。"""
    assert LinearOperator is not None
    assert Operator is not None
    assert OptimizationProblem is not None
    assert ConstrainedOptimizationProblem is not None
    assert OptimizationState is not None
    assert ConstrainedOptimizationState is not None
    assert VectorOptimizationProblem is not None
    assert PyTreeOptimizationProblem is not None
    assert FunctionalOptimizationProblem is not None
    print(json.dumps({
        "case": "protocols",
        "source_file": SOURCE_FILE,
        "test": "test_protocols_exist",
        "linear_operator": True,
        "operator": True,
        "optimization_problem": True,
        "constrained_optimization_problem": True,
        "optimization_state": True,
        "constrained_optimization_state": True,
        "vector_optimization_problem": True,
        "pytree_optimization_problem": True,
        "functional_optimization_problem": True,
    }))


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_protocols_exist()


if __name__ == "__main__":
    _run_all_tests()
