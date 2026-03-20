"""基盤層パッケージ：型定義・Protocol・演算子の共通基盤。

このパッケージは、全プロジェクトで使用される基本型、Protocol、
線形・非線形演算子を提供します。

公開インターフェース:
    型定義: Scalar, Vector, Matrix, Boolean, Integer
    Protocol: LinearOperator, Operator, SolverLike
    定数: DEFAULT_DTYPE, EPS, DEBUG, ZERO, ONE, HALF, WEAK_EPS
    関数: adjoint, linearize, ScalarFn

参照資料:
    - documents/design/protocols.md（Protocol 設計）
    - documents/coding-conventions-python.md（型規約）
"""

from __future__ import annotations


from .protocols import *
from ._env_value import *  
from .linearoperator import *
from .nonlinearoperator import *


__all__ = [
    "DEFAULT_DTYPE",
    "EPS",
    "DEBUG",
    "ZERO",
    "ONE",
    "HALF",
    "WEAK_EPS",
    "AVOID_ZERO_DIV",
    "LinOp",
    "Scalar",
    "Vector",
    "Matrix",
    "Boolean",
    "Integer",
    "LinearOperator",
    "Operator",
    "SolverLike",
    "adjoint",
    "linearize",
    "ScalarFn",
    "VectorFn",
    "OptimizationProblem",
    "ConstrainedOptimizationProblem",
    "OptimizationState",
    "ConstrainedOptimizationState",
]
