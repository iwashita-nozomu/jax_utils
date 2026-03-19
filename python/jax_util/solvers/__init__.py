from __future__ import annotations

# アルゴリズム実装をまとめたサブパッケージ。

from . import _check_mv_operator

# from . import _fgmres
from . import _minres
from . import _test_jax
from . import kkt_solver
from . import lobpcg
from . import matrix_util
from . import pcg

# `solvers` は数値ソルバ群のサブパッケージです。
# 最適化アルゴリズム（例: PDIPM）は `jax_util.optimizers` に配置し、
# `solvers` からは import しません。
#
# 目的:
# - 依存の向きを `base -> solvers -> optimizers` に固定する。
# - サブパッケージ間の循環 import を防ぎ、型チェックとテスト収集を安定させる。

__all__ = [
    "_check_mv_operator",
    # "_fgmres",
    "_minres",
    "_test_jax",
    "kkt_solver",
    "lobpcg",
    "matrix_util",
    "pcg",
]
