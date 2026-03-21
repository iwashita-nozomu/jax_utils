"""数値ソルバーパッケージ：線形方程式・固有値問題の実装。

共役勾配法（PCG）、最小残差法（MINRES）、ブロック固有値ソルバー（LOBPCG）
など、疎行列に対する高速ソルバーアルゴリズムを提供します。

主要クラス・関数:
    PCGState, pcg_solve: 共役勾配法
    LobpcgState, lobpcg_solve: ブロック固有値ソルバー
    matrix_util: 行列ユーティリティ（変換、検証）
    kkt_solver: KKT 条件ソルバー

カテゴリ分けと実装方針は documents/design/apis/solvers.md を参照。
"""

from __future__ import annotations

# アルゴリズム実装をまとめたサブパッケージ。

from . import _check_mv_operator
# from . import _fgmres
from . import _minres
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
    "kkt_solver",
    "lobpcg",
    "matrix_util",
    "pcg",
]
