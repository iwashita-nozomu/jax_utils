"""ニューラルネットワークのヤコビアン（入力感度）計算モジュール。

各出力 $y_i$ が各入力 $x_j$ にどれだけ依存しているかを自動微分で求めます。

規約:
  - 入力 x は Matrix 型 (input_dim, batch_size)。batch 軸は axis=-1。
  - ヤコビアン行列は (batch_size, output_dim, input_dim) を返します。
  - 感度（入力の重要度）は ヤコビアン列の L2 ノルムで定義します。
"""
from __future__ import annotations

import jax
from jax import numpy as jnp

from ..base import Matrix, Vector
from .neuralnetwork import NeuralNetwork


def compute_jacobian(network: NeuralNetwork, x: Matrix) -> Matrix:
    """各バッチ要素に対するヤコビアン行列 $\\partial y / \\partial x$ を計算します。

    ヤコビアン行列の $(i, j)$ 要素は $\\partial y_i / \\partial x_j$ です。
    これにより「出力 $y_i$ が入力 $x_j$ をどれだけ見ているか」が分かります。

    Parameters
    ----------
    network : NeuralNetwork
        対象のニューラルネットワーク。
    x : Matrix
        入力行列。形状 (input_dim, batch_size)。

    Returns
    -------
    Matrix
        ヤコビアン行列群。形状 (batch_size, output_dim, input_dim)。
    """

    def forward_single(x_single: Vector) -> Vector:
        # x_single: (input_dim,) を受け取り、バッチ次元を補って forward を実行する補助関数。
        # ヤコビアンを 1 サンプルずつ計算するため vmap と組み合わせて使用します。
        y = network(x_single[:, None])  # (output_dim, 1)
        return y[:, 0]  # (output_dim,)

    # x: (input_dim, batch_size) → (batch_size, input_dim) に転置して vmap 適用
    x_t = x.T
    jacobians: Matrix = jax.vmap(jax.jacobian(forward_single))(x_t)
    return jacobians  # (batch_size, output_dim, input_dim)


def input_sensitivity(network: NeuralNetwork, x: Matrix) -> Matrix:
    """各入力要素の出力への感度（ヤコビアン列の L2 ノルム）を計算します。

    感度 $s_j = \\| \\partial y / \\partial x_j \\|_2$ は、
    入力 $x_j$ が出力全体に与える影響の大きさを表します。

    Parameters
    ----------
    network : NeuralNetwork
        対象のニューラルネットワーク。
    x : Matrix
        入力行列。形状 (input_dim, batch_size)。

    Returns
    -------
    Matrix
        感度行列。形状 (batch_size, input_dim)。
    """
    # ヤコビアン: (batch_size, output_dim, input_dim)
    J = compute_jacobian(network, x)

    # 出力次元方向に L2 ノルムをとり、各入力要素の感度を計算
    sensitivity: Matrix = jnp.linalg.norm(J, axis=1)  # (batch_size, input_dim)
    return sensitivity


__all__ = [
    "compute_jacobian",
    "input_sensitivity",
]
