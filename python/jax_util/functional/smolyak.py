from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct


import equinox as eqx
import jax
import jax.numpy as jnp

from ..base import DEFAULT_DTYPE, Matrix, Vector
from .protocols import Function

Rule1D = Callable[[int], tuple[Vector, Vector]]


# 責務: 任意の 1 次元則出力をホスト側の NumPy ベクトルへ正規化する。
def _as_numpy_vector(values: Vector, /) -> NDArray[np.floating[Any]]:
    return np.asarray(values)


# 責務: dyadic 分数の canonical ID を 1 次元ノード列へ割り当てる。
def _dyadic_fraction_ids(
    numerators: NDArray[np.int64],
    denominator_power: int,
    /,
) -> NDArray[np.int64]:
    ids = np.empty_like(numerators, dtype=np.int64)
    denominator = np.int64(1 << denominator_power)

    zero_mask = numerators == 0
    one_mask = numerators == denominator
    inner_mask = ~(zero_mask | one_mask)

    ids[zero_mask] = 1
    ids[one_mask] = 2

    inner = numerators[inner_mask]
    if inner.size > 0:
        lowbit = inner & (-inner)
        shifts = np.log2(lowbit.astype(np.float64)).astype(np.int64)
        reduced_power = denominator_power - shifts
        reduced_numerators = inner >> shifts
        ids[inner_mask] = (np.int64(1) << reduced_power) + reduced_numerators

    return ids


# 責務: canonical ID から dyadic 分数の分子・分母を復元する。
def _dyadic_fraction_from_ids(
    ids: NDArray[np.int64],
    /,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    numerators = np.empty(ids.shape, dtype=np.float64)
    denominators = np.ones(ids.shape, dtype=np.float64)

    zero_mask = ids == 1
    one_mask = ids == 2
    inner_mask = ~(zero_mask | one_mask)

    numerators[zero_mask] = 0.0
    numerators[one_mask] = 1.0

    inner_ids = ids[inner_mask]
    if inner_ids.size > 0:
        powers = np.floor(np.log2(inner_ids.astype(np.float64))).astype(np.int64)
        numerators[inner_mask] = (inner_ids - (np.int64(1) << powers)).astype(np.float64)
        denominators[inner_mask] = np.exp2(powers.astype(np.float64))

    return numerators, denominators


# 責務: level ごとの Clenshaw-Curtis ノードへ canonical ID を割り当てる。
def clenshaw_curtis_node_ids(level: int, /) -> NDArray[np.int64]:
    if level < 1:
        raise ValueError("level must be positive.")
    if level == 1:
        return np.asarray([3], dtype=np.int64)

    denominator_power = level - 1
    numerators = np.arange(1 << denominator_power, -1, -1, dtype=np.int64)
    return _dyadic_fraction_ids(numerators, denominator_power)


# 責務: Clenshaw-Curtis の canonical ID 列からノード列を復元する。
def clenshaw_curtis_nodes_from_ids(ids: NDArray[np.int64], /) -> NDArray[np.floating[Any]]:
    numerators, denominators = _dyadic_fraction_from_ids(ids)
    return 0.5 * np.cos(np.pi * numerators / denominators)


# 責務: level ごとの一様 dyadic ノードへ canonical ID を割り当てる。
def trapezoidal_node_ids(level: int, /) -> NDArray[np.int64]:
    if level < 1:
        raise ValueError("level must be positive.")
    if level == 1:
        return np.asarray([3], dtype=np.int64)

    denominator_power = level - 1
    numerators = np.arange(0, (1 << denominator_power) + 1, dtype=np.int64)
    return _dyadic_fraction_ids(numerators, denominator_power)


# 責務: 一様 dyadic canonical ID 列からノード列を復元する。
def trapezoidal_nodes_from_ids(ids: NDArray[np.int64], /) -> NDArray[np.floating[Any]]:
    numerators, denominators = _dyadic_fraction_from_ids(ids)
    return -0.5 + numerators / denominators


# 責務: 対応する rule family が canonical ID を持つときは codec を返す。
def _rule_node_codec(
    rule: Rule1D,
    /,
) -> tuple[
    Callable[[int], NDArray[np.int64]],
    Callable[[NDArray[np.int64]], NDArray[np.floating[Any]]],
] | None:
    if rule is clenshaw_curtis_rule:
        return clenshaw_curtis_node_ids, clenshaw_curtis_nodes_from_ids
    if rule is trapezoidal_rule:
        return trapezoidal_node_ids, trapezoidal_nodes_from_ids
    return None


# 責務: DCT-I により Clenshaw-Curtis の重み列を安定に構成する。
def _clenshaw_curtis_weights(num_intervals: int, /) -> NDArray[np.floating[Any]]:
    coefficients = np.zeros(num_intervals + 1, dtype=np.float64)
    coefficients[0] = 1.0

    for mode in range(1, (num_intervals // 2) + 1):
        frequency = 2 * mode
        if frequency < num_intervals:
            coefficients[frequency] = -1.0 / (4.0 * mode * mode - 1.0)

    if num_intervals % 2 == 0:
        coefficients[num_intervals] = -1.0 / (num_intervals * num_intervals - 1.0)

    transformed = dct(coefficients, type=1)
    weights = np.empty(num_intervals + 1, dtype=np.float64)
    weights[0] = transformed[0] / num_intervals
    weights[-1] = transformed[-1] / num_intervals
    weights[1:-1] = 2.0 * transformed[1:-1] / num_intervals
    return 0.5 * weights


# 責務: level ごとの入れ子な Clenshaw-Curtis 則をホスト側 NumPy 配列で返す。
def _clenshaw_curtis_rule_numpy(
    level: int,
    /,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    if level < 1:
        raise ValueError("level must be positive.")

    if level == 1:
        return (
            np.asarray([0.0], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
        )

    num_intervals = 2 ** (level - 1)
    theta = np.pi * np.arange(num_intervals + 1, dtype=np.float64) / num_intervals
    nodes = 0.5 * np.cos(theta[::-1])
    weights = _clenshaw_curtis_weights(num_intervals)
    return nodes, weights


# 責務: level ごとの入れ子な Clenshaw-Curtis 則を [-0.5, 0.5] 上で返す。
def clenshaw_curtis_rule(level: int, /) -> tuple[Vector, Vector]:
    nodes_np, weights_np = _clenshaw_curtis_rule_numpy(level)
    nodes = jnp.asarray(nodes_np, dtype=DEFAULT_DTYPE)
    weights = jnp.asarray(weights_np, dtype=DEFAULT_DTYPE)
    return nodes, weights


# 責務: level ごとの入れ子な 1 次元台形則をホスト側 NumPy 配列で返す。
def _trapezoidal_rule_numpy(
    level: int,
    /,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    if level < 1:
        raise ValueError("level must be positive.")

    if level == 1:
        return (
            np.asarray([0.0], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
        )

    num_nodes = (2 ** (level - 1)) + 1
    spacing = 1.0 / (num_nodes - 1)
    nodes = np.linspace(-0.5, 0.5, num_nodes, dtype=np.float64)
    weights = np.full((num_nodes,), spacing, dtype=np.float64)
    weights[0] = 0.5 * spacing
    weights[-1] = 0.5 * spacing
    return nodes, weights


# 責務: level ごとの入れ子な 1 次元台形則を [-0.5, 0.5] 上で返す。
def trapezoidal_rule(level: int, /) -> tuple[Vector, Vector]:
    nodes_np, weights_np = _trapezoidal_rule_numpy(level)
    nodes = jnp.asarray(nodes_np, dtype=DEFAULT_DTYPE)
    weights = jnp.asarray(weights_np, dtype=DEFAULT_DTYPE)
    return nodes, weights


# 責務: 既知の 1 次元則ならホスト側で完結する builder を返す。
def _rule_numpy_builder(
    rule: Rule1D,
    /,
) -> Callable[[int], tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]] | None:
    if rule is clenshaw_curtis_rule:
        return _clenshaw_curtis_rule_numpy
    if rule is trapezoidal_rule:
        return _trapezoidal_rule_numpy
    return None


# 責務: 入れ子な 1 次元積分則から差分積分則を NumPy 上で構築する。
def _difference_rule_numpy(
    level: int,
    rule: Rule1D,
    /,
) -> tuple[
    NDArray[np.int64] | None,
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
]:
    codec = _rule_node_codec(rule)
    rule_numpy = _rule_numpy_builder(rule)

    if rule_numpy is not None:
        nodes, weights = rule_numpy(level)
    else:
        nodes = _as_numpy_vector(rule(level)[0])
        weights = _as_numpy_vector(rule(level)[1])

    if codec is not None:
        node_ids_fn, node_decoder = codec
        node_ids = node_ids_fn(level)
        if level == 1:
            return node_ids, nodes, weights

        previous_ids = node_ids_fn(level - 1)
        if rule_numpy is not None:
            previous_weights = rule_numpy(level - 1)[1]
        else:
            previous_weights = _as_numpy_vector(rule(level - 1)[1])

        all_ids = np.concatenate([node_ids, previous_ids], axis=0)
        all_weights = np.concatenate([weights, -previous_weights], axis=0)
        unique_ids, inverse = np.unique(all_ids, return_inverse=True)

        unique_weights = np.zeros(unique_ids.shape[0], dtype=all_weights.dtype)
        np.add.at(unique_weights, inverse, all_weights)

        mask = np.abs(unique_weights) > 1e-15
        filtered_ids = unique_ids[mask]
        filtered_weights = unique_weights[mask]
        filtered_nodes = node_decoder(filtered_ids)
        order = np.argsort(filtered_nodes)
        return filtered_ids[order], filtered_nodes[order], filtered_weights[order]

    if level == 1:
        return None, nodes, weights

    if rule_numpy is not None:
        previous_nodes, previous_weights = rule_numpy(level - 1)
    else:
        previous_nodes = _as_numpy_vector(rule(level - 1)[0])
        previous_weights = _as_numpy_vector(rule(level - 1)[1])
    merged_nodes = np.union1d(nodes, previous_nodes)

    current_weights = np.zeros_like(merged_nodes)
    current_weights[np.searchsorted(merged_nodes, nodes)] = weights

    coarse_weights = np.zeros_like(merged_nodes)
    coarse_weights[np.searchsorted(merged_nodes, previous_nodes)] = previous_weights

    diff_weights = current_weights - coarse_weights
    mask = np.abs(diff_weights) > 1e-15
    filtered_nodes = merged_nodes[mask]
    filtered_weights = diff_weights[mask]
    return None, filtered_nodes, filtered_weights


# 責務: 入れ子な 1 次元積分則から差分積分則 Delta_level を構築する。
def difference_rule(level: int, rule: Rule1D, /) -> tuple[Vector, Vector]:
    _, diff_nodes_np, diff_weights_np = _difference_rule_numpy(level, rule)
    diff_nodes = jnp.asarray(diff_nodes_np, dtype=DEFAULT_DTYPE)
    diff_weights = jnp.asarray(diff_weights_np, dtype=DEFAULT_DTYPE)
    return diff_nodes, diff_weights


# 責務: |k|_1 <= max_norm を満たす正整数 multi-index を exact-size 配列で列挙する。
def multi_indices(dimension: int, max_norm: int, /) -> NDArray[np.int_]:
    if dimension < 1:
        raise ValueError("dimension must be positive.")
    if max_norm < dimension:
        return np.empty((0, dimension), dtype=np.int_)

    num_indices = comb(max_norm, dimension)
    cumulative_sums = np.empty((num_indices, dimension), dtype=np.int_)

    for row, selected_sums in enumerate(combinations(range(1, max_norm + 1), dimension)):
        cumulative_sums[row, :] = selected_sums

    indices = np.empty_like(cumulative_sums)
    indices[:, 0] = cumulative_sums[:, 0]
    if dimension > 1:
        indices[:, 1:] = np.diff(cumulative_sums, axis=1)

    return indices


# 責務: 1 つの tensor 積差分則をホスト側で点列と重み列へ展開する。
def _tensor_difference_rule_numpy(
    nodes_by_axis: list[NDArray[np.floating[Any]]],
    weights_by_axis: list[NDArray[np.floating[Any]]],
    /,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    point_mesh = np.meshgrid(*nodes_by_axis, indexing="ij")
    points = np.stack(point_mesh, axis=-1).reshape(-1, len(nodes_by_axis))

    weight_mesh = np.meshgrid(*weights_by_axis, indexing="ij")
    weights = np.prod(np.stack(weight_mesh, axis=0), axis=0).reshape(-1)

    mask = np.abs(weights) > 1e-15
    return points[mask], weights[mask]


# 責務: tensor 積差分則を canonical ID 行列と重み列へ展開する。
def _tensor_difference_rule_ids_numpy(
    node_ids_by_axis: list[NDArray[np.int64]],
    weights_by_axis: list[NDArray[np.floating[Any]]],
    /,
) -> tuple[NDArray[np.int64], NDArray[np.floating[Any]]]:
    id_mesh = np.meshgrid(*node_ids_by_axis, indexing="ij")
    point_ids = np.stack(id_mesh, axis=-1).reshape(-1, len(node_ids_by_axis))

    weight_mesh = np.meshgrid(*weights_by_axis, indexing="ij")
    weights = np.prod(np.stack(weight_mesh, axis=0), axis=0).reshape(-1)

    mask = np.abs(weights) > 1e-15
    return point_ids[mask], weights[mask]


# 責務: 点列を辞書順に並べる順序を返す。
def _lexsort_points(points: NDArray[np.floating[Any]], /) -> NDArray[np.intp]:
    return np.lexsort(points.T[::-1])


# 責務: Smolyak 組合せから格子点と重みを明示的に構成する。
def smolyak_grid(
    dimension: int,
    level: int,
    *,
    rule: Rule1D = clenshaw_curtis_rule,
) -> tuple[Matrix, Vector]:
    if dimension < 1:
        raise ValueError("dimension must be positive.")
    if level < 1:
        raise ValueError("level must be positive.")

    max_norm = level + dimension - 1
    rule_cache: dict[
        int,
        tuple[
            NDArray[np.int64] | None,
            NDArray[np.floating[Any]],
            NDArray[np.floating[Any]],
        ],
    ] = {}
    codec = _rule_node_codec(rule)
    node_decoder: Callable[[NDArray[np.int64]], NDArray[np.floating[Any]]] | None = None
    term_ids: list[NDArray[np.int64]] = []
    term_points: list[NDArray[np.floating[Any]]] = []
    term_weights: list[NDArray[np.floating[Any]]] = []
    if codec is not None:
        _, node_decoder = codec

    for index in multi_indices(dimension, max_norm):
        nodes_by_axis: list[NDArray[np.floating[Any]]] = []
        weights_by_axis: list[NDArray[np.floating[Any]]] = []
        node_ids_by_axis: list[NDArray[np.int64]] = []
        for axis_level in index:
            if axis_level not in rule_cache:
                rule_cache[axis_level] = _difference_rule_numpy(axis_level, rule)
            node_ids, nodes, weights = rule_cache[axis_level]
            nodes_by_axis.append(nodes)
            weights_by_axis.append(weights)
            if codec is not None:
                assert node_ids is not None
                node_ids_by_axis.append(node_ids)
        if codec is not None:
            point_ids_np, weights_np = _tensor_difference_rule_ids_numpy(node_ids_by_axis, weights_by_axis)
            term_ids.append(point_ids_np)
        else:
            points_np, weights_np = _tensor_difference_rule_numpy(nodes_by_axis, weights_by_axis)
            term_points.append(points_np)
        term_weights.append(weights_np)

    all_weights = np.concatenate(term_weights, axis=0)
    if codec is not None:
        assert node_decoder is not None
        all_ids = np.concatenate(term_ids, axis=0)
        unique_ids, inverse = np.unique(all_ids, axis=0, return_inverse=True)
        unique_weights = np.zeros(unique_ids.shape[0], dtype=all_weights.dtype)
        np.add.at(unique_weights, inverse, all_weights)

        mask = np.abs(unique_weights) > 1e-15
        filtered_ids = unique_ids[mask]
        filtered_weights = unique_weights[mask]
        filtered_points = np.stack(
            [node_decoder(filtered_ids[:, axis]) for axis in range(dimension)],
            axis=1,
        )
    else:
        all_points = np.concatenate(term_points, axis=0)
        unique_points, inverse = np.unique(all_points, axis=0, return_inverse=True)
        unique_weights = np.zeros(unique_points.shape[0], dtype=all_weights.dtype)
        np.add.at(unique_weights, inverse, all_weights)

        mask = np.abs(unique_weights) > 1e-15
        filtered_points = unique_points[mask]
        filtered_weights = unique_weights[mask]

    order = _lexsort_points(filtered_points)
    filtered_points = filtered_points[order]
    filtered_weights = filtered_weights[order]

    points = jnp.asarray(filtered_points.T, dtype=DEFAULT_DTYPE)
    weights = jnp.asarray(filtered_weights, dtype=DEFAULT_DTYPE)
    return points, weights


# 責務: Smolyak 格子上の重み付き和としてベクトル値関数を積分する。
@eqx.filter_jit
def smolyak_integral(
    f: Function,
    points: Matrix,
    weights: Vector,
    /,
) -> Vector:
    values = jax.vmap(f, in_axes=1, out_axes=0)(points)
    return jnp.tensordot(weights, values, axes=(0, 0))


class SmolyakIntegrator(eqx.Module):
    dimension: int
    level: int
    rule: Rule1D = eqx.field(static=True)
    points: Matrix
    weights: Vector

    def __init__(
        self,
        dimension: int,
        level: int,
        rule: Rule1D = clenshaw_curtis_rule,
    ):
        self.dimension = dimension
        self.level = level
        self.rule = rule
        self.points, self.weights = smolyak_grid(dimension, level, rule=rule)

    def __call__(self, f: Function, /) -> Vector:
        return smolyak_integral(f, self.points, self.weights)

    # 責務: より細かい疎格子を使う次レベルの積分器を返す。
    def refine(self) -> "SmolyakIntegrator":
        return SmolyakIntegrator(
            dimension=self.dimension,
            level=self.level + 1,
            rule=self.rule,
        )


__all__ = [
    "SmolyakIntegrator",
    "Rule1D",
    "clenshaw_curtis_rule",
    "clenshaw_curtis_node_ids",
    "difference_rule",
    "multi_indices",
    "smolyak_grid",
    "smolyak_integral",
    "trapezoidal_node_ids",
    "trapezoidal_rule",
]
