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
from jax.typing import DTypeLike

from ..base import DEFAULT_DTYPE, Matrix, Vector
from .protocols import Function

Rule1D = Callable[[int], tuple[Vector, Vector]]


# 責務: 任意の 1 次元則出力をホスト側の NumPy ベクトルへ正規化する。
def _as_numpy_vector(values: Vector, /) -> NDArray[np.floating[Any]]:
    return np.asarray(values)


# 責務: 正の整数値の上限に対して最小限の unsigned dtype を選ぶ。
def _compact_unsigned_dtype(max_value: int, /) -> np.dtype[np.unsignedinteger[Any]]:
    if max_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if max_value <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


# 責務: dyadic 分数の canonical ID を 1 次元ノード列へ割り当てる。
def _dyadic_fraction_ids(
    numerators: NDArray[np.int64],
    denominator_power: int,
    /,
) -> NDArray[np.unsignedinteger[Any]]:
    max_id = (1 << (denominator_power + 1)) - 1
    id_dtype = _compact_unsigned_dtype(max_id)
    ids = np.empty(numerators.shape, dtype=id_dtype)
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
        ids[inner_mask] = ((np.int64(1) << reduced_power) + reduced_numerators).astype(id_dtype)

    return ids


# 責務: canonical ID から dyadic 分数の分子・分母を復元する。
def _dyadic_fraction_from_ids(
    ids: NDArray[np.integer[Any]],
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
        inner_ids_int64 = inner_ids.astype(np.int64, copy=False)
        powers = np.floor(np.log2(inner_ids_int64.astype(np.float64))).astype(np.int64)
        numerators[inner_mask] = (inner_ids_int64 - (np.int64(1) << powers)).astype(np.float64)
        denominators[inner_mask] = np.exp2(powers.astype(np.float64))

    return numerators, denominators


# 責務: level ごとの Clenshaw-Curtis ノードへ canonical ID を割り当てる。
def clenshaw_curtis_node_ids(level: int, /) -> NDArray[np.unsignedinteger[Any]]:
    if level < 1:
        raise ValueError("level must be positive.")
    if level == 1:
        return np.asarray([3], dtype=np.uint8)

    denominator_power = level - 1
    numerators = np.arange(1 << denominator_power, -1, -1, dtype=np.int64)
    return _dyadic_fraction_ids(numerators, denominator_power)


# 責務: Clenshaw-Curtis の canonical ID 列からノード列を復元する。
def clenshaw_curtis_nodes_from_ids(ids: NDArray[np.integer[Any]], /) -> NDArray[np.floating[Any]]:
    numerators, denominators = _dyadic_fraction_from_ids(ids)
    return 0.5 * np.cos(np.pi * numerators / denominators)


# 責務: level ごとの一様 dyadic ノードへ canonical ID を割り当てる。
def trapezoidal_node_ids(level: int, /) -> NDArray[np.unsignedinteger[Any]]:
    if level < 1:
        raise ValueError("level must be positive.")
    if level == 1:
        return np.asarray([3], dtype=np.uint8)

    denominator_power = level - 1
    numerators = np.arange(0, (1 << denominator_power) + 1, dtype=np.int64)
    return _dyadic_fraction_ids(numerators, denominator_power)


# 責務: 一様 dyadic canonical ID 列からノード列を復元する。
def trapezoidal_nodes_from_ids(ids: NDArray[np.integer[Any]], /) -> NDArray[np.floating[Any]]:
    numerators, denominators = _dyadic_fraction_from_ids(ids)
    return -0.5 + numerators / denominators


# 責務: 対応する rule family が canonical ID を持つときは codec を返す。
def _rule_node_codec(
    rule: Rule1D,
    /,
) -> tuple[
    Callable[[int], NDArray[np.integer[Any]]],
    Callable[[NDArray[np.integer[Any]]], NDArray[np.floating[Any]]],
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
    NDArray[np.integer[Any]] | None,
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
def difference_rule(
    level: int,
    rule: Rule1D = clenshaw_curtis_rule,
    /,
) -> tuple[Vector, Vector]:
    _, diff_nodes_np, diff_weights_np = _difference_rule_numpy(level, rule)
    diff_nodes = jnp.asarray(diff_nodes_np, dtype=DEFAULT_DTYPE)
    diff_weights = jnp.asarray(diff_weights_np, dtype=DEFAULT_DTYPE)
    return diff_nodes, diff_weights


# 責務: |k|_1 <= max_norm を満たす正整数 multi-index を exact-size 配列で列挙する。
def multi_indices(dimension: int, max_norm: int, /) -> NDArray[np.unsignedinteger[Any]]:
    if dimension < 1:
        raise ValueError("dimension must be positive.")
    index_dtype = _compact_unsigned_dtype(max(dimension, max_norm, 1))
    if max_norm < dimension:
        return np.empty((0, dimension), dtype=index_dtype)

    # `k_1 + ... + k_d <= max_norm, k_i >= 1` を
    # 累積和の組合せとして数え上げると、行数が `comb(max_norm, dimension)` で決まる。
    num_indices = comb(max_norm, dimension)
    indices = np.empty((num_indices, dimension), dtype=index_dtype)

    for row, selected_sums in enumerate(combinations(range(1, max_norm + 1), dimension)):
        previous_sum = 0
        for column, current_sum in enumerate(selected_sums):
            indices[row, column] = current_sum - previous_sum
            previous_sum = current_sum

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
    node_ids_by_axis: list[NDArray[np.integer[Any]]],
    weights_by_axis: list[NDArray[np.floating[Any]]],
    /,
) -> tuple[NDArray[np.integer[Any]], NDArray[np.floating[Any]]]:
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
            NDArray[np.integer[Any]] | None,
            NDArray[np.floating[Any]],
            NDArray[np.floating[Any]],
        ],
    ] = {}
    codec = _rule_node_codec(rule)
    node_decoder: Callable[[NDArray[np.integer[Any]]], NDArray[np.floating[Any]]] | None = None
    term_ids: list[NDArray[np.integer[Any]]] = []
    term_points: list[NDArray[np.floating[Any]]] = []
    term_weights: list[NDArray[np.floating[Any]]] = []
    if codec is not None:
        _, node_decoder = codec

    for index in multi_indices(dimension, max_norm):
        nodes_by_axis: list[NDArray[np.floating[Any]]] = []
        weights_by_axis: list[NDArray[np.floating[Any]]] = []
        node_ids_by_axis: list[NDArray[np.integer[Any]]] = []
        for axis_level in index:
            axis_level_int = int(axis_level)
            if axis_level_int not in rule_cache:
                rule_cache[axis_level_int] = _difference_rule_numpy(axis_level_int, rule)
            node_ids, nodes, weights = rule_cache[axis_level_int]
            nodes_by_axis.append(nodes)
            weights_by_axis.append(weights)
            if codec is not None:
                assert node_ids is not None
                node_ids_by_axis.append(node_ids)
        if codec is not None:
            # canonical ID を経由すると、同一点の統合を浮動小数比較ではなく整数比較で行える。
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
        # 各 tensor term の寄与を一度まとめ、同一点に落ちる重みをここで相殺する。
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
        # codec がない rule family では、座標そのものを key にして統合する。
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
    values = jax.vmap(f, in_axes=-1, out_axes=-1)(points)
    return jnp.tensordot(values, weights, axes=(-1, 0))


# 責務: int64 に収まる範囲で整数列の積を計算する。
def _safe_product_int64(values: NDArray[np.integer[Any]], /) -> int:
    limit = np.iinfo(np.int64).max
    product = 1
    for value in values:
        product *= int(value)
        if product > limit:
            raise OverflowError("term point count exceeds int64.")
    return product


# 責務: int64 に収まる範囲で整数列の総和を計算する。
def _safe_sum_int64(values: NDArray[np.int64], /) -> int:
    limit = np.iinfo(np.int64).max
    total = 0
    for value in values:
        total += int(value)
        if total > limit:
            raise OverflowError("total point count exceeds int64.")
    return total


# 責務: Smolyak level と次元から必要な最大 1 次元差分則 level を返す。
def _max_difference_rule_level(dimension: int, level: int, /) -> int:
    return dimension + level - 1


# 責務: level ごとの差分則を flat storage と offset/length へまとめる。
def _difference_rule_storage_numpy(
    max_level: int,
    /,
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    nodes_by_level: list[NDArray[np.floating[Any]]] = []
    weights_by_level: list[NDArray[np.floating[Any]]] = []
    lengths = np.empty((max_level,), dtype=np.int64)

    for current_level in range(1, max_level + 1):
        _, nodes_np, weights_np = _difference_rule_numpy(current_level, clenshaw_curtis_rule)
        nodes_by_level.append(nodes_np)
        weights_by_level.append(weights_np)
        lengths[current_level - 1] = nodes_np.shape[0]

    offsets = np.empty((max_level,), dtype=np.int64)
    total_length = 0
    for current_level, length in enumerate(lengths):
        offsets[current_level] = total_length
        total_length += int(length)

    nodes_storage = np.empty((total_length,), dtype=np.float64)
    weights_storage = np.empty((total_length,), dtype=np.float64)
    for current_level, (nodes_np, weights_np) in enumerate(zip(nodes_by_level, weights_by_level, strict=True)):
        start = int(offsets[current_level])
        stop = start + int(lengths[current_level])
        nodes_storage[start:stop] = nodes_np
        weights_storage[start:stop] = weights_np

    return nodes_storage, weights_storage, offsets, lengths


# 責務: 最大 level までの 1 次元差分則 storage を初期化する。
def _initialize_rule_storage(
    dimension: int,
    prepared_level: int,
    dtype: DTypeLike,
    /,
) -> tuple[
    Vector,
    Vector,
    jax.Array,
    jax.Array,
]:
    max_rule_level = _max_difference_rule_level(dimension, prepared_level)
    rule_nodes_np, rule_weights_np, rule_offsets_np, rule_lengths_np = _difference_rule_storage_numpy(max_rule_level)
    return (
        jnp.asarray(rule_nodes_np, dtype=dtype),
        jnp.asarray(rule_weights_np, dtype=dtype),
        jnp.asarray(rule_offsets_np, dtype=jnp.int64),
        jnp.asarray(rule_lengths_np, dtype=jnp.int64),
    )


# 責務: active level に対応する term index と評価点数メタデータを初期化する。
def _initialize_term_plan(
    dimension: int,
    level: int,
    rule_lengths: jax.Array,
    /,
) -> tuple[
    jax.Array,
    jax.Array,
    int,
    int,
]:
    if dimension < 1:
        raise ValueError("dimension must be positive.")
    if level < 1:
        raise ValueError("level must be positive.")

    max_norm = _max_difference_rule_level(dimension, level)
    term_levels_np = multi_indices(dimension, max_norm).astype(np.int32, copy=False)
    host_rule_lengths = np.asarray(rule_lengths, dtype=np.int64)
    term_num_points_np = np.empty((term_levels_np.shape[0],), dtype=np.int64)
    for term_index, levels in enumerate(term_levels_np):
        axis_lengths = host_rule_lengths[levels.astype(np.int64) - 1]
        term_num_points_np[term_index] = _safe_product_int64(axis_lengths)

    num_terms = int(term_levels_np.shape[0])
    num_evaluation_points = _safe_sum_int64(term_num_points_np)

    return (
        jnp.asarray(term_levels_np, dtype=jnp.int32),
        jnp.asarray(term_num_points_np, dtype=jnp.int64),
        num_terms,
        num_evaluation_points,
    )


# 責務: 1 次元差分則 storage と term plan の保持量を byte 単位で見積もる。
def _storage_bytes(
    rule_nodes: jax.Array,
    rule_weights: jax.Array,
    rule_offsets: jax.Array,
    rule_lengths: jax.Array,
    term_levels: jax.Array,
    term_num_points: jax.Array,
    /,
) -> int:
    return (
        int(rule_nodes.nbytes)
        + int(rule_weights.nbytes)
        + int(rule_offsets.nbytes)
        + int(rule_lengths.nbytes)
        + int(term_levels.nbytes)
        + int(term_num_points.nbytes)
    )


# 責務: 軸レベルから 1 次元差分則の flat storage 上の開始位置と長さを取得する。
def _rule_segment(
    axis_levels: jax.Array,
    axis: int,
    rule_offsets: jax.Array,
    rule_lengths: jax.Array,
    /,
) -> tuple[jax.Array, jax.Array]:
    level_index = axis_levels[axis] - 1
    rule_offset = jnp.take(rule_offsets, level_index, mode="clip")
    rule_length = jnp.take(rule_lengths, level_index, mode="clip")
    return rule_offset, rule_length


# 責務: 固定次元ベクトルの指定軸だけを新しい値へ置き換える。
def _replace_axis_value(
    point: Vector,
    axis: int,
    value: jax.Array,
    /,
) -> Vector:
    value_vector = jnp.reshape(jnp.asarray(value, dtype=point.dtype), (1,))
    return jnp.concatenate((point[:axis], value_vector, point[axis + 1 :]), axis=0)


# 責務: 固定した prefix のもとで最後の 1 軸だけを batched 評価して積分へ加算する。
def _integrate_last_axis(
    f: Function,
    axis_levels: jax.Array,
    axis: int,
    prefix_point: Vector,
    prefix_weight: jax.Array,
    acc: Vector,
    rule_nodes: Vector,
    rule_weights: Vector,
    rule_offsets: jax.Array,
    rule_lengths: jax.Array,
    *,
    chunk_size: int,
) -> Vector:
    rule_offset, rule_length = _rule_segment(axis_levels, axis, rule_offsets, rule_lengths)
    flat_chunk_indices = jnp.arange(chunk_size, dtype=jnp.int64)

    def chunk_body(chunk_index: int, leaf_acc: Vector) -> Vector:
        start = jnp.asarray(chunk_index, dtype=jnp.int64) * chunk_size
        local_indices = start + flat_chunk_indices
        valid_mask = local_indices < rule_length
        safe_local_indices = jnp.where(valid_mask, local_indices, 0)
        flat_indices = rule_offset + safe_local_indices
        axis_nodes = jnp.take(rule_nodes, flat_indices, mode="clip")
        axis_weights = jnp.take(rule_weights, flat_indices, mode="clip")

        def inject_axis_node(axis_node: jax.Array) -> Vector:
            return _replace_axis_value(prefix_point, axis, axis_node)

        points = jax.vmap(
            inject_axis_node,
            in_axes=0,
            out_axes=-1,
        )(axis_nodes)
        values = jax.vmap(f, in_axes=-1, out_axes=-1)(points)
        masked_weights = jnp.where(valid_mask, prefix_weight * axis_weights, 0)
        return leaf_acc + jnp.tensordot(values, masked_weights, axes=(-1, 0))

    num_chunks = jnp.floor_divide(rule_length + chunk_size - 1, chunk_size)
    return jax.lax.fori_loop(0, num_chunks, chunk_body, acc)


# 責務: 固定次元の Python 再帰で prefix を伸ばしながら term 積分を評価する。
def _integrate_term_recursive(
    f: Function,
    axis_levels: jax.Array,
    axis: int,
    prefix_point: Vector,
    prefix_weight: jax.Array,
    acc: Vector,
    rule_nodes: Vector,
    rule_weights: Vector,
    rule_offsets: jax.Array,
    rule_lengths: jax.Array,
    *,
    chunk_size: int,
) -> Vector:
    if axis == prefix_point.shape[0] - 1:
        return _integrate_last_axis(
            f,
            axis_levels,
            axis,
            prefix_point,
            prefix_weight,
            acc,
            rule_nodes,
            rule_weights,
            rule_offsets,
            rule_lengths,
            chunk_size=chunk_size,
        )

    rule_offset, rule_length = _rule_segment(axis_levels, axis, rule_offsets, rule_lengths)

    def axis_body(local_index: int, loop_acc: Vector) -> Vector:
        flat_index = rule_offset + jnp.asarray(local_index, dtype=jnp.int64)
        axis_node = jnp.take(rule_nodes, flat_index, mode="clip")
        axis_weight = jnp.take(rule_weights, flat_index, mode="clip")
        next_point = _replace_axis_value(prefix_point, axis, axis_node)
        next_weight = prefix_weight * axis_weight
        return _integrate_term_recursive(
            f,
            axis_levels,
            axis + 1,
            next_point,
            next_weight,
            loop_acc,
            rule_nodes,
            rule_weights,
            rule_offsets,
            rule_lengths,
            chunk_size=chunk_size,
        )

    return jax.lax.fori_loop(0, rule_length, axis_body, acc)


# 責務: 差分則 storage と term index table に基づいて Smolyak 積分を逐次実行する。
def _smolyak_plan_integral(
    f: Function,
    term_levels: jax.Array,
    term_num_points: jax.Array,
    rule_nodes: Vector,
    rule_weights: Vector,
    rule_offsets: jax.Array,
    rule_lengths: jax.Array,
    *,
    chunk_size: int,
) -> Vector:
    del term_num_points
    zero_point = jnp.zeros((term_levels.shape[1],), dtype=rule_nodes.dtype)
    initial_value = jnp.zeros_like(f(zero_point))

    def term_body(term_index: int, acc: Vector) -> Vector:
        axis_levels = term_levels[term_index]
        return _integrate_term_recursive(
            f,
            axis_levels,
            0,
            zero_point,
            jnp.asarray(1.0, dtype=rule_weights.dtype),
            acc,
            rule_nodes,
            rule_weights,
            rule_offsets,
            rule_lengths,
            chunk_size=chunk_size,
        )

    return jax.lax.fori_loop(0, term_levels.shape[0], term_body, initial_value)


class SmolyakIntegrator(eqx.Module):
    dimension: int
    level: int
    prepared_level: int
    rule: Rule1D = eqx.field(static=True)
    dtype: DTypeLike = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    rule_nodes: Vector
    rule_weights: Vector
    rule_offsets: jax.Array
    rule_lengths: jax.Array
    term_levels: jax.Array
    term_num_points: jax.Array
    num_terms: int
    num_evaluation_points: int
    storage_bytes: int
    points: Matrix | None
    weights: Vector | None

    def __init__(
        self,
        dimension: int,
        level: int,
        rule: Rule1D = clenshaw_curtis_rule,
        dtype: DTypeLike = DEFAULT_DTYPE,
        *,
        prepared_level: int | None = None,
        chunk_size: int = 16384,
        _rule_storage: tuple[Vector, Vector, jax.Array, jax.Array] | None = None,
    ):
        if dimension < 1:
            raise ValueError("dimension must be positive.")
        if level < 1:
            raise ValueError("level must be positive.")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        resolved_prepared_level = level if prepared_level is None else prepared_level
        if resolved_prepared_level < level:
            raise ValueError("prepared_level must be greater than or equal to level.")

        self.dimension = dimension
        self.level = level
        self.prepared_level = resolved_prepared_level
        self.rule = rule
        self.dtype = dtype
        self.chunk_size = chunk_size

        if rule is clenshaw_curtis_rule:
            self.points = None
            self.weights = None
            if _rule_storage is None:
                (
                    self.rule_nodes,
                    self.rule_weights,
                    self.rule_offsets,
                    self.rule_lengths,
                ) = _initialize_rule_storage(dimension, resolved_prepared_level, dtype)
            else:
                (
                    self.rule_nodes,
                    self.rule_weights,
                    self.rule_offsets,
                    self.rule_lengths,
                ) = _rule_storage
            (
                self.term_levels,
                self.term_num_points,
                self.num_terms,
                self.num_evaluation_points,
            ) = _initialize_term_plan(dimension, level, self.rule_lengths)
            self.storage_bytes = _storage_bytes(
                self.rule_nodes,
                self.rule_weights,
                self.rule_offsets,
                self.rule_lengths,
                self.term_levels,
                self.term_num_points,
            )
            return

        self.points, self.weights = smolyak_grid(dimension, level, rule=rule)
        self.points = jnp.asarray(self.points, dtype=dtype)
        self.weights = jnp.asarray(self.weights, dtype=dtype)
        self.rule_nodes = jnp.empty((0,), dtype=dtype)
        self.rule_weights = jnp.empty((0,), dtype=dtype)
        self.rule_offsets = jnp.empty((0,), dtype=jnp.int64)
        self.rule_lengths = jnp.empty((0,), dtype=jnp.int64)
        self.term_levels = jnp.empty((0, dimension), dtype=jnp.int32)
        self.term_num_points = jnp.empty((0,), dtype=jnp.int64)
        self.num_terms = 0
        self.num_evaluation_points = int(self.points.shape[1])
        self.storage_bytes = int(self.points.nbytes) + int(self.weights.nbytes)

    def integrate(self, f: Function, /) -> Vector:
        if self.points is not None and self.weights is not None:
            return smolyak_integral(f, self.points, self.weights)
        return _smolyak_plan_integral(
            f,
            self.term_levels,
            self.term_num_points,
            self.rule_nodes,
            self.rule_weights,
            self.rule_offsets,
            self.rule_lengths,
            chunk_size=self.chunk_size,
        )

    def __call__(self, f: Function, /) -> Vector:
        return self.integrate(f)

    # 責務: より細かい疎格子を使う次レベルの積分器を返す。
    def refine(self) -> "SmolyakIntegrator":
        next_level = self.level + 1
        if self.rule is clenshaw_curtis_rule:
            if next_level <= self.prepared_level:
                return SmolyakIntegrator(
                    dimension=self.dimension,
                    level=next_level,
                    rule=self.rule,
                    dtype=self.dtype,
                    prepared_level=self.prepared_level,
                    chunk_size=self.chunk_size,
                    _rule_storage=(
                        self.rule_nodes,
                        self.rule_weights,
                        self.rule_offsets,
                        self.rule_lengths,
                    ),
                )
            return SmolyakIntegrator(
                dimension=self.dimension,
                level=next_level,
                rule=self.rule,
                dtype=self.dtype,
                prepared_level=next_level,
                chunk_size=self.chunk_size,
            )
        return SmolyakIntegrator(
            dimension=self.dimension,
            level=next_level,
            rule=self.rule,
            dtype=self.dtype,
            prepared_level=max(self.prepared_level, next_level),
            chunk_size=self.chunk_size,
        )


# 責務: plan 化された Smolyak 積分器を初期化して返す。
def initialize_smolyak_integrator(
    dimension: int,
    level: int,
    *,
    rule: Rule1D = clenshaw_curtis_rule,
    prepared_level: int | None = None,
    dtype: DTypeLike = DEFAULT_DTYPE,
    chunk_size: int = 16384,
) -> SmolyakIntegrator:
    return SmolyakIntegrator(
        dimension=dimension,
        level=level,
        rule=rule,
        dtype=dtype,
        prepared_level=prepared_level,
        chunk_size=chunk_size,
    )


__all__ = [
    "SmolyakIntegrator",
    "Rule1D",
    "clenshaw_curtis_node_ids",
    "clenshaw_curtis_rule",
    "difference_rule",
    "initialize_smolyak_integrator",
    "multi_indices",
    "smolyak_grid",
    "smolyak_integral",
    "trapezoidal_node_ids",
    "trapezoidal_rule",
]
