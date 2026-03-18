from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct


import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from ..base import DEFAULT_DTYPE, Vector
from .protocols import Function


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


# 責務: 入れ子な 1 次元積分則から差分積分則を NumPy 上で構築する。
def _difference_rule_numpy(
    level: int,
    /,
) -> tuple[
    NDArray[np.integer[Any]] | None,
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
]:
    nodes, weights = _clenshaw_curtis_rule_numpy(level)
    node_ids = clenshaw_curtis_node_ids(level)
    if level == 1:
        return node_ids, nodes, weights

    previous_ids = clenshaw_curtis_node_ids(level - 1)
    previous_weights = _clenshaw_curtis_rule_numpy(level - 1)[1]

    all_ids = np.concatenate([node_ids, previous_ids], axis=0)
    all_weights = np.concatenate([weights, -previous_weights], axis=0)
    unique_ids, inverse = np.unique(all_ids, return_inverse=True)

    unique_weights = np.zeros(unique_ids.shape[0], dtype=all_weights.dtype)
    np.add.at(unique_weights, inverse, all_weights)

    mask = np.abs(unique_weights) > 1e-15
    filtered_ids = unique_ids[mask]
    filtered_weights = unique_weights[mask]
    filtered_nodes = clenshaw_curtis_nodes_from_ids(filtered_ids)
    order = np.argsort(filtered_nodes)
    return filtered_ids[order], filtered_nodes[order], filtered_weights[order]


# 責務: Clenshaw-Curtis の差分積分則 Delta_level を構築する。
def difference_rule(level: int, /) -> tuple[Vector, Vector]:
    _, diff_nodes_np, diff_weights_np = _difference_rule_numpy(level)
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

    num_indices = comb(max_norm, dimension)
    indices = np.empty((num_indices, dimension), dtype=index_dtype)

    for row, selected_sums in enumerate(combinations(range(1, max_norm + 1), dimension)):
        previous_sum = 0
        for column, current_sum in enumerate(selected_sums):
            indices[row, column] = current_sum - previous_sum
            previous_sum = current_sum

    return indices


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
        _, nodes_np, weights_np = _difference_rule_numpy(current_level)
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

        def inject_axis_node(axis_node: Vector) -> Vector:
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

    def __init__(
        self,
        dimension: int,
        level: int,
        prepared_level: int | None = None,
        dtype: DTypeLike = DEFAULT_DTYPE,
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
        self.dtype = dtype
        self.chunk_size = chunk_size
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

    def integrate(self, f: Function, /) -> Vector:
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

    # 責務: より細かい疎格子を使う次レベルの積分器を返す。
    def refine(self) -> "SmolyakIntegrator":
        next_level = self.level + 1
        if next_level <= self.prepared_level:
            return SmolyakIntegrator(
                dimension=self.dimension,
                level=next_level,
                prepared_level=self.prepared_level,
                dtype=self.dtype,
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
            prepared_level=next_level,
            dtype=self.dtype,
            chunk_size=self.chunk_size,
        )


# 責務: plan 化された Smolyak 積分器を初期化して返す。
def initialize_smolyak_integrator(
    dimension: int,
    level: int,
    *,
    prepared_level: int | None = None,
    dtype: DTypeLike = DEFAULT_DTYPE,
    chunk_size: int = 16384,
) -> SmolyakIntegrator:
    return SmolyakIntegrator(
        dimension=dimension,
        level=level,
        prepared_level=prepared_level,
        dtype=dtype,
        chunk_size=chunk_size,
    )


__all__ = [
    "SmolyakIntegrator",
    "clenshaw_curtis_rule",
    "clenshaw_curtis_node_ids",
    "difference_rule",
    "initialize_smolyak_integrator",
    "multi_indices",
]
