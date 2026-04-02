from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct


import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from ..base import DEFAULT_DTYPE, Vector
from .protocols import Function

_MAX_FULL_MATERIALIZED_PLAN_BYTES = 96 * 1024 * 1024
_MAX_INDEXED_MATERIALIZED_PLAN_BYTES = 256 * 1024 * 1024
_SUPPORTED_REQUESTED_MATERIALIZATION_MODES = frozenset(
    ("auto", "points", "indexed", "lazy-indexed", "batched")
)
_SUPPORTED_BATCHED_AXIS_ORDER_STRATEGIES = frozenset(("original", "length"))


# NumPy 配列を JAX 配列へ変換する補助関数。
def _to_jax_arrays(
    *arrays: NDArray[np.floating[Any]],
    dtype: DTypeLike,
) -> tuple[jax.Array, ...]:
    """1 つ以上の NumPy 配列を指定 dtype の JAX 配列へ変換する。"""
    return tuple(jnp.asarray(arr, dtype=dtype) for arr in arrays)


# 責務: 正の整数値の上限に対して最小限の unsigned dtype を選ぶ。
def _compact_unsigned_dtype(max_value: int, /) -> np.dtype[np.unsignedinteger[Any]]:
    if max_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if max_value <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


# 責務: level ごとの Clenshaw-Curtis ノードへ dyadic 分数の整数キーを割り当てる。
def _clenshaw_curtis_node_keys(level: int, /) -> NDArray[np.unsignedinteger[Any]]:
    if level < 1:
        raise ValueError("level must be positive.")
    if level == 1:
        return np.asarray([[1, 1]], dtype=np.uint8)

    denominator_power = level - 1
    numerators = np.arange(1 << denominator_power, -1, -1, dtype=np.int64)
    key_dtype = _compact_unsigned_dtype(max(1 << denominator_power, denominator_power))
    keys = np.empty(numerators.shape + (2,), dtype=key_dtype)
    denominator = np.int64(1 << denominator_power)

    reduced_numerators = keys[:, 0]
    reduced_powers = keys[:, 1]

    zero_mask = numerators == 0
    one_mask = numerators == denominator
    inner_mask = ~(zero_mask | one_mask)

    reduced_numerators[zero_mask] = 0
    reduced_powers[zero_mask] = 0
    reduced_numerators[one_mask] = 1
    reduced_powers[one_mask] = 0

    inner = numerators[inner_mask]
    if inner.size > 0:
        lowbit = inner & (-inner)
        shifts = np.log2(lowbit.astype(np.float64)).astype(np.int64)
        reduced_numerators[inner_mask] = (inner >> shifts).astype(key_dtype)
        reduced_powers[inner_mask] = (denominator_power - shifts).astype(key_dtype)

    return keys


# 責務: Clenshaw-Curtis の整数キー列からノード列を復元する。
def _clenshaw_curtis_nodes_from_keys(
    keys: NDArray[np.integer[Any]],
    /,
) -> NDArray[np.floating[Any]]:
    numerators = keys[:, 0].astype(np.float64, copy=False)
    denominator_powers = keys[:, 1].astype(np.float64, copy=False)
    denominators = np.exp2(denominator_powers)
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
    return cast(
        tuple[Vector, Vector],
        _to_jax_arrays(nodes_np, weights_np, dtype=DEFAULT_DTYPE),
    )


# 責務: 入れ子な 1 次元積分則から差分積分則を NumPy 上で構築する。
def _difference_rule_numpy(
    level: int,
    /,
) -> tuple[
    NDArray[np.floating[Any]],
    NDArray[np.floating[Any]],
]:
    nodes, weights = _clenshaw_curtis_rule_numpy(level)
    node_keys = _clenshaw_curtis_node_keys(level)
    if level == 1:
        return nodes, weights

    previous_keys = _clenshaw_curtis_node_keys(level - 1)
    previous_weights = _clenshaw_curtis_rule_numpy(level - 1)[1]

    all_keys = np.concatenate([node_keys, previous_keys], axis=0)
    all_weights = np.concatenate([weights, -previous_weights], axis=0)
    unique_keys, inverse = np.unique(all_keys, axis=0, return_inverse=True)

    unique_weights = np.zeros(unique_keys.shape[0], dtype=all_weights.dtype)
    np.add.at(unique_weights, inverse, all_weights)

    mask = np.abs(unique_weights) > 1e-15
    filtered_keys = unique_keys[mask]
    filtered_weights = unique_weights[mask]
    filtered_nodes = _clenshaw_curtis_nodes_from_keys(filtered_keys)
    order = np.argsort(filtered_nodes)
    return filtered_nodes[order], filtered_weights[order]


# 責務: Clenshaw-Curtis の差分積分則 Delta_level を構築する。
def difference_rule(level: int, /) -> tuple[Vector, Vector]:
    diff_nodes_np, diff_weights_np = _difference_rule_numpy(level)
    return cast(
        tuple[Vector, Vector],
        _to_jax_arrays(diff_nodes_np, diff_weights_np, dtype=DEFAULT_DTYPE),
    )


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


def _normalize_dimension_weights(
    dimension: int,
    dimension_weights: tuple[int, ...] | None,
    /,
) -> tuple[int, ...] | None:
    if dimension_weights is None:
        return None
    if len(dimension_weights) != dimension:
        raise ValueError("dimension_weights must have length equal to dimension.")
    normalized = tuple(int(weight) for weight in dimension_weights)
    if any(weight < 1 for weight in normalized):
        raise ValueError("dimension_weights must be positive integers.")
    return normalized


def _weighted_multi_indices(
    dimension: int,
    level: int,
    dimension_weights: tuple[int, ...],
    /,
) -> NDArray[np.unsignedinteger[Any]]:
    if dimension < 1:
        raise ValueError("dimension must be positive.")
    if level < 1:
        raise ValueError("level must be positive.")

    budget = level - 1
    max_level = 1 + max(budget // weight for weight in dimension_weights)
    index_dtype = _compact_unsigned_dtype(max(max_level, dimension, 1))
    count_cache: dict[tuple[int, int], int] = {}

    def count(axis: int, remaining_budget: int) -> int:
        key = (axis, remaining_budget)
        cached = count_cache.get(key)
        if cached is not None:
            return cached
        if axis == dimension:
            return 1
        weight = dimension_weights[axis]
        total = 0
        for extra_level in range((remaining_budget // weight) + 1):
            total += count(axis + 1, remaining_budget - extra_level * weight)
        count_cache[key] = total
        return total

    num_indices = count(0, budget)
    indices = np.empty((num_indices, dimension), dtype=index_dtype)
    current = np.ones((dimension,), dtype=index_dtype)

    def fill(axis: int, remaining_budget: int, row: int) -> int:
        if axis == dimension:
            indices[row] = current
            return row + 1
        weight = dimension_weights[axis]
        for extra_level in range((remaining_budget // weight) + 1):
            current[axis] = 1 + extra_level
            row = fill(axis + 1, remaining_budget - extra_level * weight, row)
        return row

    fill(0, budget, 0)
    return indices


# 責務: Smolyak 多重指数 i の 1-ノルム上限 q = d + l - 1 を返す。
def _max_multi_index_norm(dimension: int, level: int, /) -> int:
    return dimension + level - 1


# 責務: Smolyak level と次元から必要な最大 1 次元差分則 level を返す。
def _max_difference_rule_level(
    dimension: int,
    level: int,
    /,
    *,
    dimension_weights: tuple[int, ...] | None = None,
) -> int:
    del dimension
    if dimension_weights is None:
        return level
    budget = level - 1
    return 1 + max(budget // weight for weight in dimension_weights)


def _axis_level_ceilings_numpy(
    dimension: int,
    level: int,
    /,
    *,
    dimension_weights: tuple[int, ...] | None = None,
) -> NDArray[np.int32]:
    if level < 1:
        raise ValueError("level must be positive.")
    if dimension_weights is None:
        return np.full((dimension,), level, dtype=np.int32)
    budget = level - 1
    return np.asarray(
        [1 + (budget // weight) for weight in dimension_weights],
        dtype=np.int32,
    )


def _smolyak_term_levels_numpy(
    dimension: int,
    level: int,
    /,
    *,
    dimension_weights: tuple[int, ...] | None = None,
) -> NDArray[np.integer[Any]]:
    if dimension_weights is None:
        max_norm = _max_multi_index_norm(dimension, level)
        return multi_indices(dimension, max_norm)
    return _weighted_multi_indices(dimension, level, dimension_weights)


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
        nodes_np, weights_np = _difference_rule_numpy(current_level)
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
def _term_axis_strides_numpy(
    term_rule_lengths_np: NDArray[np.int64],
    /,
) -> NDArray[np.int64]:
    term_axis_strides_np = np.ones_like(term_rule_lengths_np, dtype=np.int64)
    for axis in range(term_rule_lengths_np.shape[1] - 2, -1, -1):
        term_axis_strides_np[:, axis] = (
            term_axis_strides_np[:, axis + 1] * term_rule_lengths_np[:, axis + 1]
        )
    return term_axis_strides_np


def _initialize_term_plan_numpy(
    dimension: int,
    level: int,
    rule_offsets_np: NDArray[np.int64],
    rule_lengths_np: NDArray[np.int64],
    /,
    *,
    dimension_weights: tuple[int, ...] | None = None,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    int,
    int,
]:
    term_levels_np = _smolyak_term_levels_numpy(
        dimension,
        level,
        dimension_weights=dimension_weights,
    ).astype(np.int32, copy=False)
    level_indices_np = term_levels_np.astype(np.int64, copy=False) - 1
    term_rule_offsets_np = rule_offsets_np[level_indices_np]
    term_rule_lengths_np = rule_lengths_np[level_indices_np]
    term_axis_strides_np = _term_axis_strides_numpy(term_rule_lengths_np)
    term_num_points_np = np.prod(term_rule_lengths_np, axis=1, dtype=np.int64)
    term_point_offsets_np = np.empty((term_num_points_np.shape[0] + 1,), dtype=np.int64)
    term_point_offsets_np[0] = 0
    np.cumsum(term_num_points_np, dtype=np.int64, out=term_point_offsets_np[1:])
    num_terms = int(term_levels_np.shape[0])
    num_evaluation_points = int(term_num_points_np.sum(dtype=np.int64))
    return (
        term_levels_np,
        term_rule_offsets_np,
        term_rule_lengths_np,
        term_axis_strides_np,
        term_num_points_np,
        term_point_offsets_np,
        num_terms,
        num_evaluation_points,
    )


def _estimate_dense_materialized_plan_bytes(
    dimension: int,
    num_evaluation_points: int,
    dtype: DTypeLike,
    /,
) -> int:
    dtype_bytes = np.dtype(dtype).itemsize
    return num_evaluation_points * (dimension + 1) * dtype_bytes


def _dense_materialized_plan_fits(
    dimension: int,
    num_evaluation_points: int,
    dtype: DTypeLike,
    /,
    *,
    max_materialized_plan_bytes: int,
) -> bool:
    estimated_bytes = _estimate_dense_materialized_plan_bytes(
        dimension,
        num_evaluation_points,
        dtype,
    )
    return estimated_bytes <= max_materialized_plan_bytes


def _estimate_indexed_materialized_plan_bytes(
    dimension: int,
    num_evaluation_points: int,
    dtype: DTypeLike,
    max_rule_index: int,
    /,
) -> int:
    dtype_bytes = np.dtype(dtype).itemsize
    index_dtype = _compact_unsigned_dtype(max(max_rule_index, 1))
    index_bytes = np.dtype(index_dtype).itemsize
    return num_evaluation_points * (dimension * index_bytes + dtype_bytes)


def _indexed_materialized_plan_fits(
    dimension: int,
    num_evaluation_points: int,
    dtype: DTypeLike,
    max_rule_index: int,
    /,
    *,
    max_materialized_plan_bytes: int,
) -> bool:
    estimated_bytes = _estimate_indexed_materialized_plan_bytes(
        dimension,
        num_evaluation_points,
        dtype,
        max_rule_index,
    )
    return estimated_bytes <= max_materialized_plan_bytes


def _materialization_byte_limits(requested_mode: str, /) -> tuple[int, int]:
    if requested_mode == "auto":
        return _MAX_FULL_MATERIALIZED_PLAN_BYTES, _MAX_INDEXED_MATERIALIZED_PLAN_BYTES
    if requested_mode == "points":
        return 1 << 60, 1
    if requested_mode == "indexed":
        return 1, 1 << 60
    if requested_mode == "lazy-indexed":
        return 1, 1
    if requested_mode == "batched":
        return 1, 1
    raise ValueError(f"Unknown requested_materialization_mode: {requested_mode}")


def _materialize_term_plan_numpy(
    rule_nodes_np: NDArray[np.floating[Any]],
    rule_weights_np: NDArray[np.floating[Any]],
    term_rule_offsets_np: NDArray[np.int64],
    term_rule_lengths_np: NDArray[np.int64],
    term_axis_strides_np: NDArray[np.int64],
    term_num_points_np: NDArray[np.int64],
    /,
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    dimension = int(term_rule_offsets_np.shape[1])
    total_points = int(term_num_points_np.sum(dtype=np.int64))
    plan_points_np = np.empty((dimension, total_points), dtype=rule_nodes_np.dtype)
    plan_weights_np = np.empty((total_points,), dtype=rule_weights_np.dtype)

    cursor = 0
    for term_index in range(term_rule_offsets_np.shape[0]):
        current_offsets = term_rule_offsets_np[term_index]
        current_lengths = term_rule_lengths_np[term_index]
        current_strides = term_axis_strides_np[term_index]
        current_num_points = int(term_num_points_np[term_index])
        flat_indices = np.arange(current_num_points, dtype=np.int64)
        local_indices = np.floor_divide(
            flat_indices[np.newaxis, :],
            current_strides[:, np.newaxis],
        )
        local_indices %= current_lengths[:, np.newaxis]
        rule_indices = current_offsets[:, np.newaxis] + local_indices
        next_cursor = cursor + current_num_points
        plan_points_np[:, cursor:next_cursor] = np.take(rule_nodes_np, rule_indices, mode="clip")
        plan_weights_np[cursor:next_cursor] = np.prod(
            np.take(rule_weights_np, rule_indices, mode="clip"),
            axis=0,
            dtype=rule_weights_np.dtype,
        )
        cursor = next_cursor

    return plan_points_np, plan_weights_np


def _materialize_indexed_term_plan_numpy(
    rule_weights_np: NDArray[np.floating[Any]],
    term_rule_offsets_np: NDArray[np.int64],
    term_rule_lengths_np: NDArray[np.int64],
    term_axis_strides_np: NDArray[np.int64],
    term_num_points_np: NDArray[np.int64],
    /,
) -> tuple[
    NDArray[np.unsignedinteger[Any]],
    NDArray[np.floating[Any]],
]:
    dimension = int(term_rule_offsets_np.shape[1])
    total_points = int(term_num_points_np.sum(dtype=np.int64))
    max_rule_index = max(int(rule_weights_np.shape[0]) - 1, 1)
    index_dtype = _compact_unsigned_dtype(max_rule_index)
    plan_rule_indices_np = np.empty((dimension, total_points), dtype=index_dtype)
    plan_weights_np = np.empty((total_points,), dtype=rule_weights_np.dtype)

    cursor = 0
    for term_index in range(term_rule_offsets_np.shape[0]):
        current_offsets = term_rule_offsets_np[term_index]
        current_lengths = term_rule_lengths_np[term_index]
        current_strides = term_axis_strides_np[term_index]
        current_num_points = int(term_num_points_np[term_index])
        flat_indices = np.arange(current_num_points, dtype=np.int64)
        local_indices = np.floor_divide(
            flat_indices[np.newaxis, :],
            current_strides[:, np.newaxis],
        )
        local_indices %= current_lengths[:, np.newaxis]
        rule_indices = current_offsets[:, np.newaxis] + local_indices
        next_cursor = cursor + current_num_points
        plan_rule_indices_np[:, cursor:next_cursor] = rule_indices.astype(index_dtype, copy=False)
        plan_weights_np[cursor:next_cursor] = np.prod(
            np.take(rule_weights_np, rule_indices, mode="clip"),
            axis=0,
            dtype=rule_weights_np.dtype,
        )
        cursor = next_cursor

    return plan_rule_indices_np, plan_weights_np


# 責務: 1 次元差分則 storage と term plan の保持量を byte 単位で見積もる。
def _storage_bytes(
    rule_nodes: jax.Array,
    rule_weights: jax.Array,
    rule_offsets: jax.Array,
    rule_lengths: jax.Array,
    term_levels: jax.Array,
    term_rule_offsets: jax.Array,
    term_rule_lengths: jax.Array,
    term_axis_strides: jax.Array,
    term_num_points: jax.Array,
    term_point_offsets: jax.Array,
    materialized_rule_indices: jax.Array,
    materialized_points: jax.Array,
    materialized_weights: jax.Array,
    batched_term_rule_offsets: jax.Array,
    batched_term_rule_lengths: jax.Array,
    batched_term_axis_strides: jax.Array,
    batched_term_inverse_axis_permutations: jax.Array,
    axis_level_ceilings: jax.Array,
    /,
) -> int:
    return (
        int(rule_nodes.nbytes)
        + int(rule_weights.nbytes)
        + int(rule_offsets.nbytes)
        + int(rule_lengths.nbytes)
        + int(term_levels.nbytes)
        + int(term_rule_offsets.nbytes)
        + int(term_rule_lengths.nbytes)
        + int(term_axis_strides.nbytes)
        + int(term_num_points.nbytes)
        + int(term_point_offsets.nbytes)
        + int(materialized_rule_indices.nbytes)
        + int(materialized_points.nbytes)
        + int(materialized_weights.nbytes)
        + int(batched_term_rule_offsets.nbytes)
        + int(batched_term_rule_lengths.nbytes)
        + int(batched_term_axis_strides.nbytes)
        + int(batched_term_inverse_axis_permutations.nbytes)
        + int(axis_level_ceilings.nbytes)
    )


# 責務: batched 評価用の計算軸順とその逆置換を NumPy 上で構築する。
def _batched_axis_permutations_numpy(
    term_rule_lengths_np: NDArray[np.integer[Any]],
    /,
    axis_order_strategy: str,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    term_rule_lengths_np = np.asarray(term_rule_lengths_np, dtype=np.int64)
    num_terms, dimension = term_rule_lengths_np.shape
    if axis_order_strategy == "original":
        permutations_np = np.broadcast_to(
            np.arange(dimension, dtype=np.int64),
            (num_terms, dimension),
        ).copy()
    elif axis_order_strategy == "length":
        permutations_np = np.argsort(term_rule_lengths_np, axis=1, kind="stable")
    else:
        raise ValueError(f"Unknown batched_axis_order_strategy: {axis_order_strategy}")
    inverse_permutations_np = np.empty_like(permutations_np)
    for term_index in range(num_terms):
        inverse_permutations_np[term_index, permutations_np[term_index]] = np.arange(
            dimension,
            dtype=np.int64,
        )
    ordered_lengths_np = np.take_along_axis(term_rule_lengths_np, permutations_np, axis=1)
    return permutations_np, inverse_permutations_np, ordered_lengths_np


# 責務: batched 評価用の計算軸順のもとで suffix 幅を決める。
def _choose_vectorized_suffix_config(
    ordered_term_rule_lengths_np: NDArray[np.integer[Any]],
    dimension: int,
    /,
    *,
    max_vectorized_suffix_ndim: int,
    max_vectorized_points: int = 131072,
) -> tuple[int, int]:
    ordered_term_rule_lengths_np = np.asarray(ordered_term_rule_lengths_np, dtype=np.int64)
    max_candidate_ndim = min(max_vectorized_suffix_ndim, dimension)
    for vectorized_ndim in range(max_candidate_ndim, 0, -1):
        candidate_points = int(
            np.max(
                np.prod(
                    ordered_term_rule_lengths_np[:, -vectorized_ndim:],
                    axis=1,
                    dtype=np.int64,
                )
            )
        )
        if candidate_points <= max_vectorized_points:
            return vectorized_ndim, candidate_points
    return 1, int(np.max(ordered_term_rule_lengths_np[:, -1]))


# 責務: 先頭軸を scan し、末尾 2-3 軸をまとめて batched 評価して積分値を求める。
def _batched_plan_integral(
    f: Function,
    dimension: int,
    dtype: DTypeLike,
    batched_term_rule_offsets: jax.Array,
    batched_term_rule_lengths: jax.Array,
    batched_term_axis_strides: jax.Array,
    batched_term_inverse_axis_permutations: jax.Array,
    term_num_points: jax.Array,
    rule_nodes: Vector,
    rule_weights: Vector,
    *,
    vectorized_ndim: int,
    max_vectorized_points: int,
) -> Vector:
    zero_point = jnp.zeros((dimension,), dtype=dtype)
    initial_value = jnp.zeros_like(f(zero_point))
    prefix_ndim = dimension - vectorized_ndim
    suffix_point_indices = jnp.arange(max_vectorized_points, dtype=jnp.int64)

    def term_body(
        acc: Vector,
        term_data: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[Vector, None]:
        (
            current_offsets,
            current_lengths,
            current_strides,
            current_inverse_permutation,
            current_num_points,
        ) = term_data
        current_suffix_offsets = current_offsets[prefix_ndim:]
        current_suffix_lengths = current_lengths[prefix_ndim:]
        current_suffix_strides = current_strides[prefix_ndim:]
        current_suffix_num_points = jnp.prod(current_suffix_lengths)
        valid_suffix_mask = suffix_point_indices < current_suffix_num_points
        safe_suffix_indices = jnp.where(valid_suffix_mask, suffix_point_indices, 0)
        suffix_local_indices = jnp.floor_divide(
            safe_suffix_indices[jnp.newaxis, :],
            current_suffix_strides[:, jnp.newaxis],
        )
        suffix_local_indices = jnp.mod(
            suffix_local_indices,
            current_suffix_lengths[:, jnp.newaxis],
        )
        suffix_rule_indices = current_suffix_offsets[:, jnp.newaxis] + suffix_local_indices
        suffix_points = jnp.take(rule_nodes, suffix_rule_indices, mode="clip")
        suffix_weights = jnp.prod(
            jnp.take(rule_weights, suffix_rule_indices, mode="clip"),
            axis=0,
        )
        masked_suffix_weights = jnp.where(valid_suffix_mask, suffix_weights, 0)

        if prefix_ndim == 0:
            full_points = jnp.take(suffix_points, current_inverse_permutation, axis=0, mode="clip")
            values = jax.vmap(f, in_axes=1, out_axes=-1)(full_points)
            next_acc = acc + jnp.tensordot(values, masked_suffix_weights, axes=(-1, 0))
            return next_acc, None

        current_prefix_offsets = current_offsets[:prefix_ndim]
        current_prefix_lengths = current_lengths[:prefix_ndim]
        current_prefix_strides = jnp.floor_divide(
            current_strides[:prefix_ndim],
            current_suffix_num_points,
        )
        prefix_num_points = jnp.floor_divide(current_num_points, current_suffix_num_points)

        def prefix_body(prefix_index: int, prefix_acc: Vector) -> Vector:
            prefix_local_indices = jnp.floor_divide(
                jnp.asarray(prefix_index, dtype=jnp.int64),
                current_prefix_strides,
            )
            prefix_local_indices = jnp.mod(prefix_local_indices, current_prefix_lengths)
            prefix_rule_indices = current_prefix_offsets + prefix_local_indices
            prefix_points = jnp.take(rule_nodes, prefix_rule_indices, mode="clip")
            prefix_weights = jnp.take(rule_weights, prefix_rule_indices, mode="clip")
            full_points_ordered = jnp.concatenate(
                (
                    jnp.broadcast_to(
                        prefix_points[:, jnp.newaxis],
                        (prefix_ndim, max_vectorized_points),
                    ),
                    suffix_points,
                ),
                axis=0,
            )
            full_points = jnp.take(full_points_ordered, current_inverse_permutation, axis=0, mode="clip")
            values = jax.vmap(f, in_axes=1, out_axes=-1)(full_points)
            total_weights = jnp.prod(prefix_weights) * masked_suffix_weights
            return prefix_acc + jnp.tensordot(values, total_weights, axes=(-1, 0))

        next_acc = jax.lax.fori_loop(0, prefix_num_points, prefix_body, acc)
        return next_acc, None

    final_value, _ = jax.lax.scan(
        term_body,
        initial_value,
        xs=(
            batched_term_rule_offsets,
            batched_term_rule_lengths,
            batched_term_axis_strides,
            batched_term_inverse_axis_permutations,
            term_num_points,
        ),
    )
    return final_value


def _materialized_plan_integral(
    f: Function,
    dimension: int,
    dtype: DTypeLike,
    materialized_points: jax.Array,
    materialized_weights: jax.Array,
    *,
    chunk_size: int,
) -> Vector:
    zero_point = jnp.zeros((dimension,), dtype=dtype)
    initial_value = jnp.zeros_like(f(zero_point))
    flat_chunk_indices = jnp.arange(chunk_size, dtype=jnp.int64)
    total_points = materialized_weights.shape[0]

    def chunk_body(chunk_index: int, acc: Vector) -> Vector:
        start = jnp.asarray(chunk_index, dtype=jnp.int64) * chunk_size
        local_indices = start + flat_chunk_indices
        valid_mask = local_indices < total_points
        safe_indices = jnp.where(valid_mask, local_indices, 0)
        points = jnp.take(materialized_points, safe_indices, axis=1, mode="clip")
        weights = jnp.take(materialized_weights, safe_indices, mode="clip")
        masked_weights = jnp.where(valid_mask, weights, 0)
        values = jax.vmap(f, in_axes=1, out_axes=-1)(points)
        return acc + jnp.tensordot(values, masked_weights, axes=(-1, 0))

    num_chunks = jnp.floor_divide(total_points + chunk_size - 1, chunk_size)
    return jax.lax.fori_loop(0, num_chunks, chunk_body, initial_value)


def _indexed_materialized_plan_integral(
    f: Function,
    dimension: int,
    dtype: DTypeLike,
    rule_nodes: Vector,
    materialized_rule_indices: jax.Array,
    materialized_weights: jax.Array,
    *,
    chunk_size: int,
) -> Vector:
    zero_point = jnp.zeros((dimension,), dtype=dtype)
    initial_value = jnp.zeros_like(f(zero_point))
    flat_chunk_indices = jnp.arange(chunk_size, dtype=jnp.int64)
    total_points = materialized_weights.shape[0]

    def chunk_body(chunk_index: int, acc: Vector) -> Vector:
        start = jnp.asarray(chunk_index, dtype=jnp.int64) * chunk_size
        local_indices = start + flat_chunk_indices
        valid_mask = local_indices < total_points
        safe_indices = jnp.where(valid_mask, local_indices, 0)
        point_rule_indices = jnp.take(materialized_rule_indices, safe_indices, axis=1, mode="clip")
        points = jnp.take(rule_nodes, point_rule_indices, mode="clip")
        weights = jnp.take(materialized_weights, safe_indices, mode="clip")
        masked_weights = jnp.where(valid_mask, weights, 0)
        values = jax.vmap(f, in_axes=1, out_axes=-1)(points)
        return acc + jnp.tensordot(values, masked_weights, axes=(-1, 0))

    num_chunks = jnp.floor_divide(total_points + chunk_size - 1, chunk_size)
    return jax.lax.fori_loop(0, num_chunks, chunk_body, initial_value)


def _lazy_indexed_plan_integral(
    f: Function,
    dimension: int,
    dtype: DTypeLike,
    rule_nodes: Vector,
    rule_weights: Vector,
    term_point_offsets: jax.Array,
    term_rule_offsets: jax.Array,
    term_rule_lengths: jax.Array,
    term_axis_strides: jax.Array,
    *,
    chunk_size: int,
) -> Vector:
    zero_point = jnp.zeros((dimension,), dtype=dtype)
    initial_value = jnp.zeros_like(f(zero_point))
    flat_chunk_indices = jnp.arange(chunk_size, dtype=jnp.int64)
    total_points = term_point_offsets[-1]

    def chunk_body(chunk_index: int, acc: Vector) -> Vector:
        start = jnp.asarray(chunk_index, dtype=jnp.int64) * chunk_size
        point_indices = start + flat_chunk_indices
        valid_mask = point_indices < total_points
        safe_indices = jnp.where(valid_mask, point_indices, 0)

        term_indices = jnp.searchsorted(term_point_offsets[1:], safe_indices, side="right")
        term_starts = jnp.take(term_point_offsets, term_indices, mode="clip")
        local_point_indices = safe_indices - term_starts

        current_offsets = jnp.swapaxes(
            jnp.take(term_rule_offsets, term_indices, axis=0, mode="clip"),
            0,
            1,
        )
        current_lengths = jnp.swapaxes(
            jnp.take(term_rule_lengths, term_indices, axis=0, mode="clip"),
            0,
            1,
        )
        current_strides = jnp.swapaxes(
            jnp.take(term_axis_strides, term_indices, axis=0, mode="clip"),
            0,
            1,
        )

        local_indices = jnp.floor_divide(local_point_indices[jnp.newaxis, :], current_strides)
        local_indices = jnp.mod(local_indices, current_lengths)
        rule_indices = current_offsets + local_indices
        points = jnp.take(rule_nodes, rule_indices, mode="clip")
        weights = jnp.prod(
            jnp.take(rule_weights, rule_indices, mode="clip"),
            axis=0,
        )
        masked_weights = jnp.where(valid_mask, weights, 0)
        values = jax.vmap(f, in_axes=1, out_axes=-1)(points)
        return acc + jnp.tensordot(values, masked_weights, axes=(-1, 0))

    num_chunks = jnp.floor_divide(total_points + chunk_size - 1, chunk_size)
    return jax.lax.fori_loop(0, num_chunks, chunk_body, initial_value)


class SmolyakIntegrator(eqx.Module):
    dimension: int
    level: int
    prepared_level: int
    dimension_weights: tuple[int, ...] | None = eqx.field(static=True)
    requested_materialization_mode: str = eqx.field(static=True)
    max_vectorized_suffix_ndim: int = eqx.field(static=True)
    batched_axis_order_strategy: str = eqx.field(static=True)
    dtype: DTypeLike = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    vectorized_ndim: int = eqx.field(static=True)
    max_vectorized_points: int = eqx.field(static=True)
    rule_nodes: Vector
    rule_weights: Vector
    rule_offsets: jax.Array
    rule_lengths: jax.Array
    term_levels: jax.Array
    term_rule_offsets: jax.Array
    term_rule_lengths: jax.Array
    term_axis_strides: jax.Array
    term_num_points: jax.Array
    term_point_offsets: jax.Array
    batched_term_rule_offsets: jax.Array
    batched_term_rule_lengths: jax.Array
    batched_term_axis_strides: jax.Array
    batched_term_inverse_axis_permutations: jax.Array
    materialization_mode: str = eqx.field(static=True)
    materialized_rule_indices: jax.Array
    materialized_points: jax.Array
    materialized_weights: jax.Array
    axis_level_ceilings: jax.Array
    active_axis_count: int
    num_terms: int
    num_evaluation_points: int
    storage_bytes: int

    def __init__(
        self,
        dimension: int,
        level: int,
        prepared_level: int | None = None,
        dimension_weights: tuple[int, ...] | None = None,
        requested_materialization_mode: str = "auto",
        max_vectorized_suffix_ndim: int = 3,
        batched_axis_order_strategy: str = "original",
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
        if max_vectorized_suffix_ndim < 1:
            raise ValueError("max_vectorized_suffix_ndim must be positive.")
        resolved_prepared_level = level if prepared_level is None else prepared_level
        if resolved_prepared_level < level:
            raise ValueError("prepared_level must be greater than or equal to level.")
        if requested_materialization_mode not in _SUPPORTED_REQUESTED_MATERIALIZATION_MODES:
            raise ValueError(
                "requested_materialization_mode must be one of "
                f"{sorted(_SUPPORTED_REQUESTED_MATERIALIZATION_MODES)}."
            )
        if batched_axis_order_strategy not in _SUPPORTED_BATCHED_AXIS_ORDER_STRATEGIES:
            raise ValueError(
                "batched_axis_order_strategy must be one of "
                f"{sorted(_SUPPORTED_BATCHED_AXIS_ORDER_STRATEGIES)}."
            )

        self.dimension = dimension
        self.level = level
        self.prepared_level = resolved_prepared_level
        self.dimension_weights = _normalize_dimension_weights(dimension, dimension_weights)
        self.requested_materialization_mode = requested_materialization_mode
        self.max_vectorized_suffix_ndim = max_vectorized_suffix_ndim
        self.batched_axis_order_strategy = batched_axis_order_strategy
        self.dtype = dtype
        self.chunk_size = chunk_size
        axis_level_ceilings_np = _axis_level_ceilings_numpy(
            dimension,
            level,
            dimension_weights=self.dimension_weights,
        )
        self.axis_level_ceilings = jnp.asarray(axis_level_ceilings_np, dtype=jnp.int32)
        self.active_axis_count = int(np.count_nonzero(axis_level_ceilings_np > 1))
        self.materialization_mode = "batched"
        empty_rule_indices = jnp.empty((dimension, 0), dtype=np.uint8)
        empty_points = jnp.empty((dimension, 0), dtype=dtype)
        empty_weights = jnp.empty((0,), dtype=dtype)
        if _rule_storage is None:
            max_rule_level = _max_difference_rule_level(
                dimension,
                resolved_prepared_level,
                dimension_weights=self.dimension_weights,
            )
            (
                rule_nodes_np,
                rule_weights_np,
                rule_offsets_np,
                rule_lengths_np,
            ) = _difference_rule_storage_numpy(max_rule_level)
            self.rule_nodes = jnp.asarray(rule_nodes_np, dtype=dtype)
            self.rule_weights = jnp.asarray(rule_weights_np, dtype=dtype)
            self.rule_offsets = jnp.asarray(rule_offsets_np, dtype=jnp.int64)
            self.rule_lengths = jnp.asarray(rule_lengths_np, dtype=jnp.int64)
        else:
            (
                self.rule_nodes,
                self.rule_weights,
                self.rule_offsets,
                self.rule_lengths,
            ) = _rule_storage
            rule_nodes_np = np.asarray(self.rule_nodes)
            rule_weights_np = np.asarray(self.rule_weights)
            rule_offsets_np = np.asarray(self.rule_offsets, dtype=np.int64)
            rule_lengths_np = np.asarray(self.rule_lengths, dtype=np.int64)

        (
            term_levels_np,
            term_rule_offsets_np,
            term_rule_lengths_np,
            term_axis_strides_np,
            term_num_points_np,
            term_point_offsets_np,
            self.num_terms,
            self.num_evaluation_points,
        ) = _initialize_term_plan_numpy(
            dimension,
            level,
            rule_offsets_np,
            rule_lengths_np,
            dimension_weights=self.dimension_weights,
        )
        dense_plan_limit, indexed_plan_limit = _materialization_byte_limits(requested_materialization_mode)
        max_rule_index = max(int(rule_weights_np.shape[0]) - 1, 1)
        if requested_materialization_mode == "lazy-indexed":
            self.materialization_mode = "lazy-indexed"
            self.materialized_rule_indices = empty_rule_indices
            self.materialized_points = empty_points
            self.materialized_weights = empty_weights
        elif _dense_materialized_plan_fits(
            dimension,
            self.num_evaluation_points,
            dtype,
            max_materialized_plan_bytes=dense_plan_limit,
        ):
            materialized_points_np, materialized_weights_np = _materialize_term_plan_numpy(
                rule_nodes_np,
                rule_weights_np,
                term_rule_offsets_np,
                term_rule_lengths_np,
                term_axis_strides_np,
                term_num_points_np,
            )
            self.materialization_mode = "points"
            self.materialized_rule_indices = empty_rule_indices
            self.materialized_points = jnp.asarray(materialized_points_np, dtype=dtype)
            self.materialized_weights = jnp.asarray(materialized_weights_np, dtype=dtype)
        elif _indexed_materialized_plan_fits(
            dimension,
            self.num_evaluation_points,
            dtype,
            max_rule_index,
            max_materialized_plan_bytes=indexed_plan_limit,
        ):
            materialized_rule_indices_np, materialized_weights_np = _materialize_indexed_term_plan_numpy(
                rule_weights_np,
                term_rule_offsets_np,
                term_rule_lengths_np,
                term_axis_strides_np,
                term_num_points_np,
            )
            self.materialization_mode = "indexed"
            self.materialized_rule_indices = jnp.asarray(materialized_rule_indices_np)
            self.materialized_points = empty_points
            self.materialized_weights = jnp.asarray(materialized_weights_np, dtype=dtype)
        else:
            self.materialized_rule_indices = empty_rule_indices
            self.materialized_points = empty_points
            self.materialized_weights = empty_weights
        (
            batched_axis_permutations_np,
            batched_inverse_axis_permutations_np,
            batched_term_rule_lengths_np,
        ) = _batched_axis_permutations_numpy(
            term_rule_lengths_np,
            batched_axis_order_strategy,
        )
        batched_term_rule_offsets_np = np.take_along_axis(
            term_rule_offsets_np,
            batched_axis_permutations_np,
            axis=1,
        )
        batched_term_axis_strides_np = _term_axis_strides_numpy(batched_term_rule_lengths_np)
        self.vectorized_ndim, self.max_vectorized_points = _choose_vectorized_suffix_config(
            batched_term_rule_lengths_np,
            dimension,
            max_vectorized_suffix_ndim=max_vectorized_suffix_ndim,
        )
        self.term_levels = jnp.asarray(term_levels_np, dtype=jnp.int32)
        self.term_rule_offsets = jnp.asarray(term_rule_offsets_np, dtype=jnp.int64)
        self.term_rule_lengths = jnp.asarray(term_rule_lengths_np, dtype=jnp.int64)
        self.term_axis_strides = jnp.asarray(term_axis_strides_np, dtype=jnp.int64)
        self.term_num_points = jnp.asarray(term_num_points_np, dtype=jnp.int64)
        self.term_point_offsets = jnp.asarray(term_point_offsets_np, dtype=jnp.int64)
        self.batched_term_rule_offsets = jnp.asarray(batched_term_rule_offsets_np, dtype=jnp.int64)
        self.batched_term_rule_lengths = jnp.asarray(batched_term_rule_lengths_np, dtype=jnp.int64)
        self.batched_term_axis_strides = jnp.asarray(batched_term_axis_strides_np, dtype=jnp.int64)
        self.batched_term_inverse_axis_permutations = jnp.asarray(
            batched_inverse_axis_permutations_np,
            dtype=jnp.int64,
        )
        self.storage_bytes = _storage_bytes(
            self.rule_nodes,
            self.rule_weights,
            self.rule_offsets,
            self.rule_lengths,
            self.term_levels,
            self.term_rule_offsets,
            self.term_rule_lengths,
            self.term_axis_strides,
            self.term_num_points,
            self.term_point_offsets,
            self.materialized_rule_indices,
            self.materialized_points,
            self.materialized_weights,
            self.batched_term_rule_offsets,
            self.batched_term_rule_lengths,
            self.batched_term_axis_strides,
            self.batched_term_inverse_axis_permutations,
            self.axis_level_ceilings,
        )

    def integrate(self, f: Function, /) -> Vector:
        # 実験コード側で JIT を適用する前提にして、ここでは Python callable の
        # トレース失敗を避けるため module 側の jitting は行わない。
        if self.materialization_mode == "points":
            return _materialized_plan_integral(
                f,
                self.dimension,
                self.dtype,
                self.materialized_points,
                self.materialized_weights,
                chunk_size=self.chunk_size,
            )
        if self.materialization_mode == "indexed":
            return _indexed_materialized_plan_integral(
                f,
                self.dimension,
                self.dtype,
                self.rule_nodes,
                self.materialized_rule_indices,
                self.materialized_weights,
                chunk_size=self.chunk_size,
            )
        if self.materialization_mode == "lazy-indexed":
            return _lazy_indexed_plan_integral(
                f,
                self.dimension,
                self.dtype,
                self.rule_nodes,
                self.rule_weights,
                self.term_point_offsets,
                self.term_rule_offsets,
                self.term_rule_lengths,
                self.term_axis_strides,
                chunk_size=self.chunk_size,
            )
        return _batched_plan_integral(
            f,
            self.dimension,
            self.dtype,
            self.batched_term_rule_offsets,
            self.batched_term_rule_lengths,
            self.batched_term_axis_strides,
            self.batched_term_inverse_axis_permutations,
            self.term_num_points,
            self.rule_nodes,
            self.rule_weights,
            vectorized_ndim=self.vectorized_ndim,
            max_vectorized_points=self.max_vectorized_points,
        )

    # 責務: より細かい疎格子を使う次レベルの積分器を返す。
    def refine(self) -> "SmolyakIntegrator":
        next_level = self.level + 1
        if next_level <= self.prepared_level:
            return SmolyakIntegrator(
                dimension=self.dimension,
                level=next_level,
                prepared_level=self.prepared_level,
                dimension_weights=self.dimension_weights,
                requested_materialization_mode=self.requested_materialization_mode,
                max_vectorized_suffix_ndim=self.max_vectorized_suffix_ndim,
                batched_axis_order_strategy=self.batched_axis_order_strategy,
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
            dimension_weights=self.dimension_weights,
            requested_materialization_mode=self.requested_materialization_mode,
            max_vectorized_suffix_ndim=self.max_vectorized_suffix_ndim,
            batched_axis_order_strategy=self.batched_axis_order_strategy,
            dtype=self.dtype,
            chunk_size=self.chunk_size,
        )


# 責務: plan 化された Smolyak 積分器を初期化して返す。
def initialize_smolyak_integrator(
    dimension: int,
    level: int,
    *,
    prepared_level: int | None = None,
    dimension_weights: tuple[int, ...] | None = None,
    requested_materialization_mode: str = "auto",
    max_vectorized_suffix_ndim: int = 3,
    batched_axis_order_strategy: str = "original",
    dtype: DTypeLike = DEFAULT_DTYPE,
    chunk_size: int = 16384,
) -> SmolyakIntegrator:
    return SmolyakIntegrator(
        dimension=dimension,
        level=level,
        prepared_level=prepared_level,
        dimension_weights=dimension_weights,
        requested_materialization_mode=requested_materialization_mode,
        max_vectorized_suffix_ndim=max_vectorized_suffix_ndim,
        batched_axis_order_strategy=batched_axis_order_strategy,
        dtype=dtype,
        chunk_size=chunk_size,
    )


__all__ = [
    "SmolyakIntegrator",
    "clenshaw_curtis_rule",
    "difference_rule",
    "initialize_smolyak_integrator",
    "multi_indices",
]
