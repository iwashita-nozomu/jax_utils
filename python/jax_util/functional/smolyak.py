from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import DTypeLike

from ..base import DEFAULT_DTYPE, Vector
from .protocols import Function

_SUPPORTED_INDEX_DTYPES: dict[str, np.dtype[Any]] = {
    "int8": np.dtype(np.int8),
    "uint8": np.dtype(np.uint8),
    "int16": np.dtype(np.int16),
    "int32": np.dtype(np.int32),
}


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


# 責務: even extension + FFT により DCT-I を device 上で構成する。
def _dct_type_one(values: Vector, /) -> Vector:
    if values.ndim != 1:
        raise ValueError("values must be a 1D array.")
    if values.shape[0] <= 1:
        return values
    extended = jnp.concatenate([values, values[-2:0:-1]], axis=0)
    return jnp.fft.fft(extended).real[: values.shape[0]]


# 責務: FFT ベースの DCT-I により Clenshaw-Curtis の重み列を device 上で構成する。
def _clenshaw_curtis_weights_device(num_intervals: int, /) -> Vector:
    scalar_dtype = jnp.float64
    coefficients = jnp.zeros((num_intervals + 1,), dtype=scalar_dtype)
    coefficients = coefficients.at[0].set(jnp.asarray(1.0, dtype=scalar_dtype))

    if num_intervals >= 2:
        mode_limit = num_intervals // 2
        if mode_limit > 1:
            modes = jnp.arange(1, mode_limit, dtype=jnp.int32)
            frequencies = 2 * modes
            updates = -1.0 / (4.0 * modes.astype(scalar_dtype) ** 2 - 1.0)
            coefficients = coefficients.at[frequencies].set(updates)
        coefficients = coefficients.at[num_intervals].set(
            -1.0 / (float(num_intervals * num_intervals) - 1.0)
        )

    transformed = _dct_type_one(coefficients)
    scale = jnp.asarray(float(num_intervals), dtype=scalar_dtype)
    weights = 2.0 * transformed / scale
    weights = weights.at[0].set(transformed[0] / scale)
    weights = weights.at[-1].set(transformed[-1] / scale)
    return 0.5 * weights


# 責務: level ごとの入れ子な Clenshaw-Curtis 則を device 配列で返す。
def _clenshaw_curtis_rule_device(
    level: int,
    /,
) -> tuple[Vector, Vector]:
    if level < 1:
        raise ValueError("level must be positive.")

    if level == 1:
        return (
            jnp.asarray([0.0], dtype=jnp.float64),
            jnp.asarray([1.0], dtype=jnp.float64),
        )

    num_intervals = 2 ** (level - 1)
    scalar_dtype = jnp.float64
    theta = jnp.pi * jnp.arange(num_intervals + 1, dtype=scalar_dtype) / float(num_intervals)
    nodes = 0.5 * jnp.cos(theta[::-1])
    weights = _clenshaw_curtis_weights_device(num_intervals)
    return nodes, weights


# 責務: level ごとの入れ子な Clenshaw-Curtis 則を [-0.5, 0.5] 上で返す。
def clenshaw_curtis_rule(level: int, /) -> tuple[Vector, Vector]:
    nodes, weights = _clenshaw_curtis_rule_device(level)
    return cast(
        tuple[Vector, Vector],
        (
            jnp.asarray(nodes, dtype=DEFAULT_DTYPE),
            jnp.asarray(weights, dtype=DEFAULT_DTYPE),
        ),
    )


# 責務: 入れ子な 1 次元積分則から差分積分則を device 上で構築する。
def _difference_rule_device(
    level: int,
    /,
) -> tuple[Vector, Vector]:
    nodes, weights = _clenshaw_curtis_rule_device(level)
    if level == 1:
        return nodes, weights

    previous_weights = _clenshaw_curtis_rule_device(level - 1)[1]
    if level == 2:
        overlap_indices = jnp.asarray([1], dtype=jnp.int32)
    else:
        overlap_indices = 2 * jnp.arange(previous_weights.shape[0], dtype=jnp.int32)
    diff_weights = weights.at[overlap_indices].add(-previous_weights)
    mask = jnp.abs(diff_weights) > 1e-15
    return nodes[mask], diff_weights[mask]


# 責務: Clenshaw-Curtis の差分積分則 Delta_level を構築する。
def difference_rule(level: int, /) -> tuple[Vector, Vector]:
    diff_nodes, diff_weights = _difference_rule_device(level)
    return cast(
        tuple[Vector, Vector],
        (
            jnp.asarray(diff_nodes, dtype=DEFAULT_DTYPE),
            jnp.asarray(diff_weights, dtype=DEFAULT_DTYPE),
        ),
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


def _resolve_index_dtype(index_dtype: str | None, /) -> np.dtype[Any] | None:
    if index_dtype in (None, "auto"):
        return None
    if index_dtype not in _SUPPORTED_INDEX_DTYPES:
        raise ValueError(f"index_dtype must be one of {sorted(_SUPPORTED_INDEX_DTYPES)} or None.")
    return _SUPPORTED_INDEX_DTYPES[index_dtype]


def _cast_index_array(
    array: NDArray[np.integer[Any]],
    index_dtype: np.dtype[Any] | None,
    /,
    *,
    name: str,
) -> NDArray[np.integer[Any]]:
    if index_dtype is None:
        return array
    if array.size == 0:
        return array.astype(index_dtype, copy=False)
    info = np.iinfo(index_dtype)
    min_value = int(np.min(array))
    max_value = int(np.max(array))
    if min_value < info.min or max_value > info.max:
        raise OverflowError(
            f"{name} values [{min_value}, {max_value}] do not fit in {index_dtype.name}."
        )
    return array.astype(index_dtype, copy=False)


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


def _term_generation_weights_numpy(
    dimension: int,
    dimension_weights: tuple[int, ...] | None,
    /,
) -> NDArray[np.int32]:
    if dimension_weights is None:
        return np.ones((dimension,), dtype=np.int32)
    return np.asarray(dimension_weights, dtype=np.int32)


# 責務: level ごとの差分則を flat storage と offset/length へまとめる。
def _difference_rule_storage_device(
    max_level: int,
    /,
) -> tuple[
    Vector,
    Vector,
    NDArray[np.int64],
    NDArray[np.int64],
]:
    nodes_by_level: list[Vector] = []
    weights_by_level: list[Vector] = []
    lengths = np.empty((max_level,), dtype=np.int64)

    for current_level in range(1, max_level + 1):
        nodes, weights = _difference_rule_device(current_level)
        nodes_by_level.append(nodes)
        weights_by_level.append(weights)
        lengths[current_level - 1] = int(nodes.shape[0])

    offsets = np.empty((max_level,), dtype=np.int64)
    total_length = 0
    for current_level, length in enumerate(lengths):
        offsets[current_level] = total_length
        total_length += int(length)

    nodes_storage = jnp.concatenate(nodes_by_level, axis=0)
    weights_storage = jnp.concatenate(weights_by_level, axis=0)
    return nodes_storage, weights_storage, offsets, lengths


def _count_smolyak_terms(
    dimension: int,
    level: int,
    /,
    *,
    dimension_weights: tuple[int, ...] | None = None,
) -> int:
    if dimension_weights is None:
        return comb(dimension + level - 1, dimension)
    budget = level - 1
    dp = [0] * (budget + 1)
    dp[0] = 1
    for weight in dimension_weights:
        next_dp = [0] * (budget + 1)
        for used_budget, count in enumerate(dp):
            if count == 0:
                continue
            max_extra = (budget - used_budget) // weight
            for extra_level in range(max_extra + 1):
                next_dp[used_budget + extra_level * weight] += count
        dp = next_dp
    return sum(dp)


def _count_evaluation_points(
    level: int,
    rule_lengths_np: NDArray[np.integer[Any]],
    generation_weights_np: NDArray[np.int32],
    /,
) -> int:
    budget = level - 1
    dp = [0] * (budget + 1)
    dp[0] = 1
    for weight in generation_weights_np:
        next_dp = [0] * (budget + 1)
        for used_budget, count in enumerate(dp):
            if count == 0:
                continue
            max_extra = (budget - used_budget) // int(weight)
            for extra_level in range(max_extra + 1):
                next_dp[used_budget + extra_level * int(weight)] += count * int(
                    rule_lengths_np[extra_level]
                )
        dp = next_dp
    return sum(dp)


def _max_suffix_points(
    level: int,
    rule_lengths_np: NDArray[np.integer[Any]],
    generation_weights_np: NDArray[np.int32],
    suffix_ndim: int,
    /,
) -> int:
    if suffix_ndim == 0:
        return 1
    budget = level - 1
    suffix_weights = generation_weights_np[-suffix_ndim:]
    dp = [-1] * (budget + 1)
    dp[0] = 1
    for weight in suffix_weights:
        next_dp = [-1] * (budget + 1)
        for used_budget, count in enumerate(dp):
            if count < 0:
                continue
            max_extra = (budget - used_budget) // int(weight)
            for extra_level in range(max_extra + 1):
                next_budget = used_budget + extra_level * int(weight)
                candidate = count * int(rule_lengths_np[extra_level])
                if candidate > next_dp[next_budget]:
                    next_dp[next_budget] = candidate
        dp = next_dp
    return max(dp)


# 責務: 1 次元差分則 storage と term plan の保持量を byte 単位で見積もる。
def _storage_bytes(
    rule_nodes: jax.Array,
    rule_weights: jax.Array,
    rule_offsets: jax.Array,
    rule_lengths: jax.Array,
    generation_weights: jax.Array,
    axis_level_ceilings: jax.Array,
    /,
) -> int:
    return (
        int(rule_nodes.nbytes)
        + int(rule_weights.nbytes)
        + int(rule_offsets.nbytes)
        + int(rule_lengths.nbytes)
        + int(generation_weights.nbytes)
        + int(axis_level_ceilings.nbytes)
    )


def _decode_points_and_weights(
    local_point_indices: jax.Array,
    axis_offsets: jax.Array,
    axis_lengths: jax.Array,
    rule_nodes: Vector,
    rule_weights: Vector,
    /,
) -> tuple[jax.Array, jax.Array]:
    axis_count = int(axis_offsets.shape[0])
    point_count = int(local_point_indices.shape[0])
    if axis_count == 0:
        return (
            jnp.zeros((0, point_count), dtype=rule_nodes.dtype),
            jnp.ones((point_count,), dtype=rule_weights.dtype),
        )
    axis_offsets = jnp.asarray(axis_offsets, dtype=jnp.int64)
    axis_lengths = jnp.asarray(axis_lengths, dtype=jnp.int64)
    local_point_indices = jnp.asarray(local_point_indices, dtype=jnp.int64)
    reversed_axis_offsets = lax.rev(axis_offsets, dimensions=(0,))
    reversed_axis_lengths = lax.rev(axis_lengths, dimensions=(0,))

    def decode_axis(
        carry: tuple[jax.Array, jax.Array],
        axis_data: tuple[jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        remaining_indices, accumulated_weights = carry
        axis_offset, axis_length = axis_data
        next_remaining_indices = lax.div(remaining_indices, axis_length)
        local_axis_indices = lax.rem(remaining_indices, axis_length)
        rule_indices = axis_offset + local_axis_indices
        axis_points = jnp.take(rule_nodes, rule_indices, mode="clip")
        axis_weights = jnp.take(rule_weights, rule_indices, mode="clip")
        return (next_remaining_indices, accumulated_weights * axis_weights), axis_points

    (_, point_weights), reversed_points = lax.scan(
        decode_axis,
        (
            local_point_indices,
            jnp.ones((point_count,), dtype=rule_weights.dtype),
        ),
        (
            reversed_axis_offsets,
            reversed_axis_lengths,
        ),
    )
    return lax.rev(reversed_points, dimensions=(0,)), point_weights


def _next_term_extra_levels(
    current_extra_levels: jax.Array,
    generation_weights: jax.Array,
    budget: int,
    /,
) -> tuple[jax.Array, jax.Array]:
    dimension = int(current_extra_levels.shape[0])
    extra_levels_i32 = current_extra_levels.astype(jnp.int32)

    def choose_axis_body(
        axis: int,
        carry: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        prefix_cost, chosen_axis = carry
        extra_level = extra_levels_i32[axis]
        axis_weight = generation_weights[axis]
        can_increment = prefix_cost + axis_weight * (extra_level + 1) <= budget
        next_prefix_cost = prefix_cost + axis_weight * extra_level
        next_chosen_axis = jnp.where(
            can_increment,
            jnp.asarray(axis, dtype=jnp.int32),
            chosen_axis,
        )
        return next_prefix_cost, next_chosen_axis

    _, chosen_axis = lax.fori_loop(
        0,
        dimension,
        choose_axis_body,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
        ),
    )
    has_next = chosen_axis >= 0
    safe_axis = jnp.where(has_next, chosen_axis, 0)
    incremented = current_extra_levels.at[safe_axis].add(
        jnp.asarray(1, dtype=current_extra_levels.dtype)
    )
    axis_indices = jnp.arange(dimension, dtype=jnp.int32)
    candidate_next_extra_levels = jnp.where(
        axis_indices <= safe_axis,
        incremented,
        jnp.zeros_like(current_extra_levels),
    )
    next_extra_levels = jnp.where(
        has_next,
        candidate_next_extra_levels,
        current_extra_levels,
    )
    return next_extra_levels, has_next


def _smolyak_plan_integral(
    f: Function,
    dimension: int,
    dtype: DTypeLike,
    rule_nodes: Vector,
    rule_weights: Vector,
    rule_offsets: jax.Array,
    rule_lengths: jax.Array,
    generation_weights: jax.Array,
    term_budget: int,
    *,
    chunk_size: int,
    batched_suffix_ndim: int,
    max_suffix_points: int,
) -> Vector:
    rule_offsets = jnp.asarray(rule_offsets, dtype=jnp.int64)
    rule_lengths = jnp.asarray(rule_lengths, dtype=jnp.int64)
    generation_weights = jnp.asarray(generation_weights, dtype=jnp.int32)
    zero_point = jnp.zeros((dimension,), dtype=dtype)
    initial_value = jnp.zeros_like(f(zero_point))
    suffix_ndim = batched_suffix_ndim
    prefix_ndim = dimension - suffix_ndim
    if max_suffix_points < 1:
        raise ValueError("max_suffix_points must be positive.")
    prefix_chunk_size = max(1, chunk_size // max_suffix_points)
    fixed_suffix_local_indices = jnp.arange(max_suffix_points, dtype=jnp.int64)
    fixed_prefix_local_indices = jnp.arange(prefix_chunk_size, dtype=jnp.int64)

    def term_body(state: tuple[Vector, jax.Array, jax.Array]) -> tuple[Vector, jax.Array, jax.Array]:
        acc, current_extra_levels, _ = state
        current_levels = current_extra_levels.astype(jnp.int32) + 1
        level_indices = current_levels - 1
        current_offsets = jnp.take(rule_offsets, level_indices, mode="clip")
        current_lengths = jnp.take(rule_lengths, level_indices, mode="clip")
        term_total_points = jnp.prod(current_lengths, dtype=jnp.int64)

        prefix_offsets = current_offsets[:prefix_ndim]
        prefix_lengths = current_lengths[:prefix_ndim]
        suffix_offsets = current_offsets[prefix_ndim:]
        suffix_lengths = current_lengths[prefix_ndim:]

        suffix_points_per_prefix = (
            jnp.asarray(1, dtype=jnp.int64)
            if suffix_ndim == 0
            else jnp.prod(suffix_lengths, dtype=jnp.int64)
        )
        prefix_point_count = lax.div(term_total_points, suffix_points_per_prefix)

        if suffix_ndim == 0:
            suffix_valid_mask = jnp.ones((1,), dtype=jnp.bool_)
            suffix_points = jnp.zeros((0, 1), dtype=rule_nodes.dtype)
            suffix_weights = jnp.ones((1,), dtype=rule_weights.dtype)
        else:
            suffix_valid_mask = fixed_suffix_local_indices < suffix_points_per_prefix
            suffix_points, suffix_point_weights = _decode_points_and_weights(
                fixed_suffix_local_indices,
                suffix_offsets,
                suffix_lengths,
                rule_nodes,
                rule_weights,
            )
            suffix_weights = suffix_point_weights * suffix_valid_mask.astype(rule_weights.dtype)

        def prefix_chunk_body(prefix_chunk_index: int, term_acc: Vector) -> Vector:
            prefix_start = jnp.asarray(prefix_chunk_index, dtype=jnp.int64) * prefix_chunk_size
            prefix_local_indices = prefix_start + fixed_prefix_local_indices
            prefix_valid_mask = prefix_local_indices < prefix_point_count
            prefix_points, prefix_point_weights = _decode_points_and_weights(
                prefix_local_indices,
                prefix_offsets,
                prefix_lengths,
                rule_nodes,
                rule_weights,
            )
            prefix_weights = prefix_point_weights * prefix_valid_mask.astype(rule_weights.dtype)

            if suffix_ndim == 0:
                values = jax.vmap(f, in_axes=1, out_axes=-1)(prefix_points)
                return term_acc + jnp.tensordot(values, prefix_weights, axes=(-1, 0))

            prefix_points_grid = jnp.broadcast_to(
                prefix_points[:, :, jnp.newaxis],
                (prefix_ndim, prefix_chunk_size, max_suffix_points),
            )
            suffix_points_grid = jnp.broadcast_to(
                suffix_points[:, jnp.newaxis, :],
                (suffix_ndim, prefix_chunk_size, max_suffix_points),
            )
            points_grid = jnp.concatenate([prefix_points_grid, suffix_points_grid], axis=0)

            weight_grid = prefix_weights[:, jnp.newaxis] * suffix_weights[jnp.newaxis, :]
            masked_weight_grid = weight_grid * (
                prefix_valid_mask[:, jnp.newaxis] & suffix_valid_mask[jnp.newaxis, :]
            ).astype(rule_weights.dtype)
            values = jax.vmap(
                lambda point_block: jax.vmap(f, in_axes=1, out_axes=-1)(point_block),
                in_axes=1,
                out_axes=-2,
            )(points_grid)
            return term_acc + jnp.tensordot(values, masked_weight_grid, axes=((-2, -1), (0, 1)))

        num_prefix_chunks = lax.div(
            prefix_point_count + prefix_chunk_size - 1,
            jnp.asarray(prefix_chunk_size, dtype=jnp.int64),
        )
        updated_acc = jax.lax.fori_loop(0, num_prefix_chunks, prefix_chunk_body, acc)
        next_extra_levels, has_next = _next_term_extra_levels(
            current_extra_levels,
            generation_weights,
            term_budget,
        )
        return updated_acc, next_extra_levels, has_next

    def cond_fun(state: tuple[Vector, jax.Array, jax.Array]) -> jax.Array:
        return state[2]

    initial_state = (
        initial_value,
        jnp.zeros((dimension,), dtype=jnp.uint8),
        jnp.asarray(True),
    )
    final_state = jax.lax.while_loop(cond_fun, term_body, initial_state)
    return final_state[0]


class SmolyakIntegrator(eqx.Module):
    dimension: int = eqx.field(static=True)
    level: int = eqx.field(static=True)
    prepared_level: int = eqx.field(static=True)
    dimension_weights: tuple[int, ...] | None = eqx.field(static=True)
    index_dtype_name: str = eqx.field(static=True)
    dtype: DTypeLike = eqx.field(static=True)
    chunk_size: int = eqx.field(static=True)
    batched_suffix_ndim: int = eqx.field(static=True)
    max_suffix_points: int = eqx.field(static=True)
    term_budget: int = eqx.field(static=True)
    rule_nodes: Vector
    rule_weights: Vector
    rule_offsets: jax.Array
    rule_lengths: jax.Array
    generation_weights: jax.Array
    axis_level_ceilings: jax.Array
    active_axis_count: int = eqx.field(static=True)
    num_terms: int = eqx.field(static=True)
    num_evaluation_points: int = eqx.field(static=True)
    storage_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        level: int,
        prepared_level: int | None = None,
        dimension_weights: tuple[int, ...] | None = None,
        index_dtype: str | None = None,
        dtype: DTypeLike = DEFAULT_DTYPE,
        chunk_size: int = 16384,
        batched_suffix_ndim: int = 0,
        _rule_storage: tuple[Vector, Vector, jax.Array, jax.Array] | None = None,
    ):
        if dimension < 1:
            raise ValueError("dimension must be positive.")
        if level < 1:
            raise ValueError("level must be positive.")
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if batched_suffix_ndim < 0 or batched_suffix_ndim > dimension:
            raise ValueError("batched_suffix_ndim must be between 0 and dimension.")
        resolved_prepared_level = level if prepared_level is None else prepared_level
        if resolved_prepared_level < level:
            raise ValueError("prepared_level must be greater than or equal to level.")
        if resolved_prepared_level > np.iinfo(np.uint8).max:
            raise ValueError("prepared_level must be at most 255 for uint8 term generation.")

        self.dimension = dimension
        self.level = level
        self.prepared_level = resolved_prepared_level
        self.dimension_weights = _normalize_dimension_weights(dimension, dimension_weights)
        resolved_index_dtype = _resolve_index_dtype(index_dtype)
        self.index_dtype_name = "auto" if resolved_index_dtype is None else resolved_index_dtype.name
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.batched_suffix_ndim = batched_suffix_ndim
        self.term_budget = level - 1
        axis_level_ceilings_np = _axis_level_ceilings_numpy(
            dimension,
            level,
            dimension_weights=self.dimension_weights,
        )
        if int(np.max(axis_level_ceilings_np)) > np.iinfo(np.uint8).max:
            raise ValueError("axis level ceilings must fit in uint8.")
        self.axis_level_ceilings = jnp.asarray(axis_level_ceilings_np, dtype=jnp.int32)
        self.active_axis_count = int(np.count_nonzero(axis_level_ceilings_np > 1))
        generation_weights_np = _term_generation_weights_numpy(
            dimension,
            self.dimension_weights,
        )
        self.generation_weights = jnp.asarray(generation_weights_np, dtype=jnp.int32)
        if _rule_storage is None:
            max_rule_level = _max_difference_rule_level(
                dimension,
                resolved_prepared_level,
                dimension_weights=self.dimension_weights,
            )
            (
                rule_nodes,
                rule_weights,
                rule_offsets_np,
                rule_lengths_np,
            ) = _difference_rule_storage_device(max_rule_level)
            rule_offsets_np = _cast_index_array(rule_offsets_np, resolved_index_dtype, name="rule_offsets")
            rule_lengths_np = _cast_index_array(rule_lengths_np, resolved_index_dtype, name="rule_lengths")
            self.rule_nodes = jnp.asarray(rule_nodes, dtype=dtype)
            self.rule_weights = jnp.asarray(rule_weights, dtype=dtype)
            self.rule_offsets = jnp.asarray(rule_offsets_np)
            self.rule_lengths = jnp.asarray(rule_lengths_np)
        else:
            (
                self.rule_nodes,
                self.rule_weights,
                self.rule_offsets,
            self.rule_lengths,
            ) = _rule_storage
            rule_offsets_np = np.asarray(self.rule_offsets)
            rule_lengths_np = np.asarray(self.rule_lengths)

        self.num_terms = _count_smolyak_terms(
            dimension,
            level,
            dimension_weights=self.dimension_weights,
        )
        self.num_evaluation_points = _count_evaluation_points(
            level,
            rule_lengths_np,
            generation_weights_np,
        )
        self.max_suffix_points = _max_suffix_points(
            level,
            rule_lengths_np,
            generation_weights_np,
            batched_suffix_ndim,
        )
        self.storage_bytes = _storage_bytes(
            self.rule_nodes,
            self.rule_weights,
            self.rule_offsets,
            self.rule_lengths,
            self.generation_weights,
            self.axis_level_ceilings,
        )

    def integrate(self, f: Function, /) -> Vector:
        # 実験コード側で JIT を適用する前提にして、ここでは Python callable の
        # トレース失敗を避けるため module 側の jitting は行わない。
        return _smolyak_plan_integral(
            f,
            self.dimension,
            self.dtype,
            self.rule_nodes,
            self.rule_weights,
            self.rule_offsets,
            self.rule_lengths,
            self.generation_weights,
            self.term_budget,
            chunk_size=self.chunk_size,
            batched_suffix_ndim=self.batched_suffix_ndim,
            max_suffix_points=self.max_suffix_points,
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
                index_dtype=self.index_dtype_name,
                dtype=self.dtype,
                chunk_size=self.chunk_size,
                batched_suffix_ndim=self.batched_suffix_ndim,
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
            index_dtype=self.index_dtype_name,
            dtype=self.dtype,
            chunk_size=self.chunk_size,
            batched_suffix_ndim=self.batched_suffix_ndim,
        )


# 責務: plan 化された Smolyak 積分器を初期化して返す。
def initialize_smolyak_integrator(
    dimension: int,
    level: int,
    *,
    prepared_level: int | None = None,
    dimension_weights: tuple[int, ...] | None = None,
    index_dtype: str | None = None,
    dtype: DTypeLike = DEFAULT_DTYPE,
    chunk_size: int = 16384,
    batched_suffix_ndim: int = 0,
) -> SmolyakIntegrator:
    return SmolyakIntegrator(
        dimension=dimension,
        level=level,
        prepared_level=prepared_level,
        dimension_weights=dimension_weights,
        index_dtype=index_dtype,
        dtype=dtype,
        chunk_size=chunk_size,
        batched_suffix_ndim=batched_suffix_ndim,
    )


__all__ = [
    "SmolyakIntegrator",
    "clenshaw_curtis_rule",
    "difference_rule",
    "initialize_smolyak_integrator",
    "multi_indices",
]
