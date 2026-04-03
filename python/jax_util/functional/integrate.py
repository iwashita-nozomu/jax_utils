from __future__ import annotations

from typing import Any

import numpy as np
import jax
from jax import lax
from jax.custom_batching import custom_vmap
from jax import tree_util

from ..base import Vector
from .protocols import Function, Integrator


# 責務: 被積分関数を指定した積分器へ委譲して積分値を返す。
def _integrate_impl(f: Function, integrator: Integrator, /) -> Vector:
    return integrator.integrate(f)


def _tree_any(mask: Any, /) -> bool:
    return any(bool(leaf) for leaf in tree_util.tree_leaves(mask))


def _tree_split(mask: Any, tree: Any, /) -> tuple[Any, Any]:
    mapped_tree = tree_util.tree_map(lambda batched, x: x if batched else None, mask, tree)
    broadcast_tree = tree_util.tree_map(lambda batched, x: None if batched else x, mask, tree)
    return mapped_tree, broadcast_tree


def _tree_merge(mask: Any, mapped_tree: Any, broadcast_tree: Any, /) -> Any:
    return tree_util.tree_map(
        lambda batched, mapped_leaf, broadcast_leaf: mapped_leaf if batched else broadcast_leaf,
        mask,
        mapped_tree,
        broadcast_tree,
    )


def _first_mapped_tree(mask: Any, tree: Any, /) -> Any:
    return tree_util.tree_map(
        lambda batched, x: lax.index_in_dim(x, 0, keepdims=False) if batched else None,
        mask,
        tree,
    )


def _pytree_nbytes(tree: Any, /) -> int:
    total = 0
    for leaf in tree_util.tree_leaves(tree):
        shape = getattr(leaf, "shape", None)
        dtype = getattr(leaf, "dtype", None)
        if shape is None or dtype is None:
            continue
        total += int(np.prod(shape, dtype=np.int64)) * int(np.dtype(dtype).itemsize)
    return total


def _has_array_like_leaves(tree: Any, /) -> bool:
    for leaf in tree_util.tree_leaves(tree):
        if getattr(leaf, "shape", None) is None:
            continue
        if getattr(leaf, "dtype", None) is None:
            continue
        return True
    return False


def _device_memory_limit_bytes() -> int | None:
    for device in jax.devices():
        memory_stats = getattr(device, "memory_stats", None)
        if memory_stats is None:
            continue
        stats = memory_stats()
        if isinstance(stats, dict):
            bytes_limit = stats.get("bytes_limit")
            if isinstance(bytes_limit, int) and bytes_limit > 0:
                return int(bytes_limit)
    return None


def _auto_problem_batch_tile_size(
    axis_size: int,
    f_batched: Any,
    f: Function,
    integrator: Integrator,
    /,
) -> int:
    mapped_f, broadcast_f = _tree_split(f_batched, f)
    sample_mapped_f = _first_mapped_tree(f_batched, f)
    sample_f = _tree_merge(f_batched, sample_mapped_f, broadcast_f)
    sample_output = _integrate_impl(sample_f, integrator)

    output_nbytes = max(_pytree_nbytes(sample_output), 1)
    chunk_size = max(1, int(getattr(integrator, "chunk_size", 1)))
    per_problem_bytes = max(output_nbytes * chunk_size, output_nbytes)

    memory_limit = _device_memory_limit_bytes()
    if memory_limit is None:
        target_bytes = 16 * 1024 * 1024
    else:
        target_bytes = max(
            4 * 1024 * 1024,
            min(16 * 1024 * 1024, memory_limit // 1024),
        )

    tile_size = max(1, target_bytes // per_problem_bytes)
    return max(1, min(int(axis_size), int(tile_size)))


@custom_vmap
def _integrate_batched(f: Function, integrator: Integrator, /) -> Vector:
    return _integrate_impl(f, integrator)


@_integrate_batched.def_vmap
def _integrate_vmap_rule(axis_size: int, in_batched: Any, f: Function, integrator: Integrator) -> tuple[Vector, Any]:
    f_batched, integrator_batched = in_batched

    if _tree_any(integrator_batched):
        raise NotImplementedError("integrate custom_vmap supports batching over f only.")

    if not _tree_any(f_batched):
        out = _integrate_impl(f, integrator)
        out_batched = tree_util.tree_map(lambda _: False, out)
        return out, out_batched

    mapped_f, broadcast_f = _tree_split(f_batched, f)
    tile_size = _auto_problem_batch_tile_size(axis_size, f_batched, f, integrator)

    def integrate_one(mapped_single_f: Any) -> Vector:
        single_f = _tree_merge(f_batched, mapped_single_f, broadcast_f)
        return _integrate_impl(single_f, integrator)

    out = lax.map(integrate_one, mapped_f, batch_size=tile_size)
    out_batched = tree_util.tree_map(lambda _: True, out)
    return out, out_batched


def integrate(f: Function, integrator: Integrator, /) -> Vector:
    # `custom_vmap` requires pytree leaves that JAX can batch; keep plain callables on
    # the original direct path and only use the batched wrapper for array-backed modules.
    if _has_array_like_leaves(f):
        return _integrate_batched(f, integrator)
    return _integrate_impl(f, integrator)


__all__ = [
    "integrate",
]
