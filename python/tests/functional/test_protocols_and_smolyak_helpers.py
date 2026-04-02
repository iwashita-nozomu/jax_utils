from __future__ import annotations

import os

from jax_util.xla_env import build_cpu_env, build_gpu_env

for _key, _value in {
    **build_cpu_env(),
    **build_gpu_env(disable_preallocation=True),
}.items():
    os.environ.setdefault(_key, _value)

import jax.numpy as jnp
import numpy as np
import pytest

from jax_util.functional.protocols import Func
from jax_util.functional.smolyak import (
    _clenshaw_curtis_node_keys,
    _clenshaw_curtis_nodes_from_keys,
    _clenshaw_curtis_rule_numpy,
    _clenshaw_curtis_weights,
    _compact_unsigned_dtype,
    _difference_rule_numpy,
    _difference_rule_storage_numpy,
    _max_difference_rule_level,
    multi_indices,
)


def test_func_supports_composition_and_pointwise_products() -> None:
    double = Func(lambda x: 2.0 * x)
    shift = Func(lambda x: x + 1.0)
    x = jnp.array([1.0, 2.0])

    assert jnp.allclose((double @ shift)(x), jnp.array([4.0, 6.0]))
    assert jnp.allclose((double * shift)(x), jnp.array([4.0, 12.0]))


def test_compact_unsigned_dtype_scales_with_required_range() -> None:
    assert _compact_unsigned_dtype(1) == np.dtype(np.uint8)
    assert _compact_unsigned_dtype(np.iinfo(np.uint8).max) == np.dtype(np.uint8)
    assert _compact_unsigned_dtype(np.iinfo(np.uint8).max + 1) == np.dtype(np.uint16)


def test_clenshaw_curtis_node_keys_round_trip_to_nodes() -> None:
    with pytest.raises(ValueError, match="positive"):
        _clenshaw_curtis_node_keys(0)

    keys = _clenshaw_curtis_node_keys(3)
    nodes = _clenshaw_curtis_nodes_from_keys(keys)
    expected_nodes, _ = _clenshaw_curtis_rule_numpy(3)

    assert keys.shape == (5, 2)
    assert np.allclose(nodes, expected_nodes)


def test_clenshaw_curtis_weights_are_normalized_and_symmetric() -> None:
    weights = _clenshaw_curtis_weights(4)
    assert np.allclose(np.sum(weights), 1.0)
    assert np.allclose(weights, weights[::-1])


def test_difference_rule_numpy_matches_known_level_two_values() -> None:
    nodes, weights = _difference_rule_numpy(2)
    assert np.allclose(nodes, np.array([-0.5, 0.0, 0.5]))
    assert np.allclose(weights, np.array([1.0 / 6.0, -1.0 / 3.0, 1.0 / 6.0]))


def test_difference_rule_storage_numpy_tracks_offsets_and_lengths() -> None:
    nodes_storage, weights_storage, offsets, lengths = _difference_rule_storage_numpy(3)

    assert offsets.shape == (3,)
    assert lengths.shape == (3,)
    assert int(offsets[0]) == 0

    for level in range(1, 4):
        expected_nodes, expected_weights = _difference_rule_numpy(level)
        start = int(offsets[level - 1])
        stop = start + int(lengths[level - 1])
        assert np.allclose(nodes_storage[start:stop], expected_nodes)
        assert np.allclose(weights_storage[start:stop], expected_weights)


def test_multi_indices_and_rule_level_helpers_cover_edge_cases() -> None:
    with pytest.raises(ValueError, match="positive"):
        multi_indices(0, 1)

    assert multi_indices(3, 2).shape == (0, 3)
    assert np.array_equal(
        multi_indices(2, 3),
        np.array([[1, 1], [1, 2], [2, 1]], dtype=np.uint8),
    )
    assert _max_difference_rule_level(3, 4) == 6
