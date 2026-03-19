from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import jax_util.functional.smolyak as smolyak_module
from jax_util.functional.protocols import Func
from jax_util.functional.smolyak import (
    _as_numpy_vector,
    _clenshaw_curtis_rule_numpy,
    _difference_rule_numpy,
    _lexsort_points,
    _rule_node_codec,
    _rule_numpy_builder,
    _tensor_difference_rule_ids_numpy,
    _tensor_difference_rule_numpy,
    _trapezoidal_rule_numpy,
    clenshaw_curtis_node_ids,
    clenshaw_curtis_nodes_from_ids,
    difference_rule,
    multi_indices,
    smolyak_grid,
    trapezoidal_node_ids,
    trapezoidal_nodes_from_ids,
    trapezoidal_rule,
)


def _custom_nested_rule(level: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    if level == 1:
        return jnp.array([0.0]), jnp.array([1.0])
    if level == 2:
        return jnp.array([-0.5, 0.0, 0.5]), jnp.array([0.25, 0.5, 0.25])
    raise ValueError("unsupported level for custom test rule")


def test_func_supports_composition_and_pointwise_products() -> None:
    double = Func(lambda x: 2.0 * x)
    shift = Func(lambda x: x + 1.0)
    x = jnp.array([1.0, 2.0])

    assert jnp.allclose((double @ shift)(x), jnp.array([4.0, 6.0]))
    assert jnp.allclose((double * shift)(x), jnp.array([4.0, 12.0]))


def test_smolyak_node_codecs_and_rule_builders_cover_known_and_unknown_rules() -> None:
    with pytest.raises(ValueError, match="positive"):
        clenshaw_curtis_node_ids(0)
    with pytest.raises(ValueError, match="positive"):
        trapezoidal_node_ids(0)
    with pytest.raises(ValueError, match="positive"):
        _clenshaw_curtis_rule_numpy(0)
    with pytest.raises(ValueError, match="positive"):
        _trapezoidal_rule_numpy(0)

    cc_ids = clenshaw_curtis_node_ids(3)
    trap_ids_level_1 = trapezoidal_node_ids(1)
    trap_ids = trapezoidal_node_ids(3)
    cc_nodes_expected, _ = _clenshaw_curtis_rule_numpy(3)
    trap_level_1_nodes, trap_level_1_weights = _trapezoidal_rule_numpy(1)
    trap_nodes_expected, _ = _trapezoidal_rule_numpy(3)

    assert np.allclose(clenshaw_curtis_nodes_from_ids(cc_ids), cc_nodes_expected)
    assert np.array_equal(trap_ids_level_1, np.array([3]))
    assert np.allclose(trap_level_1_nodes, np.array([0.0]))
    assert np.allclose(trap_level_1_weights, np.array([1.0]))
    assert np.allclose(trapezoidal_nodes_from_ids(trap_ids), trap_nodes_expected)
    assert _rule_node_codec(trapezoidal_rule) is not None
    assert _rule_node_codec(_custom_nested_rule) is None
    assert _rule_numpy_builder(trapezoidal_rule) is _trapezoidal_rule_numpy
    assert _rule_numpy_builder(_custom_nested_rule) is None


def test_difference_rule_numpy_supports_non_codec_rules() -> None:
    node_ids, nodes_level_1, weights_level_1 = _difference_rule_numpy(1, _custom_nested_rule)
    assert node_ids is None
    assert np.allclose(nodes_level_1, np.array([0.0]))
    assert np.allclose(weights_level_1, np.array([1.0]))

    _, diff_nodes, diff_weights = _difference_rule_numpy(2, _custom_nested_rule)
    wrapped_nodes, wrapped_weights = difference_rule(2, _custom_nested_rule)

    assert np.allclose(diff_nodes, np.array([-0.5, 0.0, 0.5]))
    assert np.allclose(diff_weights, np.array([0.25, -0.5, 0.25]))
    assert jnp.allclose(wrapped_nodes, jnp.asarray(diff_nodes))
    assert jnp.allclose(wrapped_weights, jnp.asarray(diff_weights))


def test_difference_rule_numpy_covers_codec_and_builder_fallback_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_rule_numpy_builder = smolyak_module._rule_numpy_builder
    original_rule_node_codec = smolyak_module._rule_node_codec

    monkeypatch.setattr(
        smolyak_module,
        "_rule_numpy_builder",
        lambda rule: (
            None if rule is smolyak_module.trapezoidal_rule else original_rule_numpy_builder(rule)
        ),
    )
    node_ids, _, diff_weights = smolyak_module._difference_rule_numpy(
        2, smolyak_module.trapezoidal_rule
    )
    assert node_ids is not None
    assert np.allclose(diff_weights, np.array([0.25, -0.5, 0.25]))

    monkeypatch.setattr(smolyak_module, "_rule_numpy_builder", original_rule_numpy_builder)
    monkeypatch.setattr(
        smolyak_module,
        "_rule_node_codec",
        lambda rule: (
            None if rule is smolyak_module.trapezoidal_rule else original_rule_node_codec(rule)
        ),
    )
    node_ids, diff_nodes, diff_weights = smolyak_module._difference_rule_numpy(
        2, smolyak_module.trapezoidal_rule
    )
    assert node_ids is None
    assert np.allclose(diff_nodes, np.array([-0.5, 0.0, 0.5]))
    assert np.allclose(diff_weights, np.array([0.25, -0.5, 0.25]))


def test_tensor_helpers_and_multi_indices_cover_filtering_and_edge_cases() -> None:
    assert np.allclose(_as_numpy_vector(jnp.array([1.0, 2.0])), np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="positive"):
        multi_indices(0, 1)
    assert multi_indices(3, 2).shape == (0, 3)

    points, weights = _tensor_difference_rule_numpy(
        [np.array([0.0, 1.0]), np.array([0.0, 1.0])],
        [np.array([1.0, 0.0]), np.array([2.0, 3.0])],
    )
    point_ids, id_weights = _tensor_difference_rule_ids_numpy(
        [np.array([1, 2]), np.array([3, 4])],
        [np.array([1.0, 0.0]), np.array([2.0, 3.0])],
    )
    order = _lexsort_points(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))

    assert points.shape == (2, 2)
    assert np.allclose(weights, np.array([2.0, 3.0]))
    assert np.array_equal(point_ids, np.array([[1, 3], [1, 4]]))
    assert np.allclose(id_weights, np.array([2.0, 3.0]))
    assert np.array_equal(order, np.array([2, 1, 0]))


def test_smolyak_grid_supports_custom_rules_and_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        smolyak_grid(0, 2)
    with pytest.raises(ValueError, match="positive"):
        smolyak_grid(2, 0)

    trap_nodes, trap_weights = trapezoidal_rule(2)
    assert trap_nodes.shape == (3,)
    assert jnp.allclose(jnp.sum(trap_weights), jnp.asarray(1.0))

    points, weights = smolyak_grid(2, 2, rule=_custom_nested_rule)

    assert points.shape[0] == 2
    assert bool(jnp.all(points >= -0.5))
    assert bool(jnp.all(points <= 0.5))
    assert jnp.allclose(jnp.sum(weights), jnp.asarray(1.0))


def _run_all_tests() -> None:
    """全テストを実行します。

    補助的なpython file.py実行時に使用されます。
    pytest -s python/tests/functional/test_protocols_and_smolyak_helpers.py
    と同等の実行が可能になります。
    """
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    _run_all_tests()
