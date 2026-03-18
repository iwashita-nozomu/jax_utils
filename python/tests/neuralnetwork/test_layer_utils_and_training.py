from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import pytest

try:
    from jax_util.neuralnetwork.layer_utils import (
        IcnnCtx,
        IcnnLayer,
        StandardCarry,
        StandardCtx,
        module_to_vector,
        standardNN_layer_factory,
        vector_to_module,
    )
    from jax_util.neuralnetwork.neuralnetwork import (
        build_neuralnetwork,
        forward_with_cache,
        state_initializer,
    )
    from jax_util.neuralnetwork.sequential_train import sequential_train_step
    from jax_util.neuralnetwork.train import train_loop
except (ModuleNotFoundError, TypeError) as exc:
    if "NamedTuple" in str(exc) or "optimizers.protocols" in str(exc):
        pytest.skip(f"neuralnetwork module is not importable in this environment: {exc}", allow_module_level=True)
    raise


def test_layer_factories_and_module_roundtrip_preserve_forward_values() -> None:
    key = jax.random.PRNGKey(0)
    layer, next_key = standardNN_layer_factory(2, 3, jnp.tanh, key)
    flat_params, rebuild_state = module_to_vector(layer)
    rebuilt = vector_to_module(flat_params, rebuild_state)

    carry = StandardCarry(z=jnp.ones((2, 4)))
    original_output = layer(carry, StandardCtx()).z
    rebuilt_output = rebuilt(carry, StandardCtx()).z

    assert flat_params.ndim == 1
    assert not bool(jnp.array_equal(key, next_key))
    assert jnp.allclose(original_output, rebuilt_output)

    icnn_layer, _ = IcnnLayer, None
    built_icnn_layer, _ = __import__("jax_util.neuralnetwork.layer_utils", fromlist=["icnn_layer_factory"]).icnn_layer_factory(2, 3, 2, jax.nn.softplus, key)
    icnn_output = built_icnn_layer(carry=__import__("jax_util.neuralnetwork.layer_utils", fromlist=["IcnnCarry"]).IcnnCarry(z=jnp.ones((2, 4))), ctx=IcnnCtx(x=jnp.ones((2, 4))))
    assert isinstance(icnn_layer, type)
    assert bool(jnp.all(built_icnn_layer.W >= 0.0))
    assert icnn_output.z.shape == (3, 4)


def test_state_initializer_build_errors_and_forward_cache_cover_remaining_branches() -> None:
    x = jnp.ones((2, 3))

    with pytest.raises(ValueError, match="at least 2 elements"):
        state_initializer("icnn", x, None)
    with pytest.raises(ValueError, match="Unsupported network type"):
        state_initializer("unknown", x, (2, 3))
    with pytest.raises(ValueError, match="Unsupported activation"):
        build_neuralnetwork("standard", (2, 2, 1), "unsupported", jax.random.PRNGKey(1))
    with pytest.raises(ValueError, match="Unsupported network type"):
        build_neuralnetwork("unknown", (2, 2, 1), "relu", jax.random.PRNGKey(1))

    model = build_neuralnetwork("standard", (2, 3, 1), "identity", jax.random.PRNGKey(2))
    y, carries, ctx = forward_with_cache(x, model)

    assert jnp.allclose(y, model(x))
    assert len(carries) == len(model.layers)
    assert isinstance(ctx, StandardCtx)


def test_train_loop_updates_parameters_and_sequential_train_step_is_explicitly_unimplemented() -> None:
    model = build_neuralnetwork("standard", (2, 2, 1), "identity", jax.random.PRNGKey(3))
    params, static = eqx.partition(model, eqx.is_inexact_array)
    x = jnp.ones((2, 8))
    y = jnp.ones((1, 8))
    batch = jnp.vstack([x, y])

    def loss_fn(current_params: object, current_batch: jnp.ndarray) -> jnp.ndarray:
        network = eqx.combine(current_params, static)
        prediction = network(current_batch[:2, :])
        return jnp.mean((prediction - current_batch[2:3, :]) ** 2)

    optimizer = optax.sgd(0.1)
    opt_state = optimizer.init(params)

    new_params, _ = train_loop(
        params=params,
        batch=batch,
        optimizer=optimizer,
        opt_state=opt_state,
        loss_fn=loss_fn,
        num_steps=3,
    )

    before = jnp.concatenate([leaf.reshape(-1) for leaf in jax.tree_util.tree_leaves(params)])
    after = jnp.concatenate([leaf.reshape(-1) for leaf in jax.tree_util.tree_leaves(new_params)])

    assert not bool(jnp.allclose(before, after))

    class DummyObjective(eqx.Module):
        objective: object = eqx.field(static=True)

    def squared_tree_norm(value: object) -> jnp.ndarray:
        leaves = jax.tree_util.tree_leaves(value)
        return sum(jnp.sum(jnp.square(leaf)) for leaf in leaves)

    objective = DummyObjective(objective=squared_tree_norm)
    with pytest.raises(NotImplementedError, match="experimental sequential training path is not implemented"):
        sequential_train_step(
            model=model,
            trainers=(),
            x=x,
            optim=objective,
        )


def _run_all_tests() -> None:
    """全テストを実行します。
    
    補助的なpython file.py実行時に使用されます。
    pytest -s python/tests/neuralnetwork/test_layer_utils_and_training.py
    と同等の実行が可能になります。
    """
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    _run_all_tests()
