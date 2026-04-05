"""Build and solve the toy suffix stationary problem."""

from __future__ import annotations

from collections.abc import Callable

import jax
from jax import numpy as jnp

from ...base import Matrix, Scalar, Vector
from ..layer_utils import StandardCarry
from ..neuralnetwork import NeuralNetwork
from .constraints import backward_multipliers
from .lagrangian import packed_kkt_residual, residual_breakdown
from .parameterization import (
    build_layer_vectorizations,
    build_variable_layout,
    flatten_layer_params,
    pack_variables,
    unpack_variables,
)
from .prefix import build_prefix_tape
from .types import (
    PrefixTape,
    StationaryProblem,
    StationarySolveInfo,
    StationarySolveState,
    SuffixVariables,
)


def _current_suffix_outputs(
    network: NeuralNetwork,
    prefix_tape: PrefixTape,
) -> tuple[Matrix, ...]:
    """Read the current suffix outputs from the network."""
    outputs: list[Matrix] = []
    current = prefix_tape.z_prefix
    for layer in network.layers[prefix_tape.layer_index :]:
        current = layer(StandardCarry(z=current), prefix_tape.ctx).z
        outputs.append(current)
    return tuple(outputs)


def build_stationary_problem(
    network: NeuralNetwork,
    x: Matrix,
    y: Matrix,
    layer_index: int,
    loss_from_output: Callable[[Matrix, Matrix], Scalar],
) -> StationaryProblem:
    """Build one suffix stationary problem from a network and data."""
    prefix_tape = build_prefix_tape(network=network, x=x, layer_index=layer_index)
    suffix_layers = tuple(network.layers[layer_index:])
    if not suffix_layers:
        raise ValueError("layer_index must leave at least one suffix layer.")

    raw_vectorizations = tuple(flatten_layer_params(layer) for layer in suffix_layers)
    initial_theta_tail = tuple(flat_params for flat_params, _ in raw_vectorizations)
    initial_z_tail = _current_suffix_outputs(network=network, prefix_tape=prefix_tape)
    output_shapes = tuple(tuple(int(dim) for dim in z.shape) for z in initial_z_tail)
    vectorizations = build_layer_vectorizations(layers=suffix_layers, output_shapes=output_shapes)
    layout = build_variable_layout(
        param_sizes=tuple(vectorization.param_size for vectorization in vectorizations),
        output_shapes=output_shapes,
    )
    return StationaryProblem(
        prefix_tape=prefix_tape,
        layer_vectorizations=vectorizations,
        layout=layout,
        target=y,
        loss_from_output=loss_from_output,
        initial_theta_tail=initial_theta_tail,
        initial_z_tail=initial_z_tail,
    )


def initialize_suffix_variables(problem: StationaryProblem) -> SuffixVariables:
    """Return warm-start variables from the current rollout."""
    p_tail = backward_multipliers(
        theta_tail=problem.initial_theta_tail,
        z_tail=problem.initial_z_tail,
        prefix_tape=problem.prefix_tape,
        vectorizations=problem.layer_vectorizations,
        loss_from_output=problem.loss_from_output,
        target=problem.target,
    )
    return SuffixVariables(
        theta_tail=problem.initial_theta_tail,
        z_tail=problem.initial_z_tail,
        p_tail=p_tail,
    )


def _dense_newton_step(
    problem: StationaryProblem,
    state: StationarySolveState,
) -> StationarySolveState:
    """Advance one Newton step with a dense Jacobian."""
    residual = packed_kkt_residual(state.packed_variables, problem)

    def _residual_from_vector(vec: Vector) -> Vector:
        return packed_kkt_residual(vec, problem)

    jacobian = jax.jacobian(_residual_from_vector)(state.packed_variables)
    eye = jnp.eye(jacobian.shape[0], dtype=jacobian.dtype)
    delta = jnp.linalg.solve(jacobian + 1.0e-6 * eye, -residual)

    best_vector = state.packed_variables
    best_norm = float(jnp.linalg.norm(residual))
    step_scale = 1.0
    for _ in range(8):
        trial = state.packed_variables + step_scale * delta
        trial_norm = float(jnp.linalg.norm(packed_kkt_residual(trial, problem)))
        if trial_norm < best_norm:
            best_vector = trial
            best_norm = trial_norm
            break
        step_scale *= 0.5

    return StationarySolveState(
        packed_variables=best_vector,
        iteration=state.iteration + 1,
        residual_norm=jnp.asarray(best_norm),
    )


def solve_stationary_suffix(
    problem: StationaryProblem,
    init_variables: SuffixVariables,
    *,
    tol: Scalar,
    maxiter: int,
    method: str = "dense_newton",
) -> tuple[SuffixVariables, StationarySolveInfo]:
    """Solve the suffix stationary problem with dense Newton."""
    if method != "dense_newton":
        raise ValueError(f"Unsupported method: {method}")

    packed = pack_variables(init_variables, problem.layout)
    initial_norm = jnp.linalg.norm(packed_kkt_residual(packed, problem))
    state = StationarySolveState(
        packed_variables=packed,
        iteration=0,
        residual_norm=initial_norm,
    )

    while state.iteration < maxiter and float(state.residual_norm) > float(tol):
        next_state = _dense_newton_step(problem=problem, state=state)
        if float(next_state.residual_norm) >= float(state.residual_norm):
            break
        state = next_state

    solution = unpack_variables(state.packed_variables, problem.layout)
    info = StationarySolveInfo(
        converged=float(state.residual_norm) <= float(tol),
        iterations=state.iteration,
        residual_norm=float(state.residual_norm),
        residual_breakdown=residual_breakdown(solution, problem),
    )
    return solution, info


def extract_layer_update(solution: SuffixVariables, layer_offset: int = 0) -> Vector:
    """Return the parameter update for one layer inside the solved suffix."""
    if layer_offset < 0 or layer_offset >= len(solution.theta_tail):
        raise ValueError("layer_offset is out of range.")
    return solution.theta_tail[layer_offset]


__all__ = [
    "build_stationary_problem",
    "initialize_suffix_variables",
    "solve_stationary_suffix",
    "extract_layer_update",
]
