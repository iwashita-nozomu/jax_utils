"""Experimental DF=0 stationary-update training package."""

from .constraints import adjoint_residual, primal_residual, rollout_suffix, theta_residual
from .lagrangian import (
    kkt_residual,
    kkt_residual_terms,
    local_lagrangian,
    residual_breakdown,
)
from .parameterization import (
    build_layer_vectorizations,
    build_variable_layout,
    pack_variables,
    rebuild_layer,
    unpack_variables,
)
from .prefix import build_prefix_tape, suffix_input
from .stationary_solve import (
    build_stationary_problem,
    extract_layer_update,
    initialize_suffix_variables,
    solve_stationary_suffix,
)
from .toy import build_standard_identity_network, mean_squared_output_loss
from .types import (
    BlockSlice,
    LayerVectorization,
    PrefixTape,
    StationaryProblem,
    StationarySolveInfo,
    StationarySolveState,
    SuffixVariables,
    VariableLayout,
)

__all__ = [
    "LayerVectorization",
    "PrefixTape",
    "BlockSlice",
    "VariableLayout",
    "SuffixVariables",
    "StationaryProblem",
    "StationarySolveState",
    "StationarySolveInfo",
    "build_layer_vectorizations",
    "build_variable_layout",
    "rebuild_layer",
    "pack_variables",
    "unpack_variables",
    "build_prefix_tape",
    "suffix_input",
    "rollout_suffix",
    "primal_residual",
    "adjoint_residual",
    "theta_residual",
    "local_lagrangian",
    "kkt_residual_terms",
    "kkt_residual",
    "residual_breakdown",
    "build_stationary_problem",
    "initialize_suffix_variables",
    "solve_stationary_suffix",
    "extract_layer_update",
    "mean_squared_output_loss",
    "build_standard_identity_network",
]
