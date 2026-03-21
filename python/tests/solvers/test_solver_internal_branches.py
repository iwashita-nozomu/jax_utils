from __future__ import annotations

import json
import runpy
import warnings

import jax
import jax.numpy as jnp
import pytest

from jax_util.base import DEFAULT_DTYPE, LinOp
from jax_util.solvers._check_mv_operator import (
    check_self_adjoint,
    check_spd_quadratic_form,
    print_Mv_report,
)
from jax_util.solvers._minres import MINRESState, minres_solve
# from jax_util.solvers._test_jax は移動済み（python/tests/solvers/test_jax_debug.py）
# import jax_util.solvers._test_jax as test_jax_module
from jax_util.solvers.archive._fgmres import gmres_solve
from jax_util.solvers.kkt_solver import KKTState, _kkt_block_solver, initialize_kkt_state
from jax_util.solvers.lobpcg import (
    BlockEigenState,
    SubspaceBasis,
    apply_projection,
    init_block_eigen_state,
    init_spectral_precond,
    update_subspace,
)
from jax_util.solvers.slq import _make_probe_vectors


def _diag_op(diagonal: jnp.ndarray) -> LinOp:
    return LinOp(lambda v: diagonal * v, shape=(diagonal.shape[0], diagonal.shape[0]))


def test_check_mv_operator_validation_and_reporting_cover_scalar_and_vector_json(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(ValueError, match="positive size"):
        check_self_adjoint(_diag_op(jnp.ones((1,))), (0,))
    with pytest.raises(ValueError, match="positive size"):
        check_spd_quadratic_form(_diag_op(jnp.ones((1,))), (0,))

    print_Mv_report(
        {"ok": True, "values": jnp.asarray([1.0, 2.0]), "scalar": jnp.asarray(1.5)},
        None,
        name="self-only",
    )
    print_Mv_report(
        {"ok": True, "values": jnp.asarray([1.0, 2.0])},
        {"ok": True, "min_quad": jnp.asarray(0.5)},
        name="with-spd",
    )
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]

    assert lines[0]["report"]["values"] == [1.0, 2.0]
    assert lines[0]["report"]["scalar"] == 1.5
    assert lines[-1]["event"] == "spd"


def test_archived_solver_and_internal_jax_helpers_are_executable(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(AssertionError, match="archived"):
        gmres_solve()

    # NOTE: _test_jax モジュールは python/tests/solvers/test_jax_debug.py に移動済み
    # test_jax_module._print_jax_env()
    # test_jax_module._smoke_test()
    # 代わりに test_jax_debug.py を実行してください


def test_slq_normal_probes_and_minres_wrapper_cover_remaining_solver_helpers() -> None:
    probes = _make_probe_vectors(
        dim=4,
        s=3,
        probe="normal",
        seed=0,
        dtype=DEFAULT_DTYPE,
        eps=jnp.asarray(1e-12, dtype=DEFAULT_DTYPE),
    )
    norms = jnp.linalg.norm(probes, axis=0)
    assert jnp.allclose(norms, jnp.ones((3,), dtype=DEFAULT_DTYPE), atol=1e-5)

    rhs = jnp.array([1.0, -2.0], dtype=DEFAULT_DTYPE)
    x, _, info = minres_solve(
        Mv=_diag_op(jnp.ones((2,), dtype=DEFAULT_DTYPE)),
        rhs=rhs,
        minres_state=MINRESState.initialize(x0=jnp.zeros_like(rhs)),
        Minv=_diag_op(jnp.ones((2,), dtype=DEFAULT_DTYPE)),
        maxiter=5,
        dtype=DEFAULT_DTYPE,
    )
    assert jnp.allclose(x, rhs, atol=1e-6)
    assert bool(info["converged"])


def test_lobpcg_branch_helpers_cover_initialization_projection_and_largest_ordering() -> None:
    diagonal = jnp.array([1.0, 2.0, 4.0], dtype=DEFAULT_DTYPE)
    operator = _diag_op(diagonal)
    X0 = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=DEFAULT_DTYPE)

    state = init_block_eigen_state(operator, X0)
    assert jnp.allclose(state.AX, diagonal[:, None] * X0)
    assert jnp.allclose(state.P, jnp.zeros_like(X0))
    assert jnp.allclose(state.AP, jnp.zeros_like(X0))
    assert jnp.allclose(state.eigenvalues, jnp.zeros((2,), dtype=DEFAULT_DTYPE))

    with pytest.raises(ValueError, match="n=2 < r=3"):
        init_spectral_precond(operator, n=2, r=3)

    largest_state = init_spectral_precond(operator, n=3, r=2, which="largest")
    assert float(largest_state.eigenvalues[0]) >= float(largest_state.eigenvalues[1])

    projection_basis = SubspaceBasis(
        Q=jnp.array([[1.0], [0.0], [0.0]], dtype=DEFAULT_DTYPE),
        eigenvalues=jnp.array([1.0], dtype=DEFAULT_DTYPE),
    )
    projected = apply_projection(jnp.array([[3.0], [4.0], [5.0]], dtype=DEFAULT_DTYPE), projection_basis)
    assert jnp.allclose(projected, jnp.array([[0.0], [4.0], [5.0]], dtype=DEFAULT_DTYPE))

    _, updated_state, info = update_subspace(
        operator,
        base_precond=_diag_op(jnp.ones((3,), dtype=DEFAULT_DTYPE)),
        old_state=largest_state,
        maxiter=2,
        which="largest",
    )
    assert float(updated_state.eigenvalues[0]) >= float(updated_state.eigenvalues[1])
    assert "converged" in info


def test_kkt_solver_invalid_method_branches_raise_clean_errors() -> None:
    identity = _diag_op(jnp.ones((1,), dtype=DEFAULT_DTYPE))

    with pytest.raises(AssertionError, match="archived"):
        initialize_kkt_state(
            Hv_initial=identity,
            Bv_initial=identity,
            BTv_initial=identity,
            n_primal=1,
            n_dual=1,
            r_Hv_min=1,
            r_Sv_min=1,
            method="fgmres",
            dtype=DEFAULT_DTYPE,
        )

    with pytest.raises(ValueError, match="Unknown method"):
        initialize_kkt_state(
            Hv_initial=identity,
            Bv_initial=identity,
            BTv_initial=identity,
            n_primal=1,
            n_dual=1,
            r_Hv_min=1,
            r_Sv_min=1,
            method="mystery",
            dtype=DEFAULT_DTYPE,
        )

    valid_state = initialize_kkt_state(
        Hv_initial=identity,
        Bv_initial=identity,
        BTv_initial=identity,
        n_primal=1,
        n_dual=1,
        r_Hv_min=1,
        r_Sv_min=1,
        dtype=DEFAULT_DTYPE,
    )
    invalid_state = KKTState(
        S_inv_state=valid_state.S_inv_state,
        H_inv_state=valid_state.H_inv_state,
        solver_state=valid_state.solver_state,
        method="mystery",
    )

    with pytest.raises(ValueError, match="Unknown method"):
        _kkt_block_solver(
            identity,
            identity,
            identity,
            rhs_x=jnp.zeros((1,), dtype=DEFAULT_DTYPE),
            rhs_lam=jnp.zeros((1,), dtype=DEFAULT_DTYPE),
            kkt_state=invalid_state,
            dtype=DEFAULT_DTYPE,
            maxiter=2,
        )


def _run_all_tests() -> None:
    """全テストを実行します。
    
    補助的なpython file.py実行時に使用されます。
    pytest -s python/tests/solvers/test_solver_internal_branches.py
    と同等の実行が可能になります。
    """
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    _run_all_tests()
