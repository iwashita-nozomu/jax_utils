from __future__ import annotations

import json

import numpy as np
import jax.numpy as jnp
from scipy import optimize as sp_opt

from jax_util.Algorithms.pdipm import (
    PDIPMState,
    _pdipm_solve,
    initialize_pdipm_state,
)
from jax_util.base import DEFAULT_DTYPE


def _solve_with_scipy(
    objective: callable,
    n_primal: int,
) -> tuple[np.ndarray, sp_opt.OptimizeResult]:
    """SciPy で基準解を求めます。"""
    def fun(x: np.ndarray) -> float:
        return float(objective(x))

    bounds = sp_opt.Bounds(0.0, np.inf)
    constraints = (
        {
            "type": "eq",
            "fun": lambda x: float(np.sum(x) - 1.0),
        },
    )
    x0 = np.full((n_primal,), 1.0 / n_primal, dtype=np.float64)
    result = sp_opt.minimize(
        fun,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    return result.x, result


def test_pdipm_state_initialize() -> None:
    """PDIPM の状態が初期化できることを確認します。"""
    state = initialize_pdipm_state(
        n_primal=2,
        n_dual_eq=1,
        n_dual_ineq=1,
        r_Hv=1,
        r_Sv=1,
        r_max=2,
    )
    print(json.dumps({
        "case": "pdipm_state",
        "expected_x_dim": 2,
        "expected_lam_eq_dim": 1,
        "x_dim": int(state.x.shape[0]),
        "lam_eq_dim": int(state.lam_eq.shape[0]),
    }))
    assert isinstance(state, PDIPMState)
    assert state.x.shape == (2,)
    assert state.lam_eq.shape == (1,)
    assert state.lam_ineq.shape == (1,)
    assert state.s.shape == (1,)


def test_pdipm_simple_problem() -> None:
    """単純な制約付き二次問題が解けることを確認します。"""
    n_primal = 2
    m_eq = 1
    m_ineq = 1

    def f_opt(x: jnp.ndarray) -> jnp.ndarray:
        target = jnp.array([1.0, 2.0], dtype=DEFAULT_DTYPE)
        diff = x - target
        return 0.5 * jnp.sum(diff * diff)

    def c_eq(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([x[0] + x[1] - 1.0], dtype=DEFAULT_DTYPE)

    def c_ineq(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([-x[0]], dtype=DEFAULT_DTYPE)

    state = initialize_pdipm_state(
        n_primal=n_primal,
        n_dual_eq=m_eq,
        n_dual_ineq=m_ineq,
        r_Hv=1,
        r_Sv=1,
        r_max=2,
    )

    def scipy_obj(x: np.ndarray) -> float:
        target = np.array([1.0, 2.0], dtype=np.float64)
        diff = x - target
        return 0.5 * float(np.sum(diff * diff))

    expected_x_np, scipy_result = _solve_with_scipy(scipy_obj, n_primal)
    expected_x = jnp.asarray(expected_x_np, dtype=DEFAULT_DTYPE)
    opt, new_state, info = _pdipm_solve(
        f_opt=f_opt,
        c_eq=c_eq,
        c_ineq=c_ineq,
        optimizer_state=state,
        n_primal=n_primal,
        m_eq=m_eq,
        m_ineq=m_ineq,
        max_steps=40,
        ipm_tol=jnp.asarray(1e-8, dtype=DEFAULT_DTYPE),
    )

    print(json.dumps({
        "case": "pdipm_simple",
        "expected_x": expected_x_np.tolist(),
        "scipy_success": bool(scipy_result.success),
        "scipy_status": int(scipy_result.status),
        "x": new_state.x.tolist(),
        "opt": float(opt),
        "prim_res_final": float(info["prim_res_final"]),
        "mu_final": float(info["mu_final"]),
    }))

    assert jnp.allclose(new_state.x, expected_x, rtol=1e-3, atol=1e-3)


def test_pdipm_ill_conditioned_problem() -> None:
    """悪条件な二次問題でも実行できることを確認します。"""
    n_primal = 20
    m_eq = 1
    m_ineq = n_primal

    diag = jnp.logspace(0.0, 8.0, n_primal).astype(DEFAULT_DTYPE)

    def f_opt(x: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * jnp.sum(diag * x * x)

    def c_eq(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([jnp.sum(x) - 1.0], dtype=DEFAULT_DTYPE)

    def c_ineq(x: jnp.ndarray) -> jnp.ndarray:
        return -x

    state = initialize_pdipm_state(
        n_primal=n_primal,
        n_dual_eq=m_eq,
        n_dual_ineq=m_ineq,
        r_Hv=1,
        r_Sv=1,
        r_max=2,
    )

    def scipy_obj(x: np.ndarray) -> float:
        diag_np = np.logspace(0.0, 8.0, n_primal)
        return 0.5 * float(np.sum(diag_np * x * x))

    expected_x_np, scipy_result = _solve_with_scipy(scipy_obj, n_primal)
    expected_x = jnp.asarray(expected_x_np, dtype=DEFAULT_DTYPE)

    opt, new_state, info = _pdipm_solve(
        f_opt=f_opt,
        c_eq=c_eq,
        c_ineq=c_ineq,
        optimizer_state=state,
        n_primal=n_primal,
        m_eq=m_eq,
        m_ineq=m_ineq,
        max_steps=60,
        ipm_tol=jnp.asarray(1e-8, dtype=DEFAULT_DTYPE),
    )

    print(json.dumps({
        "case": "pdipm_ill",
        "expected_x": expected_x_np.tolist(),
        "scipy_success": bool(scipy_result.success),
        "scipy_status": int(scipy_result.status),
        "x": new_state.x.tolist(),
        "opt": float(opt),
        "prim_res_final": float(info["prim_res_final"]),
        "mu_final": float(info["mu_final"]),
    }))

    assert jnp.allclose(new_state.x, expected_x, rtol=1e-2, atol=1e-2)


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_pdipm_state_initialize()
    test_pdipm_simple_problem()
    test_pdipm_ill_conditioned_problem()


if __name__ == "__main__":
    _run_all_tests()
