from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
import jax.numpy as jnp
from scipy import optimize as sp_opt #pyright: ignore

from jax_util.Algorithms.pdipm import (
    PDIPMState,
    initialize_pdipm_state,
    pdipm_solve,
)
from jax_util.base import DEFAULT_DTYPE


SOURCE_FILE = Path(__file__).name


def _solve_with_scipy(
    objective: Callable[[NDArray[np.float64]], float],
    eq_constraint: Callable[[NDArray[np.float64]], float],
    ineq_constraint: Callable[[NDArray[np.float64]], float],
    n_primal: int,
) -> tuple[NDArray[np.float64], sp_opt.OptimizeResult]:
    """SciPy で基準解を求めます。"""
    def fun(x: NDArray[np.float64]) -> float:
        return float(objective(x))

    def eq_fun(x: NDArray[np.float64]) -> float:
        return float(eq_constraint(x))

    def ineq_fun(x: NDArray[np.float64]) -> float:
        # SciPy は g(x) >= 0 を想定するので符号を反転します。
        return float(-ineq_constraint(x))

    bounds = sp_opt.Bounds(0.0, np.inf)
    constraints = (
        {
            "type": "eq",
            "fun": eq_fun,
        },
        {
            "type": "ineq",
            "fun": ineq_fun,
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
        "source_file": SOURCE_FILE,
        "test": "test_pdipm_state_initialize",
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
        term1 = (x[0] - 0.2) ** 4
        term2 = (x[1] - 0.8) ** 2
        term3 = 0.1 * jnp.sin(5.0 * x[0])
        return term1 + term2 + term3

    def c_eq(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([x[0] ** 2 + x[1] ** 2 - 1.0], dtype=DEFAULT_DTYPE)

    def c_ineq(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([0.05 - x[0] * x[1]], dtype=DEFAULT_DTYPE)

    state = initialize_pdipm_state(
        n_primal=n_primal,
        n_dual_eq=m_eq,
        n_dual_ineq=m_ineq,
        r_Hv=1,
        r_Sv=1,
        r_max=2,
    )
    # 初期点で等式制約のヤコビアンがゼロになると KKT が不定になります。
    # 非線形問題では初期点を制約近傍に置いて数値的不安定を避けます。
    x_init = jnp.asarray([0.6, 0.8], dtype=DEFAULT_DTYPE)
    state = PDIPMState(
        kkt_state=state.kkt_state,
        x=x_init,
        lam_eq=state.lam_eq,
        lam_ineq=state.lam_ineq,
        s=state.s,
    )

    def scipy_obj(x: NDArray[np.float64]) -> float:
        term1 = float((x[0] - 0.2) ** 4)
        term2 = float((x[1] - 0.8) ** 2)
        term3 = float(0.1 * np.sin(5.0 * x[0]))
        return term1 + term2 + term3

    def scipy_eq(x: NDArray[np.float64]) -> float:
        return float(x[0] ** 2 + x[1] ** 2 - 1.0)

    def scipy_ineq(x: NDArray[np.float64]) -> float:
        return float(0.05 - x[0] * x[1])

    expected_x_np, scipy_result = _solve_with_scipy(
        scipy_obj,
        scipy_eq,
        scipy_ineq,
        n_primal,
    )
    opt, new_state, info = pdipm_solve(
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

    scipy_fun = float(scipy_result.fun)
    print(json.dumps({
        "case": "pdipm_simple",
            "source_file": SOURCE_FILE,
        "test": "test_pdipm_simple_problem",
        "expected_x": expected_x_np.tolist(),
        "scipy_success": bool(scipy_result.success),
        "scipy_status": int(scipy_result.status),
        "scipy_fun": scipy_fun,
        "x": new_state.x.tolist(),
        "opt": float(opt),
        "prim_res_final": float(info["prim_res_final"]),
        "mu_final": float(info["mu_final"]),
    }))

    assert scipy_result.success
    assert float(info["prim_res_final"]) <= 1e-6
    assert float(opt) <= scipy_fun * 1.02


def test_pdipm_ill_conditioned_problem() -> None:
    """悪条件な二次問題でも実行できることを確認します。"""
    n_primal = 20
    m_eq = 1
    m_ineq = n_primal

    diag = jnp.logspace(0.0, 8.0, n_primal).astype(DEFAULT_DTYPE)

    def f_opt(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.sum(diag * x ** 4)

    def c_eq(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.array([jnp.sum(x ** 2) - 1.0], dtype=DEFAULT_DTYPE)

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
    # 等式制約の勾配がゼロにならない初期点を用意します。
    x_init = jnp.ones((n_primal,), dtype=DEFAULT_DTYPE)
    x_init = x_init / jnp.linalg.norm(x_init)
    state = PDIPMState(
        kkt_state=state.kkt_state,
        x=x_init,
        lam_eq=state.lam_eq,
        lam_ineq=state.lam_ineq,
        s=state.s,
    )

    def scipy_obj(x: NDArray[np.float64]) -> float:
        diag_np = np.logspace(0.0, 8.0, n_primal)
        return float(np.sum(diag_np * x ** 4))

    def scipy_eq(x: NDArray[np.float64]) -> float:
        return float(np.sum(x ** 2) - 1.0)

    def scipy_ineq(x: NDArray[np.float64]) -> float:
        return float(-np.min(x))

    expected_x_np, scipy_result = _solve_with_scipy(
        scipy_obj,
        scipy_eq,
        scipy_ineq,
        n_primal,
    )

    opt, new_state, info = pdipm_solve(
        f_opt=f_opt,
        c_eq=c_eq,
        c_ineq=c_ineq,
        optimizer_state=state,
        n_primal=n_primal,
        m_eq=m_eq,
        m_ineq=m_ineq,
        max_steps=100,
        ipm_tol=jnp.asarray(1e-8, dtype=DEFAULT_DTYPE),
    )

    scipy_fun = float(scipy_result.fun)
    print(json.dumps({
        "case": "pdipm_ill",
            "source_file": SOURCE_FILE,
        "test": "test_pdipm_ill_conditioned_problem",
        "expected_x": expected_x_np.tolist(),
        "scipy_success": bool(scipy_result.success),
        "scipy_status": int(scipy_result.status),
        "scipy_fun": scipy_fun,
        "x": new_state.x.tolist(),
        "opt": float(opt),
        "prim_res_final": float(info["prim_res_final"]),
        "mu_final": float(info["mu_final"]),
    }))

    assert scipy_result.success
    assert float(info["prim_res_final"]) <= 1e-6
    assert float(opt) <= scipy_fun * 1.05


def _run_all_tests() -> None:
    """このファイル内のテストを順に実行します。"""
    test_pdipm_state_initialize()
    test_pdipm_simple_problem()
    test_pdipm_ill_conditioned_problem()


if __name__ == "__main__":
    _run_all_tests()
