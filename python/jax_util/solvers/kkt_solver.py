from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp

# from jax.scipy.sparse.linalg import gmres

from jax.typing import DTypeLike

from ._check_mv_operator import (
    check_self_adjoint,
    check_spd_quadratic_form,
    print_Mv_report,
)
from ..base import (
    DEFAULT_DTYPE,
    EPS,
    DEBUG,
    LinOp,
    LinearOperator,
    ONE,
    Scalar,
    Vector,
)

# from ._fgmres import *
from ._minres import MINRESState, pminres_solve
from .lobpcg import (
    BlockEigenState,
    SubspaceBasis,
    init_spectral_precond,
    make_rank_r_spectral_precond,
    update_subspace,
)

SOURCE_FILE = Path(__file__).name


class KKTState(eqx.Module):
    S_inv_state: BlockEigenState
    H_inv_state: BlockEigenState
    solver_state: Any
    method: str = eqx.field(static=True)  # 'minres' or 'fgmres'


# 責務: KKT ソルバで再利用するスペクトル前処理状態と内部反復状態を初期化する。
def initialize_kkt_state(
    Hv_initial: LinearOperator,
    Bv_initial: LinearOperator,
    BTv_initial: LinearOperator,
    n_primal: int,
    n_dual: int,
    r_Hv_min: int,
    r_Sv_min: int,
    *,
    method: str = "minres",
    restart: Optional[int] = None,
    dtype: DTypeLike = DEFAULT_DTYPE,
) -> KKTState:
    if method == "fgmres":
        raise AssertionError("FGMRES is archived and not supported.")

    H_inv_state = init_spectral_precond(
        Mv=Hv_initial,
        n=n_primal,
        r=r_Hv_min,
        which="smallest",
    )

    H_eig, H_inv_state, H_info = update_subspace(
        Hv_initial,
        LinOp(lambda v: v),  # 単位行列を前処理に使う
        H_inv_state,
        maxiter=1000,
    )

    approx_H_inv = make_rank_r_spectral_precond(
        basis=H_eig,
    )

    # def _schur_mv(v: Vector) -> Vector:
    #     return (Bv_initial * approx_H_inv *BTv_initial) @ v

    _schur_mv = Bv_initial * approx_H_inv * BTv_initial

    S_inv_state = init_spectral_precond(
        Mv=_schur_mv,
        n=n_dual,
        r=r_Sv_min,
        which="smallest",
    )

    if method == "minres":
        solver_state = MINRESState(
            x0=jnp.zeros((n_primal + n_dual,), dtype=dtype),
        )
    # elif method=='fgmres':
    #     if restart is None:
    #         raise ValueError("restart must be specified for fgmres method")

    #     solver_state=initialize_fgmres_state(
    #         n=n_primal+n_dual,
    #         restart=restart,
    #         precond_state=None,
    #     )
    else:
        raise ValueError(f"Unknown method: {method}")

    return KKTState(
        S_inv_state=S_inv_state,
        H_inv_state=H_inv_state,
        solver_state=solver_state,
        method=method,
    )


# 責務: KKT 系の前処理更新と線形方程式求解をまとめて実行する。
def _kkt_block_solver(
    Hv: LinearOperator,
    Bv: LinearOperator,
    BTv: LinearOperator,
    rhs_x: Vector,
    rhs_lam: Vector,
    kkt_state: KKTState,
    *,
    kkt_tol: Scalar = EPS,
    maxiter: int = 1000,
    dtype: DTypeLike,
):
    base_precond_H = make_rank_r_spectral_precond(
        basis=SubspaceBasis.from_state(kkt_state.H_inv_state)
    )

    H_eig, new_H_inv_state, H_info = update_subspace(
        Hv, base_precond=base_precond_H, old_state=kkt_state.H_inv_state, maxiter=1000, tol=EPS
    )

    H_inv_approx = make_rank_r_spectral_precond(
        basis=H_eig,
    )

    if DEBUG:
        jax.debug.print(
            '{{"case":"kkt","source_file":"{source_file}",'
            '"func":"_kkt_block_solver","event":"H_inv_update","info":{info}}}',
            source_file=SOURCE_FILE,
            info=H_info,
        )

    # def Sv(v: Vector) -> Vector:
    #     #Sv = B H^{-1} B^T v
    #     Btv=BTv(v)
    #     Hinv_Btv=H_inv_approx(Btv)
    #     return Bv(Hinv_Btv)

    Sv = Bv * H_inv_approx * BTv

    base_precond_S = make_rank_r_spectral_precond(
        basis=SubspaceBasis.from_state(kkt_state.S_inv_state)
    )

    S_eig, new_S_inv_state, S_info = update_subspace(
        Sv, base_precond=base_precond_S, old_state=kkt_state.S_inv_state, maxiter=100, tol=EPS
    )
    S_inv_approx = make_rank_r_spectral_precond(
        basis=S_eig,
    )

    if DEBUG:
        jax.debug.print(
            '{{"case":"kkt","source_file":"{source_file}",'
            '"func":"_kkt_block_solver","event":"S_inv_update","info":{info}}}',
            source_file=SOURCE_FILE,
            info=S_info,
        )

    # 責務: primal-dual 連立を 1 本の KKT 作用素としてまとめる。
    def KKT_mv(v: Vector) -> Vector:
        n_primal = rhs_x.shape[0]
        n_dual = rhs_lam.shape[0]
        x = v[:n_primal]
        lam = v[n_primal:]
        top = Hv @ x + BTv @ lam
        bot = Bv @ x
        return jnp.concatenate([top, bot], axis=0)

    # 責務: ブロック対角近似で MINRES 用の前処理を与える。
    def precond(v: Vector) -> Vector:
        # KKTのminres用前処理
        n_primal = rhs_x.shape[0]
        n_dual = rhs_lam.shape[0]
        x = v[:n_primal]
        lam = v[n_primal:]

        # top=H_inv_approx(x + BTv(S_inv_approx(lam)))
        top = H_inv_approx @ x
        # bot=S_inv_approx(Bv(H_inv_approx(x))+ lam)
        bot = S_inv_approx @ lam
        return jnp.concatenate([top, bot], axis=0)

    rhs = jnp.concatenate([rhs_x, rhs_lam], axis=0)

    # Notes:
    # - これらの検査は Python 制御フロー（`bool(...)`）を含むため、JIT/while_loop のトレース中に
    #   実行すると `TracerBoolConversionError` の原因になります。
    # - そのため、トレース文脈ではスキップし、通常実行時のみ実行します。
    # pyright は `jax.core.Tracer` を解決できない場合があるため、ここは型チェック対象外とします。
    # 実行時の挙動（Tracerなら検査をスキップする）を優先します。
    if DEBUG and not (
        isinstance(
            rhs_x, getattr(jax, "core").Tracer
        )  # pyright: ignore[reportAttributeAccessIssue]
        or isinstance(
            rhs_lam, getattr(jax, "core").Tracer
        )  # pyright: ignore[reportAttributeAccessIssue]
    ):
        # while_loop / jit のトレース中は Python の bool() 変換ができず例外になります。
        # ここは数値計算ロジックではなく「自己随伴性/SPDの検査」なので、
        # JAXトレース文脈ではスキップし、通常実行時のみレポートを出します。
        print_Mv_report(
            check_self_adjoint(
                LinOp(KKT_mv),
                shape=(rhs_x.shape[0] + rhs_lam.shape[0],),
                num_trials=64,
                dtype=dtype,
            ),
            None,
            name="KKT operator",
        )

        print_Mv_report(
            check_self_adjoint(
                LinOp(precond),
                shape=(rhs_x.shape[0] + rhs_lam.shape[0],),
                num_trials=64,
                dtype=dtype,
            ),
            check_spd_quadratic_form(
                LinOp(precond),
                shape=(rhs_x.shape[0] + rhs_lam.shape[0],),
                num_trials=64,
                dtype=dtype,
            ),
            name="KKT preconditioner",
        )
    scaled_tol = kkt_tol

    if kkt_state.method == "minres":

        v, new_solver_state, solver_info = pminres_solve(
            Mv=LinOp(KKT_mv),
            Minv=LinOp(precond),
            rhs=rhs,
            minres_state=kkt_state.solver_state,
            rtol=scaled_tol,
            maxiter=maxiter,
            dtype=dtype,
        )

    # elif kkt_state.method=='fgmres':
    #     v, new_solver_state , info = gmres_solve(
    #         Mv=LinOp(KKT_mv),
    #         precond=add_state(LinOp(precond)),
    #         rhs=rhs,
    #         state=kkt_state.solver_state,
    #         rtol=scaled_tol,
    #         maxiter=maxiter,
    #     )

    else:
        raise ValueError(f"Unknown method: {kkt_state.method}")

    res = KKT_mv(v) - rhs
    res_norm = jnp.linalg.norm(res)
    rhs_norm = jnp.linalg.norm(rhs)
    rel_res = res_norm / jnp.where(rhs_norm == 0, ONE, rhs_norm)

    info = {
        "solver_info": solver_info,
        "res_norm": res_norm,
        "rhs_norm": rhs_norm,
        "rel_res": rel_res,
    }

    if DEBUG:
        jax.debug.print(
            '{{"case":"kkt","source_file":"{source_file}",'
            '"func":"_kkt_block_solver","event":"solver_info",'
            '"res_norm":{res_norm},"rhs_norm":{rhs_norm},"rel_res":{rel_res},'
            '"converged":{converged},"num_iter":{num_iter}}}',
            source_file=SOURCE_FILE,
            res_norm=info["res_norm"],
            rhs_norm=info["rhs_norm"],
            rel_res=info["rel_res"],
            converged=info["solver_info"]["converged"],
            num_iter=info["solver_info"]["num_iter"],
        )

    if DEBUG:
        res_kkt = KKT_mv(v) - jnp.concatenate([rhs_x, rhs_lam], axis=0)
        jax.debug.print(
            '{{"case":"kkt","source_file":"{source_file}",'
            '"func":"_kkt_block_solver","event":"residual_check",'
            '"residual_norm":{residual_norm}}}',
            source_file=SOURCE_FILE,
            residual_norm=jnp.linalg.norm(res_kkt),
        )

    dx = v[: rhs_x.shape[0]]
    dlam = v[rhs_x.shape[0] :]

    x = dx
    lam = dlam

    new_kkt_state = KKTState(
        S_inv_state=new_S_inv_state,
        H_inv_state=new_H_inv_state,
        solver_state=new_solver_state,
        method=kkt_state.method,
    )

    ans = (x, lam)
    if DEBUG:
        jax.debug.print(
            '{{"case":"kkt","source_file":"{source_file}",'
            '"func":"_kkt_block_solver","event":"return",'
            '"res_norm":{res_norm},"rel_res":{rel_res},"converged":{converged},'
            '"num_iter":{num_iter}}}',
            source_file=SOURCE_FILE,
            res_norm=info["res_norm"],
            rel_res=info["rel_res"],
            converged=info["solver_info"]["converged"],
            num_iter=info["solver_info"]["num_iter"],
        )
        jax.debug.print("")
    return ans, new_kkt_state, info


# 責務: 公開 API として内部 KKT ブロックソルバを既定値付きで公開する。
def kkt_block_solver(
    Hv: LinearOperator,
    Bv: LinearOperator,
    BTv: LinearOperator,
    rhs_x: Vector,
    rhs_lam: Vector,
    kkt_state: KKTState,
    *,
    kkt_tol: Scalar = EPS,
    maxiter: int = 1000,
    dtype: DTypeLike = DEFAULT_DTYPE,
):
    """公開用の KKT ブロックソルバ。"""
    return _kkt_block_solver(
        Hv=Hv,
        Bv=Bv,
        BTv=BTv,
        rhs_x=rhs_x,
        rhs_lam=rhs_lam,
        kkt_state=kkt_state,
        kkt_tol=kkt_tol,
        maxiter=maxiter,
        dtype=dtype,
    )


__all__ = [
    "KKTState",
    "initialize_kkt_state",
    "_kkt_block_solver",
    "kkt_block_solver",
]
