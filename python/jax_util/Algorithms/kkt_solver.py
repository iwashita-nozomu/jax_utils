from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
# from jax.scipy.sparse.linalg import gmres

from ._check_mv_operator import (
    check_self_adjoint,
    check_spd_quadratic_form,
    print_Mv_report,
)
from ..base import *
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
    S_inv_state:BlockEigenState
    H_inv_state:BlockEigenState
    solver_state:Any
    method:str = eqx.field(static=True)  # 'minres' or 'fgmres'

def initialize_kkt_state(
    Hv_initial:LinearOperator,
    Bv_initial:LinearOperator,
    BTv_initial:LinearOperator,
    n_primal:int,
    n_dual:int,
    r_Hv_min:int,
    r_Sv_min:int,
    *,
    method:str='minres',
    restart:Optional[int]=None,

) -> KKTState:
    if method=='fgmres':
        raise AssertionError("FGMRES is archived and not supported.")

    H_inv_state = init_spectral_precond(
        Mv=Hv_initial,
        n=n_primal,
        r=r_Hv_min,
        which='smallest',
    )

    H_eig,H_inv_state,H_info=update_subspace(
        Hv_initial,
        LinOp(lambda v: v), # 単位行列を前処理に使う
        H_inv_state,
        maxiter=1000,
    )
    
    approx_H_inv=make_rank_r_spectral_precond(basis=H_eig,)

    # def _schur_mv(v: Vector) -> Vector:
    #     return (Bv_initial * approx_H_inv *BTv_initial) @ v

    _schur_mv = Bv_initial * approx_H_inv * BTv_initial

    S_inv_state=init_spectral_precond(
        Mv=_schur_mv,
        n=n_dual,
        r=r_Sv_min,
        which='smallest',
    )


    if method=='minres':
        solver_state=MINRESState(
            x0=jnp.zeros((n_primal+n_dual,),dtype=DEFAULT_DTYPE),
        )
    # elif method=='fgmres':
    #     if restart is None:
    #         raise ValueError("restart must be specified for fgmres method")

    #     solver_state=initialize_fgmres_state(
    #         n=n_primal+n_dual,
    #         restart=restart,
    #         precond_state=None,
    #     )
    else :
        raise ValueError(f"Unknown method: {method}")

    return KKTState(
        S_inv_state=S_inv_state,
        H_inv_state=H_inv_state,
        solver_state=solver_state,
        method=method,
    )


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
):  
    base_precond_H=make_rank_r_spectral_precond(basis=SubspaceBasis.from_state(kkt_state.H_inv_state))

    H_eig,new_H_inv_state,H_info=update_subspace(Hv,
                                          base_precond=base_precond_H,
                                          old_state=kkt_state.H_inv_state,
                                          maxiter=1000,
                                          tol=EPS)
    
    H_inv_approx=make_rank_r_spectral_precond(basis=H_eig,)

    if DEBUG:
        jax.debug.print(
            "{{\"case\":\"kkt\",\"source_file\":\"{source_file}\","
            "\"func\":\"_kkt_block_solver\",\"event\":\"H_inv_update\",\"info\":{info}}}",
            source_file=SOURCE_FILE,
            info=H_info,
        )

    # def Sv(v: Vector) -> Vector:
    #     #Sv = B H^{-1} B^T v
    #     Btv=BTv(v)
    #     Hinv_Btv=H_inv_approx(Btv)
    #     return Bv(Hinv_Btv)
    
    Sv = Bv * H_inv_approx * BTv

    base_precond_S=make_rank_r_spectral_precond(basis=SubspaceBasis.from_state(kkt_state.S_inv_state))
    
    S_eig,new_S_inv_state,S_info=update_subspace(Sv,
                                          base_precond=base_precond_S,
                                          old_state=kkt_state.S_inv_state,
                                          maxiter=100,
                                          tol=EPS)
    S_inv_approx=make_rank_r_spectral_precond(basis=S_eig,)
    
    if DEBUG:
        jax.debug.print(
            "{{\"case\":\"kkt\",\"source_file\":\"{source_file}\","
            "\"func\":\"_kkt_block_solver\",\"event\":\"S_inv_update\",\"info\":{info}}}",
            source_file=SOURCE_FILE,
            info=S_info,
        )

    def KKT_mv(v: Vector) -> Vector:
        n_primal=rhs_x.shape[0]
        n_dual=rhs_lam.shape[0]
        x=v[:n_primal]
        lam=v[n_primal:]
        top=Hv @ x+BTv @ lam
        bot=Bv @ x
        return jnp.concatenate([top,bot],axis=0)


    def precond(v: Vector) -> Vector:
        #KKTのminres用前処理
        n_primal=rhs_x.shape[0]
        n_dual=rhs_lam.shape[0]
        x=v[:n_primal]
        lam=v[n_primal:]

        # top=H_inv_approx(x + BTv(S_inv_approx(lam)))
        top=H_inv_approx @ x
        # bot=S_inv_approx(Bv(H_inv_approx(x))+ lam)
        bot=S_inv_approx @ lam
        return jnp.concatenate([top,bot],axis=0)

    rhs=jnp.concatenate([rhs_x,rhs_lam],axis=0)

    if DEBUG:
        # while_loop / jit のトレース中は Python の bool() 変換ができず例外になります。
        # ここは数値計算ロジックではなく「自己随伴性/SPDの検査」なので、
        # JAXトレース文脈ではスキップし、通常実行時のみレポートを出します。
        if not isinstance(rhs_x, jax.core.Tracer) and not isinstance(rhs_lam, jax.core.Tracer):
            print_Mv_report(
                check_self_adjoint(LinOp(KKT_mv), shape=(rhs_x.shape[0] + rhs_lam.shape[0],), num_trials=64),
                None,
                name="KKT operator",
            )

            print_Mv_report(
                check_self_adjoint(LinOp(precond), shape=(rhs_x.shape[0] + rhs_lam.shape[0],), num_trials=64),
                check_spd_quadratic_form(LinOp(precond), shape=(rhs_x.shape[0] + rhs_lam.shape[0],), num_trials=64),
                name="KKT preconditioner",
            )
        else:
            jax.debug.print(
                "{{\"case\":\"kkt\",\"source_file\":\"{source_file}\","
                "\"func\":\"_kkt_block_solver\",\"event\":\"skip_mv_checks\"}}",
                source_file=SOURCE_FILE,
            )
    scaled_tol=kkt_tol
    
    if kkt_state.method=='minres':

        v, new_solver_state, solver_info = pminres_solve(
            Mv=LinOp(KKT_mv),
            Minv=LinOp(precond),
            rhs=rhs,
            minres_state=kkt_state.solver_state,
            rtol=scaled_tol,
            maxiter=maxiter,
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

    else :
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
            "{{\"case\":\"kkt\",\"source_file\":\"{source_file}\","
            "\"func\":\"_kkt_block_solver\",\"event\":\"solver_info\","
            "\"res_norm\":{res_norm},\"rhs_norm\":{rhs_norm},\"rel_res\":{rel_res},"
            "\"converged\":{converged},\"num_iter\":{num_iter}}}",
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
            "{{\"case\":\"kkt\",\"source_file\":\"{source_file}\","
            "\"func\":\"_kkt_block_solver\",\"event\":\"residual_check\","
            "\"residual_norm\":{residual_norm}}}",
            source_file=SOURCE_FILE,
            residual_norm=jnp.linalg.norm(res_kkt),
        )

    dx=v[:rhs_x.shape[0]]
    dlam=v[rhs_x.shape[0]:]
    
    x = dx
    lam = dlam

    new_kkt_state=KKTState(
        S_inv_state=new_S_inv_state,
        H_inv_state=new_H_inv_state,
        solver_state=new_solver_state,
        method=kkt_state.method,
    )

    ans = (x, lam)
    if DEBUG:
        jax.debug.print(
            "{{\"case\":\"kkt\",\"source_file\":\"{source_file}\","
            "\"func\":\"_kkt_block_solver\",\"event\":\"return\","
            "\"res_norm\":{res_norm},\"rel_res\":{rel_res},\"converged\":{converged},"
            "\"num_iter\":{num_iter}}}",
            source_file=SOURCE_FILE,
            res_norm=info["res_norm"],
            rel_res=info["rel_res"],
            converged=info["solver_info"]["converged"],
            num_iter=info["solver_info"]["num_iter"],
        )
        jax.debug.print("")
    return ans, new_kkt_state, info


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
    )

__all__ = [
    "KKTState",
    "initialize_kkt_state",
    "_kkt_block_solver",
    "kkt_block_solver",
]

    