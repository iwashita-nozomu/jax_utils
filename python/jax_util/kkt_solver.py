from __future__ import annotations

import os
if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    os.environ["JAX_ASYNC_DISPATCH"] = "0"
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

from _check_mv_operator import (
    check_self_adjoint,
    check_spd_quadratic_form,
    print_Mv_report,
)
from base import *
from _fgmres import *
from _minres import MINRESState, pminres_solve
from lobpcg import (
    BlockEigenState,
    SubspaceBasis,
    init_spectral_precond,
    make_rank_r_spectral_precond,
    update_subspace,
)



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
    r_max:int,
    *,
    method:str='minres',
    restart:Optional[int]=None,

) -> KKTState:
    if method=='fgmres':
        assert "Do not use FGMRES for now"

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
        maxiter=5,
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
    elif method=='fgmres':
        if restart is None:
            raise ValueError("restart must be specified for fgmres method")

        solver_state=initialize_fgmres_state(
            n=n_primal+n_dual,
            restart=restart,
            precond_state=None,
        )
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
                                          maxiter=100,
                                          tol=EPS)
    
    H_inv_approx=make_rank_r_spectral_precond(basis=H_eig,)

    if DEBUG:
        jax.debug.print("LOBPCG update H_inv: info: {info}\n", info=H_info)

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
        jax.debug.print("LOBPCG update S_inv: info: {info}\n", info=S_info)

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
        if not isinstance(rhs_x, jax.core.Tracer) and not isinstance(rhs_lam, jax.core.Tracer):# type: ignore
            print_Mv_report(
                check_self_adjoint(LinOp(KKT_mv), shape=(rhs_x.shape[0] + rhs_lam.shape[0],), num_trials=64),
                check_spd_quadratic_form(LinOp(KKT_mv), shape=(rhs_x.shape[0] + rhs_lam.shape[0],), num_trials=64),
                name="KKT operator",
            )

            print_Mv_report(
                check_self_adjoint(LinOp(precond), shape=(rhs_x.shape[0] + rhs_lam.shape[0],), num_trials=64),
                check_spd_quadratic_form(LinOp(precond), shape=(rhs_x.shape[0] + rhs_lam.shape[0],), num_trials=64),
                name="KKT preconditioner",
            )
        else:
            jax.debug.print("[DEBUG] skip Mv checks inside JAX tracing context")
    scaled_tol=kkt_tol
    
    if kkt_state.method=='minres':

        v, new_solver_state , info = pminres_solve(
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

    if DEBUG:
        jax.debug.print("KKT block solver: solver info: {info}\n", info=info)

    if DEBUG:
        jax.debug.print("\nKKT block solver: MINRES solution residual check\n")
        res_kkt=KKT_mv(v)-jnp.concatenate([rhs_x,rhs_lam],axis=0)
        jax.debug.print("  residual norm KKT (MINRES): {}\n", jnp.linalg.norm(res_kkt))

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

    return x,lam,new_kkt_state


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


if __name__ == "__main__":
    import jax.numpy as jnp

    def test_kkt_known_solution() -> None:
        H: Matrix = jnp.asarray([[2.0, 0.0], [0.0, 3.0]])
        B: Matrix = jnp.asarray([[1.0, 1.0]])

        def Hv(x: Vector) -> Vector:
            return H @ x

        def Bv(x: Vector) -> Vector:
            return B @ x

        def BTv(lam: Vector) -> Vector:
            return B.T @ lam

        x_true: Vector = jnp.asarray([1.0, -1.0])
        lam_true: Vector = jnp.asarray([0.5])
        rhs_x: Vector = Hv(x_true) + BTv(lam_true)
        rhs_lam: Vector = Bv(x_true)

        state = initialize_kkt_state(
            Hv_initial=H,
            Bv_initial=B,
            BTv_initial=B.T,
            n_primal=2,
            n_dual=1,
            r_Hv_min=1,
            r_Sv_min=1,
            r_max=1,
            method="minres",
            restart=5,
        )

        x, lam, _ = _kkt_block_solver(
            Hv=H,
            Bv=B,
            BTv=B.T,
            rhs_x=rhs_x,
            rhs_lam=rhs_lam,
            kkt_state=state,
            kkt_tol=EPS,
            maxiter=50,
        )

        K = jnp.block(
            [
                [H, B.T],
                [B, jnp.zeros((1, 1), dtype=H.dtype)],
            ]
        )
        sol_ref = jnp.linalg.solve(K, jnp.concatenate([rhs_x, rhs_lam]))
        x_ref = sol_ref[:2]
        lam_ref = sol_ref[2:]

        assert jnp.allclose(x, x_ref, rtol=1e-5, atol=1e-5)
        assert jnp.allclose(lam, lam_ref, rtol=1e-5, atol=1e-5)

    test_kkt_known_solution()


    # #残差二乗和最小化として解いてみる
    # print("=== running custom KKT residual norm sq check ===")
    # def kkt_residual_squared_norm(v: jax.Array) -> jax.Array:
    #     x_part = v[:n_primal]
    #     lam_part = v[n_primal:]

    #     res_top = Hv(x_part) + BTv(lam_part) - rhs_primal
    #     res_bot = Bv(x_part) - rhs_dual

    #     res = jnp.concatenate([res_top, res_bot], axis=0)
    #     return jnp.sum(res ** 2)
    
    # from jaxopt import LBFGS
    # init_v = jnp.zeros((n_primal + n_dual,), dtype=DEFAULT_DTYPE)
    # lbfgs = LBFGS(fun=kkt_residual_squared_norm, maxiter=5000, tol=1e-10)
    # sol_lbfgs, state_lbfgs = lbfgs.run(init_v)
    # x_lbfgs = sol_lbfgs[:n_primal]
    # lam_lbfgs = sol_lbfgs[n_primal:]
    # print("x (lbfgs)      shape:", x_lbfgs.shape)
    # print("lambda (lbfgs) shape:", lam_lbfgs.shape)
    # # ====== 残差チェック（lbfgs） ======
    # Kx_top = Hv(x_lbfgs) + BTv(lam_lbfgs)   # Hx + B^T λ
    # Kx_bot = Bv(x_lbfgs)              # Bx
    # Kx = jnp.concatenate([Kx_top, Kx_bot], axis=0)
    # residual_lbfgs = jnp.linalg.norm(Kx - rhs)
    # print("KKT residual (lbfgs) =", float(residual_lbfgs))

    