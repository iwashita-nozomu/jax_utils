from __future__ import annotations

import os
if __name__=="__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

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
from _env_value import DEBUG, DEFAULT_DTYPE, EPS
from _fgmres import add_state, gmres_solve, initialize_fgmres_state
from _minres import MINRESState, pminres_solve
from _type_aliaces import LinearMap, Matrix, Scalar, Vector
from lobpcg import (
    BlockEigenState,
    SubspaceBasis,
    init_spectral_precond,
    make_rank_r_spectral_precond,
    update_subspace,
)


def _identity(v: Vector) -> Vector:
    return v

class KKTState(eqx.Module):
    S_inv_state:BlockEigenState
    H_inv_state:BlockEigenState
    solver_state:Any
    method:str = eqx.field(static=True)  # 'minres' or 'fgmres'

def initialize_kkt_state(
    Hv_initial:LinearMap,
    Bv_initial:LinearMap,
    BTv_initial:LinearMap,
    n_primal:int,
    n_dual:int,
    r_Hv_min:int,
    r_Sv_min:int,
    r_max:int,
    *,
    method:str='minres',
    restart:Optional[int]=None,

) -> KKTState:
    H_inv_state = init_spectral_precond(
        Mv=Hv_initial,
        n=n_primal,
        r=r_Hv_min,
        which='smallest',
    )

    H_eig,H_inv_state,H_info=update_subspace(
        Hv_initial,
        _identity,
        H_inv_state,
        maxiter=5,
    )
    
    approx_H_inv=make_rank_r_spectral_precond(basis=H_eig,)

    def _schur_mv(v: Vector) -> Vector:
        return Bv_initial(approx_H_inv(BTv_initial(v)))

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
    Hv: LinearMap,
    Bv: LinearMap,
    BTv: LinearMap,
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
        jax.debug.print("LOBPCG update H_inv: info: {info}\n", info=H_info)

    def Sv(v: Vector) -> Vector:
        #Sv = B H^{-1} B^T v
        Btv=BTv(v)
        Hinv_Btv=H_inv_approx(Btv)
        return Bv(Hinv_Btv)
    
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
        top=Hv(x)+BTv(lam)
        bot=Bv(x)
        return jnp.concatenate([top,bot],axis=0)


    def precond(v: Vector) -> Vector:
        #KKTのminres用前処理
        n_primal=rhs_x.shape[0]
        n_dual=rhs_lam.shape[0]
        x=v[:n_primal]
        lam=v[n_primal:]

        # top=H_inv_approx(x + BTv(S_inv_approx(lam)))
        top=H_inv_approx(x)
        # bot=S_inv_approx(Bv(H_inv_approx(x))+ lam)
        bot=S_inv_approx(lam)
        return jnp.concatenate([top,bot],axis=0)

    rhs=jnp.concatenate([rhs_x,rhs_lam],axis=0)

    if DEBUG:
        # while_loop / jit のトレース中は Python の bool() 変換ができず例外になります。
        # ここは数値計算ロジックではなく「自己随伴性/SPDの検査」なので、
        # JAXトレース文脈ではスキップし、通常実行時のみレポートを出します。
        if not isinstance(rhs_x, jax.core.Tracer) and not isinstance(rhs_lam, jax.core.Tracer):# type: ignore
            print_Mv_report(
                check_self_adjoint(KKT_mv, shape=(rhs_x.shape[0] + rhs_lam.shape[0],), num_trials=64),
                check_spd_quadratic_form(KKT_mv, shape=(rhs_x.shape[0] + rhs_lam.shape[0],), num_trials=64),
                name="KKT operator",
            )

            print_Mv_report(
                check_self_adjoint(precond, shape=(rhs_x.shape[0] + rhs_lam.shape[0],), num_trials=64),
                check_spd_quadratic_form(precond, shape=(rhs_x.shape[0] + rhs_lam.shape[0],), num_trials=64),
                name="KKT preconditioner",
            )
        else:
            jax.debug.print("[DEBUG] skip Mv checks inside JAX tracing context")
    scaled_tol=kkt_tol
    
    if kkt_state.method=='minres':

        v, new_solver_state , info = pminres_solve(
            Mv=KKT_mv,
            Minv=precond,
            rhs=rhs,
            minres_state=kkt_state.solver_state,
            rtol=scaled_tol,
            maxiter=maxiter,
        )
        

    elif kkt_state.method=='fgmres':

        v, new_solver_state , info = gmres_solve(
            Mv=KKT_mv,
            precond=add_state(precond),
            rhs=rhs,
            state=kkt_state.solver_state,
            rtol=scaled_tol,
            maxiter=maxiter,
        )

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

    if DEBUG:
        def _unscaledKKT(v: Vector) -> Vector:
            n_primal=rhs_x.shape[0]
            n_dual=rhs_lam.shape[0]
            x=v[:n_primal]
            lam=v[n_primal:]
            top=Hv(x)+BTv(lam)
            bot=Bv(x)
            return jnp.concatenate([top,bot],axis=0)
        unscaled_res_kkt=_unscaledKKT(jnp.concatenate([x,lam],axis=0))-jnp.concatenate([rhs_x,rhs_lam],axis=0)
        jax.debug.print("  residual norm KKT (MINRES unscaled): {}\n", jnp.linalg.norm(unscaled_res_kkt))

    new_kkt_state=KKTState(
        S_inv_state=new_S_inv_state,
        H_inv_state=new_H_inv_state,
        solver_state=new_solver_state,
        method=kkt_state.method,
    )

    return x,lam,new_kkt_state


def kkt_block_solver(
    Hv: LinearMap,
    Bv: LinearMap,
    BTv: LinearMap,
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
    import jax
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import gmres

    key = jax.random.PRNGKey(0)

    # ====== ユーティリティ ======
    def make_spd_matrix(n: int, cond: float, key: Vector) -> Matrix:
        """条件数 cond の SPD 行列を作る（かなり悪条件にできる）"""
        k1, k2 = jax.random.split(key)
        A = jax.random.normal(k1, (n, n))
        Q, _ = jnp.linalg.qr(A)  # 直交行列

        # 固有値を [1, cond] に対数スケールでばらまく
        log_min = 0.0
        log_max = jnp.log(cond)
        eigs = jnp.exp(jnp.linspace(log_min, log_max, n))

        H = (Q * eigs) @ Q.T
        # 数値誤差で完全対称でなくなるので対称化
        H_sym = 0.5 * (H + H.T)
        return H_sym

    def make_bad_B(m: int, n: int, key: Vector) -> Matrix:
        """行がほぼ従属 & スケール悪い B を作る"""
        k1, k2 = jax.random.split(key)
        # まずランダム
        B_base = jax.random.normal(k1, (m, n))

        # 上半分とほぼ同じ行を下半分に入れて、ほぼ従属にする
        half = m // 2
        noise = 1e-3 * jax.random.normal(k2, (m - half, n))
        B = B_base.at[half:].set(B_base[: m - half] + noise)

        # 列スケーリングも悪化させる（桁違いを入れる）
        scales = jnp.logspace(0.0, 4.0, n)  # 1〜1e4
        B_scaled = B * scales
        return B_scaled

    # ====== 問題サイズ & 悪条件設定 ======
    n_primal = 10000   # x の次元
    n_dual = 5000      # 制約本数
    cond_H = 1e4     # H の条件数（かなり悪い）

    # ====== H, B, 右辺の生成 ======
    key_H, key_B, key_rhs = jax.random.split(key, 3)

    H = make_spd_matrix(n_primal, cond_H, key_H)
    B = make_bad_B(n_dual, n_primal, key_B)

    def Hv(x: Vector) -> Vector:
        return H @ x

    def Bv(x: Vector) -> Vector:
        return B @ x

    def BTv(lam: Vector) -> Vector:
        return B.T @ lam
    rho = 1e-1  # とりあえずパラメータ。あとでチューニング

    def Hv_aug(x: Vector) -> Vector:
        # Hv_aug(x) = Hx + ρ B^T B x
        return Hv(x) + rho * BTv(Bv(x))
    # ====== 既知解から rhs を生成（必ず整合する KKT） ======
    # KKT:
    # [ H  B^T ] [x] = [r_x]
    # [ B   0  ] [λ]   [r_λ]
    # まず (x_true, lam_true) をランダムに作り、rhs をそれから作る。
    key_x, key_lam = jax.random.split(key_rhs, 2)
    x_true = jax.random.normal(key_x, (n_primal,), dtype=DEFAULT_DTYPE)
    lam_true = jax.random.normal(key_lam, (n_dual,), dtype=DEFAULT_DTYPE)

    rhs_primal = Hv(x_true) + BTv(lam_true)
    rhs_dual = Bv(x_true)

    # ====== KKT state 初期化 ======
    state = initialize_kkt_state(
        Hv_initial=Hv,
        Bv_initial=Bv,
        BTv_initial=BTv,
        n_primal=n_primal,
        n_dual=n_dual,
        r_Hv_min=256,
        r_Sv_min=256,
        r_max=256,
        method='minres',
        restart=200,
    )

    # ====== あなたの KKT ソルバを叩く ======
    print("=== running custom KKT solver on hard instance ===")
    x, lam, state = _kkt_block_solver(
        Hv=Hv,
        Bv=Bv,
        BTv=BTv,
        rhs_x=rhs_primal,
        rhs_lam=rhs_dual,
        kkt_state=state,
        kkt_tol=EPS,
        maxiter=15000,
    )

    print("x (custom)      shape:", x.shape)
    print("lambda (custom) shape:", lam.shape)

    # ====== 残差チェック（custom solver） ======
    Kx_top = Hv(x) + BTv(lam)
    Kx_bot = Bv(x)
    Kx = jnp.concatenate([Kx_top, Kx_bot], axis=0)

    rhs = jnp.concatenate([rhs_primal, rhs_dual], axis=0)
    residual_custom = jnp.linalg.norm(Kx - rhs) / jnp.linalg.norm(rhs)
    print("KKT rel residual (custom) =", float(residual_custom))

    # ====== 既知解との差分 ======
    err_x_true = jnp.linalg.norm(x - x_true) / (jnp.linalg.norm(x_true) + 1e-16)
    err_lam_true = jnp.linalg.norm(lam - lam_true) / (jnp.linalg.norm(lam_true) + 1e-16)
    print("rel error vs true x   =", float(err_x_true))
    print("rel error vs true lam =", float(err_lam_true))

    # ====== GMRES で「ほぼ正解」を作る ======
    print("=== running GMRES on full KKT (reference) ===")
    K = jnp.block([
        [H, B.T],
        [B, jnp.zeros((n_dual, n_dual), dtype=H.dtype)],
    ])

    sol_direct, info = gmres(K, rhs, tol=1e-10, atol=0.0, maxiter=1000)
    x_ref = sol_direct[:n_primal]
    lam_ref = sol_direct[n_primal:]

    print("gmres info =", info)  # 0 が成功のはず


    # ====== 残差チェック（gmres） ======
    Kx_ref = K @ sol_direct
    residual_ref = jnp.linalg.norm(Kx_ref - rhs) / jnp.linalg.norm(rhs)
    print("KKT rel residual (gmres)  =", float(residual_ref))
    print("x (gmres)      shape:", x_ref.shape)
    print("lambda (gmres) shape:", lam_ref.shape)
    err_x_true_gmres = jnp.linalg.norm(x_ref - x_true) / (jnp.linalg.norm(x_true) + 1e-16)
    err_lam_true_gmres = jnp.linalg.norm(lam_ref - lam_true) / (jnp.linalg.norm(lam_true) + 1e-16)
    print("rel error vs true x   =", float(err_x_true_gmres))
    print("rel error vs true lam =", float(err_lam_true_gmres))
    print("")

    # ====== 解の差分 ======
    err_x = jnp.linalg.norm(x - x_ref) / (jnp.linalg.norm(x_ref) + 1e-16)
    err_lam = jnp.linalg.norm(lam - lam_ref) / (jnp.linalg.norm(lam_ref) + 1e-16)
    print("rel error x (vs gmres)   =", float(err_x))
    print("rel error lam (vs gmres) =", float(err_lam))


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

    