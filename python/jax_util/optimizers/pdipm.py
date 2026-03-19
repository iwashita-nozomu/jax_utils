"""Mehrotra 型内点法（PDIPM）の実装。

References
----------
- Mehrotra, S. (1992).
  "On the implementation of a primal-dual interior point method."
  SIAM journal on optimization, 2(4), 575-601.
  https://epubs.siam.org/doi/abs/10.1137/0802028

- Wright, S. J. (1997).
  "Primal-Dual Interior-Point Methods."
  SIAM, Philadelphia, PA.

  このモジュールは Mehrotra の predictor-corrector スキームを採用し、
  凸最適化問題 (P) minimize c^T x s.t. Ax=b, x≥0 および双対問題 (D) を
  解きます。KKT ブロックソルバと組み合わせることで、大規模稀疎問題でも
  高速収束を実現する inexact Newton 法による求解が可能です。
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Callable
from pathlib import Path

import equinox as eqx
import jax
from jax import numpy as jnp

from ..base import (
    AVOID_ZERO_DIV,
    DEBUG,
    DEFAULT_DTYPE,
    EPS,
    LinOp,
    LinearOperator,
    ONE,
    Scalar,
    Vector,
    ZERO,
    adjoint,
    linearize,
)


from ..solvers.kkt_solver import KKTState, initialize_kkt_state, kkt_block_solver

from jax.typing import DTypeLike

SOURCE_FILE = Path(__file__).name


class PDIPMState(eqx.Module):
    kkt_state: KKTState
    x: Vector
    lam_eq: Vector
    lam_ineq: Vector
    s: Vector


# 責務: PDIPM の初期 primal-dual 点と内部 KKT 状態を組み立てる。
def initialize_pdipm_state(
    n_primal: int,
    n_dual_eq: int,
    n_dual_ineq: int,
    r_Hv: int = 16,
    r_Sv: int = 16,
    *,
    dtype: DTypeLike = DEFAULT_DTYPE,
) -> PDIPMState:
    # 責務: 初期 KKT 近似で使う単位 Hessian 作用素を与える。
    def _identity(v: Vector) -> Vector:
        return v

    # 責務: 等式制約が未設定の初期化で零作用素を与える。
    def _zero_dual(_: Vector) -> Vector:
        return jnp.zeros((n_dual_eq,), dtype=dtype)

    # 責務: 等式制約転置が未設定の初期化で零作用素を与える。
    def _zero_primal(_: Vector) -> Vector:
        return jnp.zeros((n_primal,), dtype=dtype)

    kkt_state = initialize_kkt_state(
        Hv_initial=LinOp(_identity),
        Bv_initial=LinOp(_zero_dual),
        BTv_initial=LinOp(_zero_primal),
        n_primal=n_primal,
        n_dual=n_dual_eq,
        r_Hv_min=r_Hv,
        r_Sv_min=r_Sv,
        dtype=dtype,
    )
    return PDIPMState(
        kkt_state=kkt_state,
        x=jnp.zeros((n_primal,), dtype=dtype),
        lam_eq=jnp.zeros((n_dual_eq,), dtype=dtype),
        lam_ineq=jnp.ones((n_dual_ineq,), dtype=dtype),
        s=jnp.ones((n_dual_ineq,), dtype=dtype),
    )


# 責務: Mehrotra 型 PDIPM を 1 問題分まとめて解き切る。
def _pdipm_solve(
    f_opt: Callable[[Vector], Scalar],
    c_eq: Callable[[Vector], Vector],
    c_ineq: Callable[[Vector], Vector],
    optimizer_state: PDIPMState,
    n_primal: int,
    m_eq: int,
    m_ineq: int,
    *,
    reset: bool = False,
    # Mehrotra fraction-to-boundary の τ（典型 0.995）
    alpha_max: Scalar,
    max_steps: int = 30,
    ipm_tol: Scalar = EPS,
    # inexact Newton forcing（KKT相対残差をIPM残差から決める）
    kkt_rtol_min: Scalar,
    kkt_rtol_max: Scalar,
    kkt_forcing_c: Scalar,
    kkt_forcing_alpha: Scalar,
    dtype: DTypeLike,
) -> tuple[Scalar, PDIPMState, Dict[str, Any]]:
    # ---- init ----
    if reset:
        x_init = jnp.zeros((n_primal,), dtype=dtype)
        lam_eq_init = jnp.zeros((m_eq,), dtype=dtype)
        lam_ineq_init = jnp.ones((m_ineq,), dtype=dtype)
        s_init = jnp.ones((m_ineq,), dtype=dtype)
        kkt_state_init = optimizer_state.kkt_state
    else:
        x_init = optimizer_state.x
        lam_eq_init = optimizer_state.lam_eq
        lam_ineq_init = optimizer_state.lam_ineq
        s_init = optimizer_state.s
        kkt_state_init = optimizer_state.kkt_state

    grad_f = jax.grad(f_opt)

    # ---- fraction-to-boundary (Mehrotra; primal/dual separate) ----
    # 責務: 正の領域を保つ最大ステップ長を成分ごとに評価する。
    def frac_to_boundary(v: Vector, dv: Vector, tau: Scalar) -> Scalar:
        mask = dv < ZERO
        cand = jnp.where(mask, -v / dv, jnp.inf)
        a = jnp.min(cand)
        # dv>=0 only -> a=inf -> min(1, tau*inf)=1
        a = jnp.minimum(ONE, tau * a)
        return jnp.minimum(a, ONE)

    # 責務: 反復回数と IPM 残差に基づいて継続条件を判定する。
    def cond(carry: Tuple[Any, ...]) -> Scalar:
        step_count, res, *_ = carry
        return jnp.logical_and(step_count < max_steps, res > ipm_tol)

    # 責務: 線形化・KKT 求解・Mehrotra 更新を 1 ステップ進める。
    def pdipm_step(carry: Tuple[Any, ...]) -> Tuple[Any, ...]:
        step_count, _, x, lam_eq, lam, s, kkt_state = carry

        # ========= 1) 一度だけ：値・ヤコビアン・VJP・勾配を作る（重複排除） =========
        # constraints values
        ce = c_eq(x)
        ci = c_ineq(x)

        # residuals: primal feasibility in equality form
        r_eq = ce
        r_ineq = ci + s

        # VJP (J^T) at current x
        # _, vjp_eq = eqx.filter_vjp(c_eq, x)
        # _, vjp_ineq = eqx.filter_vjp(c_ineq, x)

        # # J operators at current x (for J*v)
        # _, J_eq_lin = jax.linearize(c_eq, x)
        # _, J_ineq_lin = jax.linearize(c_ineq, x)

        ce, J_eq = linearize(c_eq, x)
        ci, J_ineq = linearize(c_ineq, x)
        # J_eq: LinOp = LinOp(J_eq_lin)
        # J_ineq: LinOp = LinOp(J_ineq_lin)

        # def _vjp_eq(w: Vector) -> Vector:
        #     return vjp_eq(w)[0]

        # def _vjp_ineq(w: Vector) -> Vector:
        #     return vjp_ineq(w)[0]

        _, J_eq_T = adjoint(c_eq, x)
        _, J_ineq_T = adjoint(c_ineq, x)

        # J_eq_T: LinOp = LinOp(_vjp_eq)
        # J_ineq_T: LinOp = LinOp(_vjp_ineq)

        # grad f at current x
        g = grad_f(x)

        # stationarity residual vector r_x = ∇f + J_eq^T lam_eq + J_ineq^T lam
        r_x = g + J_eq_T @ lam_eq + J_ineq_T @ lam

        # complementarity measure
        m = jnp.asarray(m_ineq, dtype=dtype)
        mu = jnp.dot(s, lam) / m
        ones = jnp.ones_like(s)

        # diag ops (S_inv uses safe divisor to avoid NaN; positivity should be ensured by step lengths)
        safe_s = jnp.maximum(s, AVOID_ZERO_DIV)
        S_inv_vec = ONE / safe_s
        diag_S = LinOp(lambda v: s * v)
        diag_S_inv = LinOp(lambda v: S_inv_vec * v)
        diag_Lam = LinOp(lambda v: lam * v)

        # Hessian-vector product of ∇_x L(x, lam_eq, lam) at current x
        # NOTE: predictor/corrector で2回KKTを解くが、どちらも「現在の (x, lam_eq, lam)」の線形化を使うので1回で良い
        # 責務: 現在点でのラグランジアンを定義して Hessian 線形化に渡す。
        def _L_for_H(xx: Vector) -> Scalar:
            return f_opt(xx) + jnp.dot(lam_eq, c_eq(xx)) + jnp.dot(lam, c_ineq(xx))

        grad_L = jax.grad(_L_for_H)
        # _, H_L = jax.linearize(grad_L, x)

        _, H_L = linearize(grad_L, x)

        # H_eff = H_L + J_ineq^T diag(lam/s) J_ineq
        w = lam * S_inv_vec  # lam/s
        diag_W = LinOp(lambda v: w * v)
        # def _h_eff(v: Vector) -> Vector:
        #     return H_L(v) + J_ineq_T(diag_W @ J_ineq(v))

        # H_eff: LinOp = LinOp(_h_eff)

        H_eff: LinearOperator = H_L + J_ineq_T * diag_W * J_ineq
        # ============================================================
        # PRE: relative residuals (no extra differentiation)
        # ============================================================
        # ========= 2) KKT solver accuracy (inexact Newton forcing) =========

        R_x = jnp.linalg.norm(r_x)
        R_eq = jnp.linalg.norm(r_eq)
        R_ineq = jnp.linalg.norm(r_ineq)

        g_norm = jnp.linalg.norm(g)
        jtlam_norm = jnp.linalg.norm(r_x - g)  # = ||J^T λ||
        R_x_rel = R_x / (ONE + g_norm + jtlam_norm)

        R_eq_rel = R_eq / (ONE + R_eq)

        R_ineq_rel = R_ineq / (ONE + jnp.linalg.norm(ci) + jnp.linalg.norm(s))

        comp_vec = s * lam
        mu = jnp.sum(comp_vec) / m
        R_c = jnp.linalg.norm(comp_vec - mu)
        R_c_rel = R_c / (ONE + jnp.linalg.norm(comp_vec))

        ipm_res = jnp.maximum(jnp.maximum(R_x_rel, jnp.maximum(R_eq_rel, R_ineq_rel)), R_c_rel)

        kkt_rtol = kkt_forcing_c * jnp.power(
            jnp.maximum(ipm_res, AVOID_ZERO_DIV), kkt_forcing_alpha
        )
        kkt_rtol = jnp.clip(kkt_rtol, kkt_rtol_min, kkt_rtol_max)

        # ========= 3) direction solver (Mehrotra form) =========
        # We pass r_c_used such that: S dlam + Lam ds = - r_c_used
        # Schur-eliminated:
        # (H + J^T S^{-1}Lam J) dx + J_eq^T dlam_eq = -r_x + J^T S^{-1}(r_c - Lam r_ineq)
        # 責務: 指定した相補残差に対する Newton 方向を KKT 解法で求める。
        def solve_direction(r_c_used: Vector, kkt_state_in: KKTState):
            rhs_top = -r_x + (J_ineq_T * diag_S_inv) @ (r_c_used - diag_Lam @ r_ineq)
            rhs_bot = -r_eq

            (dx, dlam_eq_dir), kkt_state_out, _ = kkt_block_solver(
                Hv=H_eff,
                Bv=J_eq,
                BTv=J_eq_T,
                rhs_x=rhs_top,
                rhs_lam=rhs_bot,
                kkt_state=kkt_state_in,
                kkt_tol=kkt_rtol,  # ★ 外部から決めた相対残差
                maxiter=5000,
                dtype=dtype,
            )

            ds = -(r_ineq + J_ineq @ dx)
            dlam_dir = -(diag_S_inv @ (r_c_used + diag_Lam @ ds))
            return dx, dlam_eq_dir, ds, dlam_dir, kkt_state_out, rhs_top

        # ========= 4) Mehrotra predictor–corrector (原著の形に寄せる) =========
        tau_fb = alpha_max

        # (a) affine predictor: target mu_aff = 0
        r_c_aff = diag_S @ lam  # S*lam - 0*1
        dx_aff, dlam_eq_aff, ds_aff, dlam_aff, kkt_state_mid, rhs_top_aff = solve_direction(
            r_c_aff, kkt_state
        )

        alpha_aff_pri = frac_to_boundary(s, ds_aff, tau_fb)
        alpha_aff_dual = frac_to_boundary(lam, dlam_aff, tau_fb)

        s_aff = s + alpha_aff_pri * ds_aff
        lam_aff = lam + alpha_aff_dual * dlam_aff
        mu_aff = jnp.dot(s_aff, lam_aff) / m

        # (b) centering parameter sigma = (mu_aff/mu)^3  (Mehrotra)
        sigma = jnp.power(mu_aff / jnp.maximum(mu, AVOID_ZERO_DIV), jnp.asarray(3.0, dtype=dtype))
        sigma = jnp.clip(sigma, ZERO, ONE)

        # (c) corrector: r_c = S*lam - sigma*mu*1 + (ds_aff ∘ dlam_aff)
        r_c_corr = (diag_S @ lam) - (sigma * mu) * ones + (ds_aff * dlam_aff)

        dx, dlam_eq_dir, ds, dlam_dir, kkt_state_next, rhs_top_corr = solve_direction(
            r_c_corr, kkt_state_mid
        )

        # (d) final step lengths (primal/dual separate; original Mehrotra style)
        alpha_pri = frac_to_boundary(s, ds, tau_fb)
        alpha_dual = frac_to_boundary(lam, dlam_dir, tau_fb)

        # ========= 5) update =========
        x_next = x + alpha_pri * dx
        s_next = s + alpha_pri * ds
        lam_eq_next = lam_eq + alpha_dual * dlam_eq_dir
        lam_next = lam + alpha_dual * dlam_dir

        # 原著はクリップを書かない（正の領域をステップ長で保証する前提）
        # ただし数値誤差で負になるのが怖いなら保険としてON：
        s_next = jnp.maximum(s_next, ZERO)
        lam_next = jnp.maximum(lam_next, ZERO)

        # ========= 6) DEBUG prints（不要な関数化/微分の再計算を避ける） =========
        if DEBUG:
            f_val = f_opt(x)
            x_norm = jnp.linalg.norm(x)
            s_norm = jnp.linalg.norm(s)
            lam_norm = jnp.linalg.norm(lam)
            # ============================================================
            # (A) PRE: stopping residuals (MAIN)
            # ============================================================
            jax.debug.print(
                '{{"case":"pdipm","source_file":"{source_file}",'
                '"func":"_pdipm_solve","event":"residuals","step":{step},'
                '"ipm_res":{ipm},"rx_rel":{rx},"req_rel":{req},"ri_rel":{ri},'
                '"rc_rel":{rc},"mu":{mu},"s_min":{smin},"lam_min":{lmin}}}',
                source_file=SOURCE_FILE,
                step=step_count,
                ipm=ipm_res,
                rx=R_x_rel,
                req=R_eq_rel,
                ri=R_ineq_rel,
                rc=R_c_rel,
                mu=mu,
                smin=jnp.min(s),
                lmin=jnp.min(lam),
            )
            # ============================================================
            # (A-2) additional summary
            # ============================================================
            jax.debug.print(
                '{{"case":"pdipm","source_file":"{source_file}",'
                '"func":"_pdipm_solve","event":"summary","step":{step},'
                '"obj":{obj},"x_norm":{xn},"s_norm":{sn},"lam_norm":{ln},'
                '"rx_norm":{rxn},"req_norm":{ren},"ri_norm":{rin}}}',
                source_file=SOURCE_FILE,
                step=step_count,
                obj=f_val,
                xn=x_norm,
                sn=s_norm,
                ln=lam_norm,
                rxn=R_x,
                ren=R_eq,
                rin=R_ineq,
            )

            # ============================================================
            # (B) Mehrotra predictor–corrector summary
            # ============================================================
            jax.debug.print(
                '{{"case":"pdipm","source_file":"{source_file}",'
                '"func":"_pdipm_solve","event":"mehrotra","step":{step},'
                '"alpha_aff_pri":{aap},"alpha_aff_dual":{aad},"mu_aff":{mua},'
                '"sigma":{sg},"alpha_pri":{ap},"alpha_dual":{ad}}}',
                source_file=SOURCE_FILE,
                step=step_count,
                aap=alpha_aff_pri,
                aad=alpha_aff_dual,
                mua=mu_aff,
                sg=sigma,
                ap=alpha_pri,
                ad=alpha_dual,
            )

            # ============================================================
            # (C) KKT difficulty / forcing
            # ============================================================
            jax.debug.print(
                '{{"case":"pdipm","source_file":"{source_file}",'
                '"func":"_pdipm_solve","event":"kkt","step":{step},'
                '"kkt_rtol":{rt},"rhs_aff_norm":{ra},"rhs_corr_norm":{rc}}}',
                source_file=SOURCE_FILE,
                step=step_count,
                rt=kkt_rtol,
                ra=jnp.linalg.norm(rhs_top_aff),
                rc=jnp.linalg.norm(rhs_top_corr),
            )

            # ============================================================
            # (D) POST: value-only diagnostics
            # ============================================================
            ce_next = c_eq(x_next)
            ci_next = c_ineq(x_next)
            prim_post = jnp.linalg.norm(ce_next) + jnp.linalg.norm(ci_next + s_next)

            comp_vec_next = s_next * lam_next
            mu_post = jnp.sum(comp_vec_next) / m

            jax.debug.print(
                '{{"case":"pdipm","source_file":"{source_file}",'
                '"func":"_pdipm_solve","event":"post","step":{step},'
                '"prim":{pr},"mu":{mu},"s_min":{smin},"lam_min":{lmin},"step_norm":{sdx}}}',
                source_file=SOURCE_FILE,
                step=step_count,
                pr=prim_post,
                mu=mu_post,
                smin=jnp.min(s_next),
                lmin=jnp.min(lam_next),
                sdx=jnp.linalg.norm(alpha_pri * dx),
            )

        return (step_count + 1, ipm_res, x_next, lam_eq_next, lam_next, s_next, kkt_state_next)

    ipm_res0 = jnp.asarray(jnp.inf, dtype=dtype)
    carry0 = (0, ipm_res0, x_init, lam_eq_init, lam_ineq_init, s_init, kkt_state_init)
    step_count, ipm_res_f, x_f, lam_eq_f, lam_f, s_f, kkt_state_f = jax.lax.while_loop(
        cond, pdipm_step, carry0
    )

    # ---- final info (ここだけは1回だけ微分し直してOK) ----
    ce_f = c_eq(x_f)
    ci_f = c_ineq(x_f)

    _, vjp_eq_f = eqx.filter_vjp(c_eq, x_f)
    _, vjp_ineq_f = eqx.filter_vjp(c_ineq, x_f)
    g_f = jax.grad(f_opt)(x_f)
    r_x_f = g_f + vjp_eq_f(lam_eq_f)[0] + vjp_ineq_f(lam_f)[0]

    m = jnp.asarray(m_ineq, dtype=dtype)
    mu_f = jnp.dot(s_f, lam_f) / m

    info = {
        "step_count": step_count,
        "dual_res_final": jnp.linalg.norm(r_x_f),
        "eq_res_final": jnp.linalg.norm(ce_f),
        "ineq_res_final": jnp.linalg.norm(ci_f + s_f),
        "prim_res_final": jnp.linalg.norm(ce_f) + jnp.linalg.norm(ci_f + s_f),
        "mu_final": mu_f,
        "compl_final": jnp.dot(lam_f, s_f),
    }

    if DEBUG:
        jax.debug.print(
            '{{"case":"pdipm","source_file":"{source_file}",'
            '"func":"_pdipm_solve","event":"return","step":{step},'
            '"prim_res_final":{prim},"dual_res_final":{dual},"mu_final":{mu}}}',
            source_file=SOURCE_FILE,
            step=step_count,
            prim=info["prim_res_final"],
            dual=info["dual_res_final"],
            mu=info["mu_final"],
        )
        jax.debug.print("")

    opt = f_opt(x_f)
    return (
        opt,
        PDIPMState(
            kkt_state=kkt_state_f,
            x=x_f,
            lam_eq=lam_eq_f,
            lam_ineq=lam_f,
            s=s_f,
        ),
        info,
    )


# 責務: 公開 API として既定パラメータ付きの PDIPM ソルバを提供する。
def pdipm_solve(
    f_opt: Callable[[Vector], Scalar],
    c_eq: Callable[[Vector], Vector],
    c_ineq: Callable[[Vector], Vector],
    optimizer_state: PDIPMState,
    n_primal: int,
    m_eq: int,
    m_ineq: int,
    *,
    reset: bool = False,
    alpha_max: Scalar = jnp.asarray(0.995, dtype=DEFAULT_DTYPE),
    max_steps: int = 30,
    ipm_tol: Scalar = EPS,
    kkt_rtol_min: Scalar = jnp.asarray(1e-12, dtype=DEFAULT_DTYPE),
    kkt_rtol_max: Scalar = jnp.asarray(1e-2, dtype=DEFAULT_DTYPE),
    kkt_forcing_c: Scalar = jnp.asarray(0.5, dtype=DEFAULT_DTYPE),
    kkt_forcing_alpha: Scalar = jnp.asarray(0.5, dtype=DEFAULT_DTYPE),
    dtype: DTypeLike = DEFAULT_DTYPE,
) -> tuple[Scalar, PDIPMState, Dict[str, Any]]:
    """公開用の PDIPM ソルバー。"""
    return _pdipm_solve(
        f_opt=f_opt,
        c_eq=c_eq,
        c_ineq=c_ineq,
        optimizer_state=optimizer_state,
        n_primal=n_primal,
        m_eq=m_eq,
        m_ineq=m_ineq,
        reset=reset,
        alpha_max=jnp.asarray(alpha_max, dtype=dtype),
        max_steps=max_steps,
        ipm_tol=ipm_tol,
        kkt_rtol_min=jnp.asarray(kkt_rtol_min, dtype=dtype),
        kkt_rtol_max=jnp.asarray(kkt_rtol_max, dtype=dtype),
        kkt_forcing_c=jnp.asarray(kkt_forcing_c, dtype=dtype),
        kkt_forcing_alpha=jnp.asarray(kkt_forcing_alpha, dtype=dtype),
        dtype=dtype,
    )


__all__ = [
    "PDIPMState",
    "initialize_pdipm_state",
    "_pdipm_solve",
    "pdipm_solve",
]
