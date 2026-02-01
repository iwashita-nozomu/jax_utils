from __future__ import annotations

if __name__ == "__main__":
    
    import os
    # 必要なら GPU 固定（不要なら消してOK）
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import equinox as eqx
import jax
from jax import numpy as jnp

from _env_value import AVOID_ZERO_DIV, DEBUG, DEFAULT_DTYPE, EPS, ONE, ZERO
from kkt_solver import KKTState, initialize_kkt_state, kkt_block_solver
from _linop_utils import DiagOp, LinOp
from _type_aliaces import Hom, Matrix, Scalar, Vector

class PDIPMState(eqx.Module):
    kkt_state: KKTState
    x: Vector
    lam_eq: Vector
    lam_ineq: Vector
    s: Vector

def initialize_pdipm_state(
    n_primal:int,
    n_dual_eq:int,
    n_dual_ineq:int,
    r_Hv:int=16,
    r_Sv:int=16,
    r_max:int=64,
)->PDIPMState:
    def _identity(v: Vector) -> Vector:
        return v

    def _zero_dual(_: Vector) -> Vector:
        return jnp.zeros((n_dual_eq,), dtype=DEFAULT_DTYPE)

    def _zero_primal(_: Vector) -> Vector:
        return jnp.zeros((n_primal,), dtype=DEFAULT_DTYPE)

    kkt_state=initialize_kkt_state(
        Hv_initial=_identity,
        Bv_initial=_zero_dual,
        BTv_initial=_zero_primal,
        n_primal=n_primal,
        n_dual=n_dual_eq,
        r_Hv_min=r_Hv,
        r_Sv_min=r_Sv,
        r_max=r_max,
    )
    return PDIPMState(
        kkt_state=kkt_state,
        x=jnp.zeros((n_primal,),dtype=DEFAULT_DTYPE),
        lam_eq=jnp.zeros((n_dual_eq,),dtype=DEFAULT_DTYPE),
        lam_ineq=jnp.ones((n_dual_ineq,),dtype=DEFAULT_DTYPE),
        s=jnp.ones((n_dual_ineq,),dtype=DEFAULT_DTYPE),
    )
def _pdipm_solve(
    f_opt: Hom[Vector, Scalar],
    c_eq: Hom[Vector, Vector],
    c_ineq: Hom[Vector, Vector],
    optimizer_state: PDIPMState,
    n_primal: int,
    m_eq: int,
    m_ineq: int,
    eta: Scalar = jnp.asarray(0.1, dtype=DEFAULT_DTYPE),  # 未使用に近い（残すなら保険のcentering下限等に）
    *,
    reset: bool = False,
    # Mehrotra fraction-to-boundary の τ（典型 0.995）
    alpha_max: Scalar = jnp.asarray(0.995, dtype=DEFAULT_DTYPE),
    max_steps: int = 30,
    ipm_tol: Scalar = EPS,
    log_every: int = 1,  # ※ while_loop内でPython ifできないので、DEBUG時は毎回出す（今までの方針踏襲）
    # inexact Newton forcing（KKT相対残差をIPM残差から決める）
    kkt_rtol_min: Scalar = jnp.asarray(1e-12, dtype=DEFAULT_DTYPE),
    kkt_rtol_max: Scalar = jnp.asarray(1e-2, dtype=DEFAULT_DTYPE),
    kkt_forcing_c: Scalar = jnp.asarray(0.5, dtype=DEFAULT_DTYPE),
    kkt_forcing_alpha: Scalar = jnp.asarray(0.5, dtype=DEFAULT_DTYPE),
) -> tuple[Scalar, PDIPMState, Dict[str, Any]]:
    # ---- init ----
    if reset:
        x_init = jnp.zeros((n_primal,), dtype=DEFAULT_DTYPE)
        lam_eq_init = jnp.zeros((m_eq,), dtype=DEFAULT_DTYPE)
        lam_ineq_init = jnp.ones((m_ineq,), dtype=DEFAULT_DTYPE)
        s_init = jnp.ones((m_ineq,), dtype=DEFAULT_DTYPE)
        kkt_state_init = optimizer_state.kkt_state
    else:
        x_init = optimizer_state.x
        lam_eq_init = optimizer_state.lam_eq
        lam_ineq_init = optimizer_state.lam_ineq
        s_init = optimizer_state.s
        kkt_state_init = optimizer_state.kkt_state

    # ---- fraction-to-boundary (Mehrotra; primal/dual separate) ----
    def frac_to_boundary(v: Vector, dv: Vector, tau: Scalar) -> Scalar:
        mask = dv < ZERO
        cand = jnp.where(mask, -v / dv, jnp.inf)
        a = jnp.min(cand)
        # dv>=0 only -> a=inf -> min(1, tau*inf)=1
        a = jnp.minimum(ONE, tau * a)
        return jnp.minimum(a, ONE)

    def cond(carry: Tuple[Any, ...]) -> Scalar:
        step_count, res,*_ = carry
        return jnp.logical_and(step_count < max_steps, res > ipm_tol)

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
        _, vjp_eq = eqx.filter_vjp(c_eq, x)
        _, vjp_ineq = eqx.filter_vjp(c_ineq, x)

        # J operators at current x (for J*v)
        _, J_eq_lin = jax.linearize(c_eq, x)
        _, J_ineq_lin = jax.linearize(c_ineq, x)

        J_eq: LinOp = LinOp(J_eq_lin)
        J_ineq: LinOp = LinOp(J_ineq_lin)

        def _vjp_eq(w: Vector) -> Vector:
            return vjp_eq(w)[0]

        def _vjp_ineq(w: Vector) -> Vector:
            return vjp_ineq(w)[0]

        J_eq_T: LinOp = LinOp(_vjp_eq)
        J_ineq_T: LinOp = LinOp(_vjp_ineq)

        # grad f at current x
        g = jax.grad(f_opt)(x)

        # stationarity residual vector r_x = ∇f + J_eq^T lam_eq + J_ineq^T lam
        r_x = g + vjp_eq(lam_eq)[0] + vjp_ineq(lam)[0]

        # complementarity measure
        m = jnp.asarray(m_ineq, dtype=DEFAULT_DTYPE)
        mu = jnp.dot(s, lam) / m
        ones = jnp.ones_like(s)

        # diag ops (S_inv uses safe divisor to avoid NaN; positivity should be ensured by step lengths)
        safe_s = jnp.maximum(s, AVOID_ZERO_DIV)
        S_inv_vec = ONE / safe_s
        diag_S = DiagOp(s)
        diag_S_inv = DiagOp(S_inv_vec)
        diag_Lam = DiagOp(lam)

        # Hessian-vector product of ∇_x L(x, lam_eq, lam) at current x
        # NOTE: predictor/corrector で2回KKTを解くが、どちらも「現在の (x, lam_eq, lam)」の線形化を使うので1回で良い
        def _L_for_H(xx: Vector) -> Scalar:
            return f_opt(xx) + jnp.dot(lam_eq, c_eq(xx)) + jnp.dot(lam, c_ineq(xx))

        grad_L = jax.grad(_L_for_H)
        _, H_L = jax.linearize(grad_L, x)

        # H_eff = H_L + J_ineq^T diag(lam/s) J_ineq
        w = lam * S_inv_vec  # lam/s
        diag_W = DiagOp(w)
        def _h_eff(v: Vector) -> Vector:
            return H_L(v) + J_ineq_T(diag_W @ J_ineq(v))

        H_eff: LinOp = LinOp(_h_eff)

        # ============================================================
        # PRE: relative residuals (no extra differentiation)
        # ============================================================
        # ========= 2) KKT solver accuracy (inexact Newton forcing) =========

        R_x = jnp.linalg.norm(r_x)
        R_eq = jnp.linalg.norm(r_eq)
        R_ineq = jnp.linalg.norm(r_ineq)
    
        g_norm = jnp.linalg.norm(g)
        jtlam_norm = jnp.linalg.norm(r_x - g)   # = ||J^T λ||
        R_x_rel = R_x / (ONE + g_norm + jtlam_norm)

        R_eq_rel = R_eq / (ONE + R_eq)
        
        R_ineq_rel = R_ineq / (ONE + jnp.linalg.norm(ci) + jnp.linalg.norm(s))

        comp_vec = s * lam
        mu = jnp.sum(comp_vec) / m
        R_c = jnp.linalg.norm(comp_vec - mu)
        R_c_rel = R_c / (ONE + jnp.linalg.norm(comp_vec))

        ipm_res = jnp.maximum(jnp.maximum(R_x_rel, jnp.maximum(R_eq_rel, R_ineq_rel)), R_c_rel)

        kkt_rtol = kkt_forcing_c * jnp.power(jnp.maximum(ipm_res, AVOID_ZERO_DIV), kkt_forcing_alpha)
        kkt_rtol = jnp.clip(kkt_rtol, kkt_rtol_min, kkt_rtol_max)

        # ========= 3) direction solver (Mehrotra form) =========
        # We pass r_c_used such that: S dlam + Lam ds = - r_c_used
        # Schur-eliminated:
        # (H + J^T S^{-1}Lam J) dx + J_eq^T dlam_eq = -r_x + J^T S^{-1}(r_c - Lam r_ineq)
        def solve_direction(r_c_used: Vector, kkt_state_in: KKTState):
            rhs_top = -r_x + J_ineq_T(diag_S_inv @ (r_c_used - diag_Lam @ r_ineq))
            rhs_bot = -r_eq

            dx, dlam_eq_dir, kkt_state_out = kkt_block_solver(
                Hv=H_eff,
                Bv=J_eq,
                BTv=J_eq_T,
                rhs_x=rhs_top,
                rhs_lam=rhs_bot,
                kkt_state=kkt_state_in,
                kkt_tol=kkt_rtol,  # ★ 外部から決めた相対残差
                maxiter = 5000,
            )

            ds = -(r_ineq + J_ineq(dx))
            dlam_dir = -(diag_S_inv @ (r_c_used + diag_Lam @ ds))
            return dx, dlam_eq_dir, ds, dlam_dir, kkt_state_out, rhs_top

        # ========= 4) Mehrotra predictor–corrector (原著の形に寄せる) =========
        tau_fb = alpha_max

        # (a) affine predictor: target mu_aff = 0
        r_c_aff = (diag_S @ lam)  # S*lam - 0*1
        dx_aff, dlam_eq_aff, ds_aff, dlam_aff, kkt_state_mid, rhs_top_aff = solve_direction(r_c_aff, kkt_state)

        alpha_aff_pri = frac_to_boundary(s, ds_aff, tau_fb)
        alpha_aff_dual = frac_to_boundary(lam, dlam_aff, tau_fb)

        s_aff = s + alpha_aff_pri * ds_aff
        lam_aff = lam + alpha_aff_dual * dlam_aff
        mu_aff = jnp.dot(s_aff, lam_aff) / m

        # (b) centering parameter sigma = (mu_aff/mu)^3  (Mehrotra)
        sigma = jnp.power(mu_aff / jnp.maximum(mu, AVOID_ZERO_DIV), jnp.asarray(3.0, dtype=DEFAULT_DTYPE))
        sigma = jnp.clip(sigma, ZERO, ONE)

        # (c) corrector: r_c = S*lam - sigma*mu*1 + (ds_aff ∘ dlam_aff)
        r_c_corr = (diag_S @ lam) - (sigma * mu) * ones + (ds_aff * dlam_aff)

        dx, dlam_eq_dir, ds, dlam_dir, kkt_state_next, rhs_top_corr = solve_direction(r_c_corr, kkt_state_mid)

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
            # ============================================================
            # (A) PRE: stopping residuals (MAIN)
            # ============================================================
            jax.debug.print(
                "[k={k}] ipm_res={ipm:.3e} "
                "(Rx,Req,Ri,Rc)rel=({rx:.2e},{req:.2e},{ri:.2e},{rc:.2e}) "
                "mu={mu:.3e} min(s)={smin:.2e} min(lam)={lmin:.2e}",
                k=step_count,
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
            # (B) Mehrotra predictor–corrector summary
            # ============================================================
            jax.debug.print(
                "  mehrotra: "
                "a_aff(pri,dual)=({aap:.3f},{aad:.3f}) "
                "mu_aff={mua:.3e} sigma={sg:.3e} "
                "a(pri,dual)=({ap:.3f},{ad:.3f})",
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
                "  kkt: rtol={rt:.1e} ||rhs_aff||={ra:.2e} ||rhs_corr||={rc:.2e}",
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
                "  post: prim={pr:.2e} mu={mu:.3e} "
                "min(s)={smin:.2e} min(lam)={lmin:.2e} ||a*dx||={sdx:.2e}",
                pr=prim_post,
                mu=mu_post,
                smin=jnp.min(s_next),
                lmin=jnp.min(lam_next),
                sdx=jnp.linalg.norm(alpha_pri * dx),
            )




        return (step_count + 1, ipm_res, x_next, lam_eq_next, lam_next, s_next, kkt_state_next)

    ipm_res0 = jnp.asarray(jnp.inf, dtype=DEFAULT_DTYPE) 
    carry0 = (0, ipm_res0,x_init, lam_eq_init, lam_ineq_init, s_init, kkt_state_init)
    step_count, ipm_res_f,x_f, lam_eq_f, lam_f, s_f, kkt_state_f = jax.lax.while_loop(cond, pdipm_step, carry0)

    # ---- final info (ここだけは1回だけ微分し直してOK) ----
    ce_f = c_eq(x_f)
    ci_f = c_ineq(x_f)

    _, vjp_eq_f = eqx.filter_vjp(c_eq, x_f)
    _, vjp_ineq_f = eqx.filter_vjp(c_ineq, x_f)
    g_f = jax.grad(f_opt)(x_f)
    r_x_f = g_f + vjp_eq_f(lam_eq_f)[0] + vjp_ineq_f(lam_f)[0]

    m = jnp.asarray(m_ineq, dtype=DEFAULT_DTYPE)
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


@dataclass
class Problem:
    n: int
    meq: int
    mineq_lin: int
    mineq_nonlin: int
    Q: Matrix              # (n,n) SPD-ish
    q: Vector              # (n,)
    Aeq: Matrix            # (meq,n)
    beq: Vector            # (meq,)
    Aiq: Matrix            # (mineq_lin,n)
    biq: Vector            # (mineq_lin,)
    C: Matrix              # (mineq_nonlin,n)
    d: Vector              # (mineq_nonlin,)
    x0: Vector             # (n,) feasible point


def make_problem(
    key: Vector,
    n: int = 2000,
    meq: int = 200,
    mineq_lin: int = 2000,
    mineq_nonlin: int = 2000,
    margin: float = 0.2,      # strict feasibility margin
) -> Problem:
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)

    # --- Choose a "known" feasible point x0 ---
    x0 = jax.random.normal(k1, (n,)) * 0.2

    # --- Objective: quadratic, strongly convex-ish ---
    # Q = I + U U^T / n  (SPD)
    U = jax.random.normal(k2, (n, 64))
    Q = jnp.eye(n) + (U @ U.T) / n
    q = jax.random.normal(k3, (n,)) * 0.1

    # --- Equalities: Aeq x = beq, with beq = Aeq x0 (so x0 feasible) ---
    Aeq = jax.random.normal(k4, (meq, n)) / jnp.sqrt(n)
    beq = Aeq @ x0

    # --- Linear inequalities: Aiq x - biq <= 0, set biq so x0 is strictly feasible ---
    Aiq = jax.random.normal(k5, (mineq_lin, n)) / jnp.sqrt(n)
    # Want Aiq x0 - biq <= -margin  ->  biq = Aiq x0 + margin
    biq = Aiq @ x0 + margin

    # --- Nonlinear inequalities: sin(Cx) - d <= 0, set d so x0 is strictly feasible ---
    C = jax.random.normal(k6, (mineq_nonlin, n)) / jnp.sqrt(n)
    # Want sin(Cx0) - d <= -margin  ->  d = sin(Cx0) + margin
    d = jnp.sin(C @ x0) + margin

    return Problem(
        n=n,
        meq=meq,
        mineq_lin=mineq_lin,
        mineq_nonlin=mineq_nonlin,
        Q=Q,
        q=q,
        Aeq=Aeq,
        beq=beq,
        Aiq=Aiq,
        biq=biq,
        C=C,
        d=d,
        x0=x0,
    )


def feasibility_report(
    c_eq: Hom[Vector, Vector],
    c_ineq: Hom[Vector, Vector],
    x: Vector,
    name: str,
) -> None:
    """可行性（実行可能性）の簡易レポート。

    ここでの目的は「問題が実行可能でない（可行解が存在しない）可能性」を切り分けることです。
    - 等式は c_eq(x)=0 が満たせるか（少なくとも提示点ではどれくらいズレているか）
    - 不等式は c_ineq(x)<=0 を満たせるか（min/max で境界からの距離を見る）

    注意:
    - これは“証明”ではなく、デバッグ用の観測です。
    - 問題が実行可能でも、初期点が悪いとPDIPMは不安定になり得ます。
    """
    ce = c_eq(x)
    ci = c_ineq(x)
    jax.debug.print(
        "[feas:{name}] ||c_eq||={ce_norm}",
        name=name,
        ce_norm=jnp.linalg.norm(ce),
    )
    jax.debug.print(
        "[feas:{name}] c_ineq[min,max]=[{ci_min},{ci_max}] (<=0 is feasible)",
        name=name,
        ci_min=jnp.min(ci),
        ci_max=jnp.max(ci),
    )


# ---- Define objective and constraints in the style you want for PDIPM ----

def f_opt_factory(p: Problem) -> Hom[Vector, Scalar]:
    # f(x) = 0.5 x^T Q x + q^T x
    def f_opt(x: Vector) -> Scalar:
        return 0.5 * (x @ (p.Q @ x)) + p.q @ x
    return f_opt


def c_eq_factory(p: Problem) -> Hom[Vector, Vector]:
    # c_eq(x) = Aeq x - beq = 0
    def c_eq(x: Vector) -> Vector:
        return p.Aeq @ x - p.beq
    return c_eq


def c_ineq_factory(p: Problem) -> Hom[Vector, Vector]:
    # Standard form for PDIPM often uses c_ineq(x) <= 0.
    # We'll stack linear and nonlinear inequalities:
    #   c_lin(x) = Aiq x - biq <= 0
    #   c_nonlin(x) = sin(Cx) - d <= 0
    def c_ineq(x: Vector) -> Vector:
        c_lin = p.Aiq @ x - p.biq
        c_nonlin = jnp.sin(p.C @ x) - p.d
        return jnp.concatenate([c_lin, c_nonlin], axis=0)
    return c_ineq


# ---- Quick sanity check ----
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    prob = make_problem(key, n=2000, meq=200, mineq_lin=2000, mineq_nonlin=2000, margin=0.2)

    f_opt = f_opt_factory(prob)
    c_eq = c_eq_factory(prob)
    c_ineq = c_ineq_factory(prob)

    # 可行性（実行可能性）を、生成時に保証している点 x0 で確認します。
    # ここで不等式が c_ineq(x0) <= 0 を満たしていなければ、問題設定の整合性が崩れています。
    feasibility_report(c_eq=c_eq, c_ineq=c_ineq, x=prob.x0, name="x0")

    # 参考: PDIPMのデフォルト初期点（reset=True の場合は x=0）でも観測します。
    # 可行な問題でも、x=0 が境界から遠いと初期反復が不安定になりやすいです。
    feasibility_report(
        c_eq=c_eq,
        c_ineq=c_ineq,
        x=jnp.zeros((prob.n,), dtype=DEFAULT_DTYPE),
        name="x=0",
    )

    pdipm_state = initialize_pdipm_state(
        n_primal=prob.n,
        n_dual_eq=prob.meq,
        n_dual_ineq=prob.mineq_lin + prob.mineq_nonlin,
        r_Hv=64,
        r_Sv=64,
        r_max=256,
    )

    opt, pdipm_state_new, info = eqx.filter_jit(_pdipm_solve)(
        f_opt=f_opt,
        c_eq=c_eq,
        c_ineq=c_ineq,
        optimizer_state=pdipm_state,
        n_primal=prob.n,
        m_eq=prob.meq,
        m_ineq=prob.mineq_lin + prob.mineq_nonlin,
        eta=jnp.asarray(0.1, dtype=DEFAULT_DTYPE),
        reset=True,
        alpha_max=jnp.asarray(0.995, dtype=DEFAULT_DTYPE),
        max_steps=30,
        log_every=1,
    )

    jax.debug.print("PDIPM finished in steps: {s}", s=info["step_count"])
    jax.debug.print("Optimal value estimate: {v}", v=opt)




    