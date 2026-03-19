"""最小残差法（MINRES）ソルバの実装。

References
----------
- Choi, S., & Saunders, M. A. (1992).
  "Solution of sparse indefinite systems of linear equations."
  SIAM journal on numerical analysis, 29(4), 1146-1173.
  https://epubs.siam.org/doi/abs/10.1137/0729071

  このモジュールは Choi–Saunders の unnormalized 形式を採用し、
  対称不定値線形系（Ax = b, A ∈ R^{n×n} symmetric）を直接求解します。
  前処理行列 M ≻ 0 を投入可能で、不安定性が生じやすい系にも対応しています。
"""

from __future__ import annotations
from typing import Any, Dict, Tuple
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from ..base import (
    AVOID_ZERO_DIV,
    DEBUG,
    DEFAULT_DTYPE,
    EPS,
    LinearOperator,
    LinOp,
    ONE,
    Scalar,
    Vector,
    ZERO,
)

from jax.typing import DTypeLike

SOURCE_FILE = Path(__file__).name


class MINRESState(eqx.Module):
    x0: Vector

    @staticmethod
    # 責務: 次回の反復に引き継ぐ初期解を状態として包む。
    def initialize(x0: Vector) -> "MINRESState":
        return MINRESState(x0=x0)


# 責務: MINRES の Givens 回転係数を数値的に安定に計算する。
def _sym_ortho(a: Scalar, b: Scalar, avoid_zero_div: Scalar) -> Tuple[Scalar, Scalar, Scalar]:
    """Algorithm 2 SymOrtho(a,b): returns (c,s,r) with r>=0, [c s; s -c][a;b]=[r;0]."""
    abs_a: Scalar = jnp.abs(a)
    abs_b: Scalar = jnp.abs(b)

    # 責務: b が 0 のとき回転を使わず a の符号だけを残す。
    def case_b0() -> Tuple[Scalar, Scalar, Scalar]:
        r = abs_a
        c = jnp.where(a == ZERO, ONE, jnp.sign(a))
        s = ZERO
        return c, s, r

    # 責務: a が 0 のとき s 側だけで回転を構成する。
    def case_a0() -> Tuple[Scalar, Scalar, Scalar]:
        r = abs_b
        c = ZERO
        s: Scalar = jnp.where(b == ZERO, ZERO, jnp.sign(b))
        return c, s, r

    # 責務: a, b がともに非零の一般ケースを分岐込みで処理する。
    def general() -> Tuple[Scalar, Scalar, Scalar]:
        # 責務: |b| >= |a| のとき b を基準に正規化して回転を作る。
        def branch1() -> Tuple[Scalar, Scalar, Scalar]:  # |b| >= |a|
            tau: Scalar = a / jnp.where(b == ZERO, avoid_zero_div, b)
            s: Scalar = jnp.sign(b) / jnp.sqrt(ONE + tau * tau)
            c: Scalar = s * tau
            r: Scalar = b / jnp.where(s == ZERO, avoid_zero_div, s)
            return c, s, jnp.abs(r)

        # 責務: |a| > |b| のとき a を基準に正規化して回転を作る。
        def branch2() -> Tuple[Scalar, Scalar, Scalar]:  # |a| > |b|
            tau: Scalar = b / jnp.where(a == ZERO, avoid_zero_div, a)
            c: Scalar = jnp.sign(a) / jnp.sqrt(ONE + tau * tau)
            s: Scalar = c * tau
            r: Scalar = a / jnp.where(c == ZERO, avoid_zero_div, c)
            return c, s, jnp.abs(r)

        return lax.cond(abs_b >= abs_a, branch1, branch2)

    # 責務: b != 0 側で a の零判定と一般ケースを切り替える。
    def case_not_b0() -> Tuple[Scalar, Scalar, Scalar]:
        return lax.cond(a == ZERO, case_a0, general)

    return lax.cond(b == ZERO, case_b0, case_not_b0)


# 責務: 前処理付き MINRES の本体反復を実行して近似解と次状態を返す。
def pminres_solve(
    Mv: LinearOperator,
    rhs: Vector,
    minres_state: MINRESState,
    *,
    Minv: LinearOperator,  # q = Minv(z) means solve M q = z, with M SPD
    maxiter: int,
    rtol: Scalar | None = None,
    atol: Scalar | None = None,
    avoid_zero_div: Scalar = AVOID_ZERO_DIV,
    proj: LinearOperator = LinOp(lambda x: x),
    dtype: DTypeLike = DEFAULT_DTYPE,
) -> Tuple[Vector, MINRESState, Dict[str, Any]]:
    """
    Preconditioned MINRES (MINRES phase only) in the *unnormalized* Choi–Saunders form.

    Key points:
      - No Lanczos normalization (do NOT force q^T z = 1).
      - Uses Choi–Saunders Algorithm 1 (lines 8–16, 22–23) + MINRES-phase dbar update.
      - Robust stopping uses TRUE residual ||b - A x|| (recommended for debugging and for safety).
      - Handles Lanczos breakdown (beta_next ~ 0) BEFORE any division to avoid NaNs.

    Assumptions:
      - A is symmetric (indef OK)
      - Minv represents an SPD preconditioner solve: q = Minv(z) = M^{-1} z, with M SPD and self-adjoint
    """

    b = proj @ rhs
    x = proj @ minres_state.x0

    bnorm = jnp.linalg.norm(b)
    atol_val = jnp.asarray(0.0, dtype) if atol is None else jnp.asarray(atol, dtype)
    rtol_val = jnp.asarray(EPS, dtype) if rtol is None else jnp.asarray(rtol, dtype)
    tol_true = jnp.maximum(atol_val, rtol_val * bnorm)

    # Initial residual
    r0 = proj @ (b - Mv @ x)
    rnorm0 = jnp.linalg.norm(r0)
    done0 = (rnorm0 <= tol_true) | (bnorm == ZERO)

    if DEBUG:
        jax.debug.print(
            '{{"case":"minres","source_file":"{source_file}",'
            '"func":"pminres_solve","event":"init",'
            '"r0_norm":{r0},"b_norm":{bn},"tol":{tol},"maxiter":{maxiter}}}',
            source_file=SOURCE_FILE,
            r0=rnorm0,
            bn=bnorm,
            tol=tol_true,
            maxiter=maxiter,
        )

    # ---- Choi–Saunders initialization (use z1 = r0) ----
    z_prev = jnp.zeros_like(r0)  # z0
    z = r0  # z1

    q = proj @ (Minv @ z)  # q1 = M^{-1} z1
    beta = jnp.sqrt(jnp.maximum(jnp.dot(q, z), ZERO))  # beta1 = sqrt(q1^T z1)

    # MINRES scalar
    phi = beta

    # Left reflection init
    c_old = -ONE
    s_old = ZERO

    # Algorithm scalars (δ1=0, ϵ1=0)
    delta_alg = ZERO
    eps_alg = ZERO

    # For Lanczos line-9 coefficient beta_k/beta_{k-1}
    beta_prev = ONE  # dummy; k==0 branch will ignore

    # MINRES phase recurrence vectors
    dbar_old = jnp.zeros_like(q)  # dbar_{k-1}
    dbar_oold = jnp.zeros_like(q)  # dbar_{k-2}

    # 責務: 反復継続条件として反復回数と収束フラグを判定する。
    def cond_fun(carry: Tuple[Any, ...]) -> bool:
        k, _x, *_rest, done = carry
        return (k < maxiter) & (~done)

    # 責務: Lanczos 1 ステップと MINRES 更新をまとめて進める。
    def body_fun(carry: Tuple[Any, ...]) -> Tuple[Any, ...]:
        (
            k,
            x,
            z_prev,
            z,
            q,
            beta_prev,
            beta,
            delta_alg,
            eps_alg,
            c_old,
            s_old,
            phi,
            dbar_old,
            dbar_oold,
            done,
        ) = carry

        # ---- Algorithm 1 line 8: p_k = A q_k ----
        p = proj @ (Mv @ q)

        # alpha_k = (q_k^T p_k) / beta_k^2
        beta_safe = jnp.where(beta < avoid_zero_div, avoid_zero_div, beta)
        beta_safe = jnp.asarray(beta_safe, dtype=dtype)
        beta2 = beta_safe * beta_safe
        alpha = jnp.dot(q, p) / beta2

        # ---- Algorithm 1 line 9: z_{k+1} recurrence (unnormalized) ----
        inv_beta = ONE / beta_safe
        coef_prev = jnp.where(
            k == 0,
            ZERO,
            beta / jnp.where(beta_prev < avoid_zero_div, avoid_zero_div, beta_prev),
        )
        z_next = inv_beta * p - (alpha * inv_beta) * z - coef_prev * z_prev
        # z_next = p - alpha * z - coef_prev * z_prev

        # ---- Algorithm 1 line 10: q_{k+1} and beta_{k+1} ----
        q_next = proj @ (Minv @ z_next)
        beta_next = jnp.sqrt(jnp.maximum(jnp.dot(q_next, z_next), ZERO))

        beta_break = beta_next <= avoid_zero_div

        # ---- Algorithm 1 lines 12-15 (previous left reflection) ----
        delta2 = c_old * delta_alg + s_old * alpha
        gamma1 = s_old * delta_alg - c_old * alpha
        eps_next = s_old * beta_next
        delta_next = -c_old * beta_next

        # ---- Algorithm 1 line 16 (current left reflection) ----
        c, s, gamma2 = _sym_ortho(gamma1, beta_next, avoid_zero_div)

        # ---- Algorithm 1 lines 22-23 ----
        tau = c * phi
        phi_new = s * phi

        # ---- MINRES-phase update (dbar and x) ----
        # dbar_k = ((1/beta_k) q_k - delta2 dbar_{k-1} - eps_k dbar_{k-2}) / gamma2
        q_over_beta = q / jnp.where(beta < avoid_zero_div, avoid_zero_div, beta)
        dbar = (q_over_beta - delta2 * dbar_old - eps_alg * dbar_oold) / jnp.where(
            gamma2 < avoid_zero_div, avoid_zero_div, gamma2
        )

        x_new = proj @ (x + tau * dbar)

        # ---- robust stopping by TRUE residual ----
        r_true = proj @ (b - Mv @ x_new)
        rnorm = jnp.linalg.norm(r_true)
        done_new = (rnorm <= tol_true) | beta_break

        # if DEBUG:
        #     rel = jnp.where(bnorm > ZERO, rnorm / bnorm, rnorm)
        #     jax.debug.print(
        #         "PMINRES k={k} rel={rel:.3e} phi={phi:.3e} beta={be:.3e} alpha={al:.3e} break={br}",
        #         k=k + 1, rel=rel, phi=phi_new, be=beta_next, al=alpha, br=beta_break
        #     )

        # If breakdown, stop without using divisions by beta_next anywhere (we already avoided)
        # 責務: Lanczos breakdown 時に現在の解で安全に反復を打ち切る。
        def branch_break(_):
            return (
                k + 1,
                x_new,
                z_prev,
                z,
                q,
                beta_prev,
                beta,
                delta_alg,
                eps_alg,
                c_old,
                s_old,
                phi_new,
                dbar_old,
                dbar_oold,
                True,
            )

        # 責務: 次反復に必要な Lanczos/MINRES 状態を更新して引き渡す。
        def branch_ok(_):
            return (
                k + 1,
                x_new,
                z,  # z_prev <- z_k
                z_next,  # z <- z_{k+1}
                q_next,  # q <- q_{k+1}
                beta,  # beta_prev <- beta_k
                beta_next,  # beta <- beta_{k+1}
                delta_next,
                eps_next,
                c,
                s,
                phi_new,
                dbar,
                dbar_old,
                done_new,
            )

        return lax.cond(beta_break, branch_break, branch_ok, operand=None)

    init = (
        0,
        x,
        z_prev,
        z,
        q,
        beta_prev,
        beta,
        delta_alg,
        eps_alg,
        c_old,
        s_old,
        phi,
        dbar_old,
        dbar_oold,
        done0,
    )

    k_f, x_f, *_ = lax.while_loop(cond_fun, body_fun, init)

    r_f = proj @ (b - Mv @ x_f)
    rnorm_f = jnp.linalg.norm(r_f)

    # JAXの `while_loop` / `jit` 内から呼ばれる場合があるため、
    # Python の `float/int/bool` への変換は行わず、JAX配列のまま返す。
    info: Dict[str, Any] = {
        "final_norm_r": rnorm_f,
        "final_rel_r": jnp.where(bnorm > ZERO, rnorm_f / bnorm, rnorm_f),
        "converged": (rnorm_f <= tol_true),
        "num_iter": k_f,
    }
    if DEBUG:
        jax.debug.print(
            '{{"case":"minres","source_file":"{source_file}",'
            '"func":"pminres_solve","event":"return",'
            '"num_iter":{num_iter},"final_norm_r":{final_norm_r},'
            '"final_rel_r":{final_rel_r},"converged":{converged}}}',
            source_file=SOURCE_FILE,
            num_iter=info["num_iter"],
            final_norm_r=info["final_norm_r"],
            final_rel_r=info["final_rel_r"],
            converged=info["converged"],
        )
        jax.debug.print("")
    return x_f, MINRESState.initialize(x0=x_f), info


# 責務: 後方互換の公開名から本体の PMINRES 実装を呼び出す。
def minres_solve(
    Mv: LinearOperator,
    rhs: Vector,
    minres_state: MINRESState,
    *,
    Minv: LinearOperator,
    maxiter: int,
    rtol: Scalar | None = None,
    atol: Scalar | None = None,
    avoid_zero_div: Scalar = AVOID_ZERO_DIV,
    proj: LinearOperator = LinOp(lambda x: x),
    dtype: DTypeLike = DEFAULT_DTYPE,
) -> Tuple[Vector, MINRESState, Dict[str, Any]]:
    return pminres_solve(
        Mv=Mv,
        rhs=rhs,
        minres_state=minres_state,
        Minv=Minv,
        maxiter=maxiter,
        rtol=rtol,
        atol=atol,
        avoid_zero_div=avoid_zero_div,
        proj=proj,
        dtype=dtype,
    )


__all__ = [
    "MINRESState",
    "minres_solve",
    "pminres_solve",
]
