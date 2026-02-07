from __future__ import annotations

from typing import Any, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from ..base import (
    AVOID_ZERO_DIV,
    DEFAULT_DTYPE,
    HALF,
    LinOp,
    LinearOperator,
    Matrix,
    Integer,
    ONE,
    Scalar,
    Vector,
    WEAK_EPS,
)
from .matrix_util import orthonormalize


# ================================================================
#  内部: block preconditioned Rayleigh–Ritz 固有値ソルバ
# ================================================================

class BlockEigenState(eqx.Module):
    """SPD 行列 H に対する block preconditioned Rayleigh–Ritz の状態.

    X : 近似固有ベクトル (n, r)
    AX : H X (n, r)
    P : 検索方向 (n, r)
    AP : H P (n, r)
    eigvals : 近似固有値 (r,)
    iteration : 反復回数（ウォームスタート用メタ情報）
    """

    X: Matrix
    AX: Matrix
    P: Matrix
    AP: Matrix
    eigenvalues: Vector
    iteration: Integer

def init_block_eigen_state(Mv: LinearOperator, X0: Matrix) -> BlockEigenState:
    """BlockEigState の 1 回目初期化."""
    AX0 = Mv @ X0
    P0 = jnp.zeros_like(X0)
    AP0 = jnp.zeros_like(X0)
    eig0 = jnp.zeros((X0.shape[1],), dtype=X0.dtype)
    return BlockEigenState(
        X=X0,
        AX=AX0,
        P=P0,
        AP=AP0,
        eigenvalues=eig0,
        iteration=jnp.asarray(0, dtype=jnp.int32),
    )

def _block_preconditioned_rayleigh_ritz(
    Mv: LinearOperator,
    T_mv: LinearOperator,
    state: BlockEigenState,
    *,
    projection: LinearOperator | None = None,
    maxiter: int = 50,
    tol: Scalar = WEAK_EPS,
    which: str = "smallest",
) -> Tuple[Matrix, Vector, BlockEigenState, Dict[str, Any]]:
    """Preconditioned block Rayleigh–Ritz eigen solver (LOBPCG 系).

    - SPD H に対して、最小 (or 最大) 固有値 r 本を求める。
    - 各ステップで trial subspace S = [X, T(HX - XΛ), P] を作り、
      その上で Rayleigh–Ritz を行う (3-term LOBPCG)。
    - S は QR で直交化するので Gram 行列は常に I (B=I)。
      → 一般化固有値問題ではなく普通の eigenproblem を解けばよい。

    projection:
        (n,r) または (n,) のベクトル/ブロックに左から作用する射影（LinOp）。
        例: v -> v - U (U^T v) など。
        LOBPCG 内の X, R, W, P, trial-subspace の生成に対して適用される。

    注意:
        projection のデフォルト値を `LinOp(lambda x: x)` のように書くと、
        型推論（特に静的解析）が破綻しやすいため、関数内で恒等写像を設定する。
    """

    if projection is None:
        # デフォルトは恒等写像（何もしない射影）とする。
        def _identity_projection(x: Matrix, /) -> Matrix:
            return x

        projection = LinOp(_identity_projection)

    # ---- 初期 X を直交化し、射影してからもう一度直交化 ----
    X_base, _ = jnp.linalg.qr(state.X)  # 念のため直交化
    X_proj = projection @ X_base
    X = orthonormalize(X_proj)

    AX = Mv @ X      # (n, r)
    P = jnp.zeros_like(X)        # (n, r)
    AP = jnp.zeros_like(X)      # (n, r)
    eigvals = jnp.einsum("ir,ir->r", X, AX)   # diag(X^T AX)
    iter0 = state.iteration

    n, r = X.shape
    steps0 = jnp.array(0, dtype=jnp.int32)

    def cond_fun(carry:Any):
        Xc, AXc, Pc, APc, lambdac, steps = carry

        R = AXc - Xc * lambdac  # (n, r)

        r_norm = jnp.linalg.norm(R, axis=0)
        ax_norm = jnp.linalg.norm(AXc, axis=0)
        ax_norm = jnp.where(
            ax_norm == 0,
            ONE,
            ax_norm,
        )
        rel = r_norm / ax_norm
        max_rel = jnp.max(rel)

        not_converged = max_rel > tol
        not_maxiter = steps < maxiter
        return jnp.logical_and(not_converged, not_maxiter)

    def body_fun(carry:Tuple[Any, ...]) -> Tuple[Any, ...]:
        Xc, AXc, Pc, APc, lambdac, steps = carry

        # ---- 1. 残差 + 前処理 ----
        R:Matrix = AXc - Xc * lambdac          # (n, r)
        R_proj = projection @ R
        R = orthonormalize(R_proj)
        W:Matrix = T_mv @ R                     # (n, r)
        W = projection @ W
        # W = orthonormalize(W)

        # 既存探索方向も（数値誤差で漏れるので）射影して整える
        Pc = projection @ Pc
        Pc = Pc - Xc @ (Xc.T @ Pc)
        # Pc = orthonormalize(Pc)

        # ---- 2. trial subspace S = [X, W, P] を QR で直交化 ----
        S_raw = jnp.concatenate([Xc, W, Pc], axis=1)   # (n, 3r)
        S_raw = projection @ S_raw
        S = orthonormalize(S_raw)                     # (n, 3r), S^T S = I

        # ---- 3. S 上での Rayleigh–Ritz ----
        HS = Mv @ S                     # (n, 3r)
        A_small = S.T @ HS               # (3r, 3r), 対称
        A_small = HALF * (A_small + A_small.T)

        theta_all, Y_all = jnp.linalg.eigh(A_small)  # 昇順

        if which == "largest":
            idx = jnp.argsort(theta_all)[::-1][:r]
        else:
            idx = jnp.argsort(theta_all)[:r]

        theta = theta_all[idx]           # (r,)
        Y = Y_all[:, idx]                # (3r, r)

        # 係数を X / W / P 部分に分割
        Y_X = Y[0:r, :]          # (r, r)
        Y_W = Y[r:2*r, :]        # (r, r)
        Y_P = Y[2*r:3*r, :]      # (r, r)

        # ---- 4. 新しい Ritz ベクトル X_new ----
        X_new = S @ Y                    # (n, r)
        X_new = projection @ X_new
        X_new = orthonormalize(X_new)
        AX_new = Mv @ X_new             # (n, r)

        # ---- 4'. 新しい検索方向 P_new ----
        # W と P 部分だけで新しい P を作る (標準的な LOBPCG の取り方の一つ)
        #   P_new = W * Y_W + P * Y_P
        # ただし W, P 自体は S の中に含まれているので、
        #   S[:, r:2r] -> W 成分,  S[:, 2r:3r] -> P 成分
        S_W = S[:, r:2*r]         # (n, r)
        S_P = S[:, 2*r:3*r]       # (n, r)

        P_new = S_W @ Y_W + S_P @ Y_P   # (n, r)

        # X_new に直交化しておく
        P_new = P_new - X_new @ (X_new.T @ P_new)
        P_new = projection @ P_new
        # P_new = P_new - X_new @ (X_new.T @ P_new)
        P_new = orthonormalize(P_new)
        AP_new = Mv @ P_new            # (n, r)

        # ---- 5. Rayleigh quotient でもう一度 λ を整える ----
        RQ = X_new.T @ AX_new            # (r, r)
        lam, V = jnp.linalg.eigh(RQ)

        if which == "largest":
            idx2 = jnp.argsort(lam)[::-1]
        else:
            idx2 = jnp.argsort(lam)

        lam = lam[idx2]                  # (r,)
        V = V[:, idx2]

        # Ritz ベクトルの並び替え
        X_new = X_new @ V                # (n, r)
        AX_new = AX_new @ V              # (n, r)
        P_new = P_new @ V                # (n, r)
        AP_new = AP_new @ V              # (n, r)

        steps_new = steps + 1
        return (X_new, AX_new, P_new, AP_new, lam, steps_new)

    carry0 = (X, AX, P, AP, eigvals, steps0)
    X_f, AX_f, P_f, AP_f, lam_f, steps_f = jax.lax.while_loop(
        cond_fun, body_fun, carry0
    )

    R_f = AX_f - X_f * lam_f  # (n, r)

    r_norm = jnp.linalg.norm(R_f, axis=0)
    ax_norm = jnp.linalg.norm(AX_f, axis=0)
    ax_norm = jnp.where(
        ax_norm == 0,
        ONE,
        ax_norm,
    )
    rel = r_norm / ax_norm
    rel_f = jnp.max(rel)


    info = {
        "num_iter": steps_f,
        "final_rel_residual": rel_f,
        "converged": rel_f <= tol,
    }
    new_state = BlockEigenState(
        X=X_f,
        AX=AX_f,
        P=P_f,
        AP=AP_f,
        eigenvalues=lam_f,
        iteration=iter0 + steps_f,
    )

    return X_f, lam_f, new_state, info


# ================================================================
#  ユーザー向け: スペクトル前処理モジュール (4つのインターフェース)
# ================================================================


class SubspaceBasis(eqx.Module):
    """block preconditioned RR で作る H^{-1} 用 rank-r スペクトル前処理の状態."""

    Q: Matrix         # (n, r)
    eigenvalues: Vector    # (r,)
    # dtype: Any
    @staticmethod
    def from_state(
        state: BlockEigenState,
    ) -> SubspaceBasis:
        """BlockEigenState から SubspaceBasis を作る。"""
        return SubspaceBasis(
            Q=state.X,
            eigenvalues=state.eigenvalues,
        )



def init_spectral_precond(
    Mv: LinearOperator,
    n: int,
    r: int,
    *,
    which: str = "smallest",
    dtype: DTypeLike = DEFAULT_DTYPE,
) -> BlockEigenState:
    """H, base_precond を使って初期のスペクトル前処理状態を作る。

    1) X0 をランダムに生成して直交化
    2) Rayleigh quotient RQ = Xᵀ H X の固有分解で初期 Ritz ペア
    """

    if n < r:
        raise ValueError(f"init_spectral_precond: n={n} < r={r}")

    key = jax.random.PRNGKey(0)

    X0 = jax.random.normal(key, (n, r), dtype=dtype)
    X = orthonormalize(X0)

    # AX = jax.vmap(Mv,in_axes=1,out_axes=1)(X)?
    AX = Mv @ X
    RQ = X.T @ AX
    lam, V = jnp.linalg.eigh(RQ)

    if which == "largest":
        idx = jnp.argsort(lam)[::-1]
    else:
        idx = jnp.argsort(lam)

    lam = lam[idx]
    V = V[:, idx]

    X = X @ V
    AX = AX @ V


    
    eig_state = BlockEigenState(
        X=X,
        AX=AX,
        P=jnp.zeros_like(X),
        AP=jnp.zeros_like(AX),
        eigenvalues=lam,
        iteration=jnp.asarray(0, dtype=jnp.int32),
    )

    return eig_state

def update_subspace(
    Mv: LinearOperator,
    base_precond: LinearOperator,
    old_state: BlockEigenState,
    *,
    maxiter: int = 50,
    tol: Scalar = WEAK_EPS,
    which: str = "smallest",
) -> Tuple[SubspaceBasis, BlockEigenState, Dict[str, Any]]:
    """block preconditioned RR で Q, λ を更新する。"""


    Q_new, eigvals_new, eig_state_new, info = _block_preconditioned_rayleigh_ritz(
        Mv=Mv,
        T_mv=base_precond,
        state=old_state,
        maxiter=maxiter,
        tol=tol,
        which=which,
    )
    Q_orth = orthonormalize(Q_new)
    # if DEBUG:
    #     jax.debug.print("eigenvalues({which}): {lam}", which=which, lam=eigvals_new)
    #     qtq = Q_orth.T @ Q_orth
    #     r = Q_orth.shape[1]
    #     ident = jnp.eye(r, dtype=Q_orth.dtype)
    #     ortho_err = jnp.linalg.norm(qtq - ident)
    #     jax.debug.print("Q orthonormality ||Q^TQ-I||_F: {err}", err=ortho_err)
    subspace_basis = SubspaceBasis(
        Q=Q_orth,
        eigenvalues=eigvals_new,
    )
    return subspace_basis, eig_state_new,info

#射影関数
def apply_projection(
    v: Matrix,
    state: SubspaceBasis,
) -> Matrix:
    """
    I - Q Q^T による射影を作用させる。
    """
    Q = state.Q          # (n, r)

    return v - Q @ (Q.T @ v)

def make_rank_r_spectral_precond(
    basis: SubspaceBasis,
    base_precond: LinearOperator = LinOp (lambda x: x),
) -> LinearOperator:
    """
    Factory: build a rank-r spectral-correction preconditioner

        M^{-1} v = Q (Λ+eps I)^{-1} Q^T v  +  (I-QQ^T) base_precond( (I-QQ^T) v )

    Inputs:
      - basis.Q: small-eigenvalue directions (n,r)
      - basis.eigenvalues: corresponding eigenvalues (r,)
      - base_precond: cheap preconditioner for the complement space
      - eps: diagonal shift for safety (useful if eigenvalues are tiny/noisy)
      - assume_base_precond_projects:
          True  -> base_precond already internally applies (I-QQ^T) (your earlier assumption)
          False -> we explicitly apply (I-QQ^T) on input/output around base_precond
      - reorthonormalize_Q: QR-orthonormalize Q inside this factory (recommended)

        Returns:
            - precond(v): Matrix with same shape as v.
    """
    Q : Matrix = basis.Q
    lam = basis.eigenvalues

    # Safe inverse of eigenvalues (rank-r correction uses Λ^{-1})

    inv_lam = ONE / (lam + AVOID_ZERO_DIV)

    def precond(v: Matrix, /) -> Matrix:
        # Spectral (rank-r) piece
        alpha = Q.T @ v                      # (r,)
        y_spec = Q @ (inv_lam * alpha)       # (n,)

        # v_perp = apply_projection(v, basis)
        v_perp: Matrix = v - Q @ alpha
        y_rest = base_precond @ v_perp
        # y_rest = apply_projection(y_rest, basis) # ensure orthogonal to Q

        y_rest = y_rest - Q @ (Q.T @ y_rest)

        return y_spec + y_rest
    
    return LinOp(precond)



__all__ = [
    "BlockEigenState",
    "SubspaceBasis",
    "init_spectral_precond",
    "update_subspace",
    "apply_projection",
    "make_rank_r_spectral_precond",
]
