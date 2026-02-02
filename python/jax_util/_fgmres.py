from __future__ import annotations
from typing import Any, Tuple, Optional, Dict

import os
if __name__ == "__main__":
    # 必要なら GPU 固定（不要なら消してOK）
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from base import *  # Array, DEFAULT_DTYPE, StatefulMvlike, etc.



# ============================================================
#  ウォームスタート用 GMRESState
# ============================================================

class GMRESState(eqx.Module):
    """GMRES のウォームスタート用スナップショット."""
    x0: Vector
    Mv_state: Any = None

    @staticmethod
    def initialize(x0: Vector, Mv_state: Any = None) -> "GMRESState":
        return GMRESState(x0=x0, Mv_state=Mv_state)


# ============================================================
#  大きいバッファは Workspace に持たせる
# ============================================================

class GMRESWorkspace(eqx.Module):
    """再起動長 restart 用のワークスペース."""
    V: Matrix
    Z: Matrix
    H: Matrix
    cs: Vector
    sn: Vector
    g: Vector

    @staticmethod
    def initialize(n: int, restart: int,) -> "GMRESWorkspace":
        V = jnp.zeros((n, restart + 1), dtype=DEFAULT_DTYPE)
        Z = jnp.zeros((n, restart), dtype=DEFAULT_DTYPE)
        H = jnp.zeros((restart + 1, restart), dtype=DEFAULT_DTYPE)
        cs = jnp.zeros((restart,), dtype=DEFAULT_DTYPE)
        sn = jnp.zeros((restart,), dtype=DEFAULT_DTYPE)
        g = jnp.zeros((restart + 1,), dtype=DEFAULT_DTYPE)
        return GMRESWorkspace(V=V, Z=Z, H=H, cs=cs, sn=sn, g=g)


def initialize_fgmres_state(n: int, restart: int,precond_state: Any) -> Tuple[GMRESWorkspace, GMRESState, Any]:
    ws = GMRESWorkspace.initialize(n=n, restart=restart)
    st = GMRESState.initialize(x0=jnp.zeros((n,), dtype=DEFAULT_DTYPE))
    return ws, st, precond_state

# ============================================================
#  射影（proj）は StatefulMvlike: (x, state) -> (Px, new_state)
#  precond も StatefulMvlike
#  Mv      も StatefulMvlike
# ============================================================

def _ws_zero(ws: GMRESWorkspace) -> GMRESWorkspace:
    """Workspace をまとめて 0 クリア."""
    return GMRESWorkspace(
        V=ws.V.at[:, :].set(ZERO),
        Z=ws.Z.at[:, :].set(ZERO),
        H=ws.H.at[:, :].set(ZERO),
        cs=ws.cs.at[:].set(ZERO),
        sn=ws.sn.at[:].set(ZERO),
        g=ws.g.at[:].set(ZERO),
    )


def _begin_cycle(
    *,
    rhs: Vector,
    x: Vector,
    avoid: Scalar,
    Mv: LinearOperator,
    ws: GMRESWorkspace,
    proj: LinearOperator,
) -> Tuple[Scalar, GMRESWorkspace, Boolean]:
    """
    サイクル先頭の残差・v0・g0・ws 初期化をまとめる.

    戻り値:
      r0_norm2 = ||P(b - Ax)||^2
      ws
      cycle_done = (beta <= avoid)  # 残差ゼロ(扱い)なら True
    """
    Ax = Mv @ x
    r0 = rhs - Ax
    r0 = proj @ r0

    r0_norm2 = jnp.dot(r0, r0)
    beta = jnp.sqrt(r0_norm2)

    # ===== FIX-1: v0 は常に正規化形にする。beta==0(扱い)ならサイクル即終了へ =====
    cycle_done = beta <= avoid
    beta_safe = jnp.where(cycle_done, ONE, beta)  # 0割回避用（cycle_doneなら値は何でもよい）
    v0 = r0 / beta_safe                           # 常に「正規化の形」

    ws = _ws_zero(ws)
    ws = GMRESWorkspace(
        V=ws.V.at[:, 0].set(v0),
        Z=ws.Z,
        H=ws.H,
        cs=ws.cs,
        sn=ws.sn,
        g=ws.g.at[0].set(beta),  # g[0]は“真のbeta”を入れる（0なら0）
    )
    return r0_norm2, ws, cycle_done


def _apply_givens_pair(c: Scalar, s: Scalar, a: Scalar, b: Scalar) -> Tuple[Scalar, Scalar]:
    ap = c * a + s * b
    bp = -s * a + c * b
    return ap, bp


def _apply_prev_givens_to_col(H: Matrix, cs: Vector, sn: Vector, j: Scalar, m: Integer) -> Matrix:
    """既存の Givens 回転（0..j-1）を H[:, j] に適用."""
    def body(idx, H_local):  # pyright: ignore
        def do_step(H_l):  # pyright: ignore
            a = H_l[idx, j]
            b = H_l[idx + 1, j]
            ap, bp = _apply_givens_pair(cs[idx], sn[idx], a, b)
            H_l = H_l.at[idx, j].set(ap)
            H_l = H_l.at[idx + 1, j].set(bp)
            return H_l

        return lax.cond(idx < j, do_step, lambda x: x, H_local)  # pyright: ignore

    H = lax.fori_loop(0, m, body, H)  # pyright: ignore
    return H


def _make_and_apply_new_givens(
    H: Matrix, g: Vector, cs: Vector, sn: Vector, j: Scalar, avoid: Scalar
) -> Tuple[Matrix, Vector, Vector, Vector, Scalar]:
    """新しい Givens を作って H と g を更新。戻り値最後は g_{j+1}（残差推定）."""
    h_ii = H[j, j]
    h_ip1i = H[j + 1, j]
    diag = jnp.sqrt(h_ii * h_ii + h_ip1i * h_ip1i)
    diag = jnp.where(diag < avoid, avoid, diag)
    c = h_ii / diag
    s = h_ip1i / diag

    cs = cs.at[j].set(c)
    sn = sn.at[j].set(s)

    # 上三角化
    H = H.at[j, j].set(diag)          # pyright: ignore
    H = H.at[j + 1, j].set(0.0)       # pyright: ignore

    # g にも同じ回転
    gj, gj1 = g[j], g[j + 1]
    gj_new, gj1_new = _apply_givens_pair(c, s, gj, gj1)
    g = g.at[j].set(gj_new)
    g = g.at[j + 1].set(gj1_new)

    return H, g, cs, sn, gj1_new


def _upper_tri_solve(H: Matrix, g: Vector, k: Scalar, avoid: Scalar) -> Vector:
    """上三角 R y = g を解く（0..k-1 が有効）。戻り値は長さ m (=H.shape[1]) のベクトル。"""
    m = H.shape[1]
    y0 = jnp.zeros((m,), dtype=H.dtype)

    def body_i(y, i_rev):  # pyright: ignore
        i = (k - 1) - i_rev

        def do_step(y):  # pyright: ignore
            def body_j(s, j):  # pyright: ignore
                use = (j > i) & (j < k)
                s = jnp.where(use, s + H[i, j] * y[j], s)
                return s, None

            s0 = jnp.asarray(0.0, dtype=H.dtype)
            s, _ = lax.scan(body_j, s0, jnp.arange(m))

            denom = jnp.where(jnp.abs(H[i, i]) < avoid, avoid, H[i, i])
            yi = (g[i] - s) / denom
            y = y.at[i].set(yi)
            return y

        y = lax.cond((i >= 0) & (i < k), do_step, lambda y: y, y)  # pyright: ignore
        return y, None

    y, _ = lax.scan(body_i, y0, jnp.arange(m))
    return y


# ============================================================
#  右前処理付き 再起動 GMRES(m)
# ============================================================
def gmres_solve(
    Mv: LinearOperator,
    precond: LinearOperator,
    rhs: Vector,
    state: Tuple[GMRESWorkspace, GMRESState, Any],
    *,
    maxiter: int = 200,
    rtol: Optional[Scalar] = EPS,    # 相対
    atol: Optional[Scalar] = None,   # 絶対
    avoid_zero_div: Scalar = AVOID_ZERO_DIV,
    proj: LinearOperator = LinOp(lambda x: x),
) -> Tuple[Vector, Tuple[GMRESWorkspace, GMRESState, Any],Dict[str, Any]]:
    """右前処理付き 再起動 GMRES(m) で A x = rhs を解く。"""
    workspace, gmres_state,precond_state = state
    dtype = rhs.dtype
    m = workspace.H.shape[1]

    # PCG流：None は -1 を入れて「常に未収束扱い」にできるようにする
    rtol_val = jnp.asarray(-1.0, dtype=dtype) if rtol is None else jnp.asarray(rtol, dtype=dtype)
    atol_val = jnp.asarray(-1.0, dtype=dtype) if atol is None else jnp.asarray(atol, dtype=dtype)

    # 二乗にして持つ（sign トリック込み）
    rtol2 = jnp.sign(rtol_val) * (rtol_val * rtol_val)
    atol2 = jnp.sign(atol_val) * (atol_val * atol_val)

    # ウォームスタート
    x_init = gmres_state.x0
    _Mv_state_init = gmres_state.Mv_state  # 現状未使用

    # グローバル初期残差（射影付き、二乗）
    Ax0 = Mv @ x_init
    r0_global = rhs - Ax0
    r0_global = proj @ r0_global
    r0_norm2_global = jnp.dot(r0_global, r0_global)

    global_tol = jnp.maximum(atol2, rtol2 * r0_norm2_global)

    init_carry = (
        jnp.asarray(0),     # i
        jnp.asarray(0),     # j
        x_init,                              # x
        x_init,                              # x0_cycle
        r0_norm2_global,                     # r0_norm2 (cycle)
        workspace,                           # ws
        precond_state,                       # prec_state
        jnp.asarray(False),                  # done
        jnp.asarray(0),                      # num_restart
    )

    def cond_fun(carry: Tuple[Any, ...]) -> bool:
        i, j, x, x0_cycle, r0_norm2, ws, prec_state, done, num_restart = carry
        return (i < maxiter) & (~done)

    def body_fun(carry: Tuple[Any, ...]) -> Tuple[Any, ...]:
        i, j, x, x0_cycle, r0_norm2, ws, prec_state, done, num_restart = carry

        # ===== サイクル先頭 =====
        def init_cycle(c: Tuple[Any, ...]) -> Tuple[Any, ...]:
            i_, j_, x_, x0_cycle_, r0_norm2_, ws_, prec_state_, done_, num_restart_ = c
            x0_cycle_ = x_
            r0_norm2_, ws_, cycle_done_ = _begin_cycle(
                rhs=rhs,
                x=x_,
                avoid=avoid_zero_div,
                Mv=Mv,
                ws=ws_,
                proj=proj,
            )
            num_restart_ = num_restart_ + 1
            done_ = done_ | cycle_done_
            return (i_, j_, x_, x0_cycle_, r0_norm2_, ws_, prec_state_, done_, num_restart_)

        carry = lax.cond(j == 0, init_cycle, lambda c: c, carry)  # pyright: ignore
        i, j, x, x0_cycle, r0_norm2, ws, prec_state, done, num_restart = carry  # pyright: ignore

        # サイクル開始直後に done なら何もしない（while が抜ける）
        def do_gmres_step(c: Tuple[Any, ...]) -> Tuple[Any, ...]:
            i, j, x, x0_cycle, r0_norm2, ws, prec_state, done, num_restart = c

            # ===== GMRES 1 ステップ =====
            v_j = ws.V[:, j]
            v_j = proj @ v_j

            # 右前処理
            z_j, prec_state = precond @ v_j, prec_state

            # ws.Z[:, j] = z_j
            Z = lax.dynamic_update_slice(ws.Z, z_j[:, None], (0, j))
            ws = GMRESWorkspace(V=ws.V, Z=Z, H=ws.H, cs=ws.cs, sn=ws.sn, g=ws.g)

            z_j = proj @ z_j
            w = proj @ Mv @ z_j

            # ----- Arnoldi -----
            def arnoldi_body(c2: Tuple[Any, ...], idx: int) -> Tuple[Tuple[Any, ...], None]:
                w_, H_col_ = c2

                def do_step(args: Tuple[Any, ...]) -> Tuple[Any, ...]:
                    w__, Hc__ = args
                    v_idx = ws.V[:, idx]
                    h = jnp.dot(w__, v_idx)
                    Hc__ = lax.dynamic_update_index_in_dim(Hc__, h, idx, axis=0)
                    w__ = w__ - h * v_idx
                    return w__, Hc__

                w_, H_col_ = lax.cond(idx <= j, do_step, lambda args: args, (w_, H_col_))  # pyright: ignore
                return (w_, H_col_), None

            H_col0 = jnp.zeros((m + 1,), dtype=ws.H.dtype)
            (w, H_col), _ = lax.scan(arnoldi_body, (w, H_col0), jnp.arange(m + 1))  # pyright: ignore

            # ===== FIX-2: happy breakdown を丸めずに処理 =====
            h_next_raw = jnp.linalg.norm(w)
            breakdown = h_next_raw <= avoid_zero_div

            def normal_path(args: Tuple[Any, ...]) -> Tuple[GMRESWorkspace, Boolean]:
                w, ws, H_col = args
                h_next = jnp.where(h_next_raw < avoid_zero_div, avoid_zero_div, h_next_raw)
                H_col_n = lax.dynamic_update_index_in_dim(H_col, h_next, j + 1, axis=0)  # pyright: ignore
                H_n = lax.dynamic_update_index_in_dim(ws.H, H_col_n, j, axis=1)
                v_next = proj @ w / h_next
                V_n = lax.dynamic_update_slice(ws.V, v_next[:, None], (0, j + 1))
                ws_n = GMRESWorkspace(V=V_n, Z=ws.Z, H=H_n, cs=ws.cs, sn=ws.sn, g=ws.g)
                return ws_n, jnp.asarray(False)

            def breakdown_path(args:Tuple[Any, ...]) -> Tuple[GMRESWorkspace, Boolean]:
                _w, ws, H_col = args
                # subdiagonalは0として扱い、Vの更新はしない
                H_col_b = lax.dynamic_update_index_in_dim(H_col, ZERO, j + 1, axis=0)  # pyright: ignore
                H_b = lax.dynamic_update_index_in_dim(ws.H, H_col_b, j, axis=1)
                ws_b = GMRESWorkspace(V=ws.V, Z=ws.Z, H=H_b, cs=ws.cs, sn=ws.sn, g=ws.g)
                return ws_b, jnp.asarray(True)

            ws, did_breakdown = lax.cond(
                breakdown,
                breakdown_path,
                normal_path,
                (w, ws, H_col),
            )

            # ----- Givens -----
            H2 = _apply_prev_givens_to_col(ws.H, ws.cs, ws.sn, j, m)
            H3, g3, cs3, sn3, gj1_new = _make_and_apply_new_givens(H2, ws.g, ws.cs, ws.sn, j, avoid_zero_div)
            ws = GMRESWorkspace(V=ws.V, Z=ws.Z, H=H3, cs=cs3, sn=sn3, g=g3)

            # 収束判定（推定残差）
            res2 = gj1_new * gj1_new
            tol2_cycle = jnp.maximum(atol2, rtol2 * r0_norm2)
            done = done | (res2 <= global_tol) | did_breakdown  # breakdownならこれ以上空間が伸びないので終了

            if DEBUG:
                # === 推定残差（GMRES内部）===
                est_res2 = res2
                est_res = jnp.sqrt(est_res2)

                # === 真の残差 ===
                r_true = proj @ (rhs - Mv @ x)
                true_res2 = jnp.dot(r_true, r_true)
                true_res = jnp.sqrt(true_res2)

                # === RHS の大きさ ===
                bnorm2 = jnp.dot(rhs, rhs)
                bnorm = jnp.sqrt(bnorm2)

                # === 相対残差 ===
                rel_est = jnp.where(bnorm > ZERO, est_res / bnorm, est_res)
                rel_true = jnp.where(bnorm > ZERO, true_res / bnorm, true_res)

                jax.debug.print(
                    "GMRES i={i} (restart={nr}) j={j}  est_res2={est2} est_res={est} rel_est={rel_est}  true_res2={tru2} true_res={tru} rel_true={rel_true}  bnorm={bn}  tol2_cycle={tol2} global_tol={gtol}  breakdown={bd} done={done}",
                    i=i,
                    nr=num_restart,
                    j=j,
                    est2=est_res2,
                    est=est_res,
                    rel_est=rel_est,
                    tru2=true_res2,
                    tru=true_res,
                    rel_true=rel_true,
                    bn=bnorm,
                    tol2=tol2_cycle,
                    gtol=global_tol,
                    bd=did_breakdown,
                    done=done,
                )
                jax.debug.print("||v_j||={}  ||z_j||={}  ||A z_j||={}",
                    jnp.linalg.norm(v_j),
                    jnp.linalg.norm(z_j),
                    jnp.linalg.norm(w))

            # 解更新（重要：k まで）
            k = j + 1
            y = _upper_tri_solve(ws.H, ws.g, k, avoid_zero_div)  # (m,)
            delta_x = ws.Z @ y
            x = proj @ (x0_cycle + delta_x)

            # カウンタ
            i = i + 1
            j_next = (j + 1) % m

            return (i, j_next, x, x0_cycle, r0_norm2, ws, prec_state, done, num_restart)

        carry = lax.cond(done, lambda c: c, do_gmres_step, (i, j, x, x0_cycle, r0_norm2, ws, prec_state, done, num_restart))  # pyright: ignore
        return carry

    carry_f = lax.while_loop(cond_fun, body_fun, init_carry)

    (
        i_f,
        j_f,
        x_f,
        x0_cycle_f,
        r0_norm2_f,
        ws_f,
        precond_state_f,
        done_f,
        num_restart_f,
    ) = carry_f

    # 最終残差（射影付きで一貫、二乗）
    Ax_f = Mv @ x_f
    r_final = proj @ (rhs - Ax_f)
    res2_final = jnp.dot(r_final, r_final)

    final_norm_r = jnp.sqrt(res2_final)
    final_rel_r = jnp.where(
        r0_norm2_global > ZERO,
        jnp.sqrt(res2_final / r0_norm2_global),
        final_norm_r,
    )

    # JAXの `while_loop` / `jit` 内から呼ばれる場合があるため、
    # Python の `float/int/bool` への変換は行わず、JAX配列のまま返す。
    info = {
        "final_norm_r": final_norm_r,
        "final_rel_r": final_rel_r,
        "converged": done_f,
        "num_iter": i_f,
        "num_restart": num_restart_f,
    }
    gmres_state_f = GMRESState(x0=x_f, Mv_state=None)
    new_state = (ws_f, gmres_state_f, precond_state_f)
    
    return x_f, new_state, info



# ============================================================
#  テストコード：dense 非対称行列で動作確認
# ============================================================

def _make_random_nonsym_system(key: Any, n: int) -> Tuple[Matrix, Vector, Vector]:
    k1, k2 = jax.random.split(key)
    M = jax.random.normal(k1, (n, n))
    A = M + 0.3 * M.T
    diag = jnp.sum(jnp.abs(A), axis=1) + 1.0
    A = A + jnp.diag(diag)  # pyright: ignore

    x_true = jax.random.normal(k2, (n,))
    b = A @ x_true
    return A, x_true, b


def _dense_Mv_from_A(A: Matrix) -> LinearOperator:
    def Mv(v: Vector) -> Vector:
        return A @ v
    return LinOp(Mv)


def _identity_precond():
    def precond(r: Vector, state: Any) -> Tuple[Vector, Any]:
        return r, state
    return precond


def test_gmres_dense():
    key = jax.random.PRNGKey(0)
    n = 50
    restart = 50

    A, x_true, b = _make_random_nonsym_system(key, n)
    Mv = _dense_Mv_from_A(A)
    precond = _identity_precond()

    x0 = jnp.zeros_like(b)
    gmres_state = GMRESState.initialize(x0=x0, Mv_state=None)
    precond_state = None
    workspace = GMRESWorkspace.initialize(n=n, restart=restart)

    x_approx, new_state, info = gmres_solve(
        Mv=Mv,
        precond=precond,
        rhs=b,
        state=(
            workspace,
            gmres_state,
            precond_state,
        ),
        maxiter=100,
    )

    print("GMRES info:", info)
    abs_err = jnp.linalg.norm(x_approx - x_true)
    rel_err = abs_err / jnp.linalg.norm(x_true)
    print(f"abs error = {float(abs_err):.3e}")
    print(f"rel error = {float(rel_err):.3e}")

    assert rel_err < 1e-6, "GMRES did not converge sufficiently."


if __name__ == "__main__":
    test_gmres_dense()
    print("GMRES test passed.")


__all__ = [
    "GMRESState",
    "GMRESWorkspace",
    "initialize_fgmres_state",
    "gmres_solve",
]
