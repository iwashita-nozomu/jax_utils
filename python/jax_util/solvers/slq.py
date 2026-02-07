from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from jax import lax
from jax.typing import DTypeLike

from ..base import DEFAULT_DTYPE
from ..base import LinearOperator
from ..base import Matrix
from ..base import Scalar
from ..base import Vector
# -----------------------------
# 1) Lanczos: build tridiagonal T (m x m) from matvec only
# -----------------------------

def _lanczos_tridiag_from_v0(
    mv: LinearOperator,
    v0: Matrix,
    m: int,
    *,
    eps: Scalar,
) -> Matrix:
    """
    m-step Lanczos により対称三重対角行列 T を構成します。

    Notes
    -----
    - 3項漸化式（再直交化なし）です。
    - `eps` はゼロ割り回避のための小さい正の定数です。
    """
    v0 = v0 / (jnp.linalg.norm(v0) + eps)

    def step(carry: tuple[Vector, Vector, Scalar], _):
        v_prev, v_curr, beta_prev = carry

        w = mv @ v_curr - beta_prev * v_prev
        alpha: Scalar = jnp.dot(v_curr, w)
        w = w - alpha * v_curr
        beta: Scalar = jnp.linalg.norm(w)

        v_next = w / (beta + eps)
        return (v_curr, v_next, beta), (alpha, beta)

    carry0 = (
        jnp.zeros_like(v0),
        v0,
        jnp.asarray(0.0, dtype=v0.dtype),
    )
    _, (alphas, betas) = lax.scan(step, carry0, xs=None, length=m)

    off = betas[:-1]  # length m-1
    T = jnp.diag(alphas) + jnp.diag(off, 1) + jnp.diag(off, -1)
    return T


# -----------------------------
# 2) One-probe SLQ quadrature nodes+weights from T
# -----------------------------

def _nodes_weights_from_T(T: Matrix) -> tuple[Vector, Vector]:
    """Lanczos が誘導する Gauss 求積のノード・重みを返します。"""
    evals, evecs = jnp.linalg.eigh(T)
    w = evecs[0, :] ** 2
    return evals, w


def _make_probe_vectors(
    *,
    dim: int,
    s: int,
    probe: Literal["rademacher", "normal"],
    seed: int,
    dtype: DTypeLike,
    eps: Scalar,
) -> Matrix:
    """SLQ のプローブベクトル（shape==(dim, s)）を生成します。"""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, s)

    if probe == "rademacher":
        def one(k: jax.Array) -> Vector:
            z = jax.random.rademacher(k, (dim,), dtype=dtype)
            return z / jnp.sqrt(jnp.asarray(dim, dtype=dtype))

        v = jax.vmap(one)(keys)  # (s, dim)
        return jnp.asarray(v.T)

    def one(k: jax.Array) -> Vector:
        x = jax.random.normal(k, (dim,), dtype=dtype)
        return x / (jnp.linalg.norm(x) + eps)

    v = jax.vmap(one)(keys)  # (s, dim)
    return jnp.asarray(v.T)


# -----------------------------
# 3) SLQ estimator of trace(f(A))
# -----------------------------
def slq_trace_f(
    mv: LinearOperator,
    dim: int,
    f_on_nodes: Callable[[Vector], Vector],
    *,
    m: int,
    s: int,
    probe: Literal["rademacher", "normal"] = "rademacher",
    seed: int = 0,
    dtype: DTypeLike = DEFAULT_DTYPE,
    eps: Scalar | None = None,
) -> Vector:
    """
    SLQ により `tr(f(A))` を推定します。

    Notes
    -----
    - `f_on_nodes(nodes)` は `nodes.shape == (m,)` を受け取り、同形状を返す想定です。
    - 返り値はスカラー相当ですが、本プロジェクトの `Scalar`/`Vector` 型に合わせて返します。
    """
    if eps is None:
        eps = jnp.asarray(1e-12, dtype=dtype)

    probes = _make_probe_vectors(dim=dim, s=s, probe=probe, seed=seed, dtype=dtype, eps=eps)

    def one_probe(v0: Vector) -> Scalar:
        T = _lanczos_tridiag_from_v0(mv, v0, m, eps=eps)
        nodes, w = _nodes_weights_from_T(T)
        return jnp.dot(w, f_on_nodes(nodes))

    vals = jax.vmap(one_probe)(probes.T)
    return jnp.asarray(dim, dtype=vals.dtype) * jnp.mean(vals)


# -----------------------------
# 4) Spectral density (PDF-like) and CDF via SLQ
# -----------------------------
def slq_spectral_density(
    mv: LinearOperator,
    dim: int,
    grid: Vector,
    *,
    m: int,
    s: int,
    sigma: Scalar,
    probe: Literal["rademacher", "normal"] = "rademacher",
    seed: int = 0,
    dtype: DTypeLike = DEFAULT_DTYPE,
    eps: Scalar | None = None,
) -> Vector:
    """
    SLQ によりスペクトル密度（PDF 風）を推定します。

    Notes
    -----
    - `grid` は評価点（shape==(G,)）です。
    - ガウス核により平滑化した密度を返します。
    """
    if eps is None:
        eps = jnp.asarray(1e-12, dtype=dtype)

    grid = jnp.asarray(grid, dtype=dtype)
    sigma = jnp.asarray(sigma, dtype=dtype)

    probes = _make_probe_vectors(dim=dim, s=s, probe=probe, seed=seed, dtype=dtype, eps=eps)

    def one_probe(v0: Vector) -> Vector:
        T = _lanczos_tridiag_from_v0(mv, v0, m, eps=eps)
        nodes, w = _nodes_weights_from_T(T)  # (m,), (m,)

        # Evaluate Gaussian kernels at all grid points: f_lambda(nodes) for each lambda
        # f_lambda(x) = N(lambda | x, sigma^2) == N(x | lambda, sigma^2)
        diff = (grid[:, None] - nodes[None, :]) / sigma
        coef = jnp.asarray(1.0, dtype=dtype) / (jnp.sqrt(jnp.asarray(2.0, dtype=dtype) * jnp.pi) * sigma)
        gauss = jnp.exp(jnp.asarray(-0.5, dtype=dtype) * diff**2) * coef  # (G,m)
        return gauss @ w  # (G,)

    rhos = jax.vmap(one_probe)(probes.T)  # (s, G)
    rho = jnp.mean(rhos, axis=0)      # (G,)
    # This rho is already normalized to integrate to ~1 (subject to grid range and smoothing).
    return rho


def slq_spectral_cdf(
    mv: LinearOperator,
    dim: int,
    grid: Vector,
    *,
    m: int,
    s: int,
    probe: Literal["rademacher", "normal"] = "rademacher",
    seed: int = 0,
    dtype: DTypeLike = DEFAULT_DTYPE,
    eps: Scalar | None = None,
) -> Vector:
    """
    SLQ によりスペクトル CDF（CESM 風）を推定します。
    """
    if eps is None:
        eps = jnp.asarray(1e-12, dtype=dtype)

    grid = jnp.asarray(grid, dtype=dtype)
    probes = _make_probe_vectors(dim=dim, s=s, probe=probe, seed=seed, dtype=dtype, eps=eps)

    def one_probe(v0: Vector) -> Vector:
        T = _lanczos_tridiag_from_v0(mv, v0, m, eps=eps)
        nodes, w = _nodes_weights_from_T(T)  # (m,), (m,)

        # For each grid λ: sum_j w_j * 1[nodes_j <= λ]
        return (nodes[None, :] <= grid[:, None]).astype(w.dtype) @ w

    phis = jax.vmap(one_probe)(probes.T)  # (s, G)
    return jnp.mean(phis, axis=0)

__all__ = [
    "slq_trace_f",
    "slq_spectral_density",
    "slq_spectral_cdf",
]