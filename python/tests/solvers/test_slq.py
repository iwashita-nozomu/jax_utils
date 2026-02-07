from __future__ import annotations

import json

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from jax_util.solvers.slq import slq_spectral_cdf
from jax_util.solvers.slq import slq_spectral_density
from jax_util.solvers.slq import slq_trace_f
from jax_util.base import DEFAULT_DTYPE
from jax_util.base import LinOp
from jax_util.base import Vector
from jax_util.base import DEFAULT_DTYPE
from jax_util.base import LinOp
from jax_util.base import Vector


def _make_symmetric_matrix(dim: int, *, seed: int, dtype: DTypeLike) -> jnp.ndarray:
    """テスト用の明示な対称行列を生成します。

    Notes
    -----
    - SLQ の理論は「対称（Hermitian）行列」を前提にするため、ここでは対称化します。
    - SPD（正定値）である必要はありません。
    - 実装は単純さを優先し、`A := (B + B^T) / 2` を用います。
    """
    key = jax.random.PRNGKey(seed)
    b = jax.random.normal(key, (dim, dim), dtype=dtype)
    a = (b + b.T) * jnp.asarray(0.5, dtype=dtype)
    return a


def test_slq_trace_f_matches_direct_eig_for_trace_square() -> None:
    dim = 256
    dtype = DEFAULT_DTYPE

    a = _make_symmetric_matrix(dim, seed=0, dtype=dtype)
    mv = LinOp(lambda v, /: a @ v)

    evals = jnp.linalg.eigvalsh(a)
    target = jnp.sum(evals**2)

    def f_on_nodes(nodes: Vector) -> Vector:
        return nodes**2

    est = slq_trace_f(
        mv,
        dim,
        f_on_nodes,
        m=80,
        s=16,
        probe="rademacher",
        seed=1,
        dtype=dtype,
    )

    rel_err = jnp.abs(est - target) / jnp.maximum(jnp.abs(target), jnp.asarray(1e-12, dtype=dtype))

    print(
        json.dumps(
            {
                "case": "slq_trace_f",
                "dim": dim,
                "f": "square",
                "m": 80,
                "s": 16,
                "probe": "rademacher",
                "seed": 1,
                "expected": float(target),
                "actual": float(est),
                "rel_err": float(rel_err),
            },
            ensure_ascii=False,
        )
    )
    assert float(rel_err) < 0.10


def test_slq_cdf_matches_direct_ecdf() -> None:
    """CDF を直接の ECDF と比較します。"""
    dim = 256
    dtype = DEFAULT_DTYPE

    a = _make_symmetric_matrix(dim, seed=2, dtype=dtype)
    mv = LinOp(lambda v, /: a @ v)

    evals = jnp.linalg.eigvalsh(a)
    g = 101
    grid = jnp.linspace(evals[0], evals[-1], g, dtype=dtype)

    # 直接の ECDF
    ecdf = jnp.mean((evals[None, :] <= grid[:, None]).astype(dtype), axis=1)

    est = slq_spectral_cdf(
        mv,
        dim,
        grid,
        m=90,
        s=24,
        probe="rademacher",
        seed=3,
        dtype=dtype,
    )

    max_err = jnp.max(jnp.abs(est - ecdf))

    print(
        json.dumps(
            {
                "case": "slq_spectral_cdf",
                "dim": dim,
                "g": g,
                "m": 90,
                "s": 24,
                "probe": "rademacher",
                "seed": 3,
                "max_abs_err": float(max_err),
            },
            ensure_ascii=False,
        )
    )
    assert float(max_err) < 0.10


def test_slq_density_integrates_to_one() -> None:
    """密度推定が概ね1に正規化されることを確認します。"""
    dim = 256
    dtype = DEFAULT_DTYPE

    a = _make_symmetric_matrix(dim, seed=4, dtype=dtype)
    mv = LinOp(lambda v, /: a @ v)

    evals = jnp.linalg.eigvalsh(a)
    g = 151
    grid = jnp.linspace(evals[0], evals[-1], g, dtype=dtype)

    # カーネル密度は sigma に敏感なので、外部から指定できることを確認する目的でここでは明示指定します。
    sigma = (grid[-1] - grid[0]) / jnp.asarray(200.0, dtype=dtype)

    rho = slq_spectral_density(
        mv,
        dim,
        grid,
        m=90,
        s=24,
        sigma=sigma,
        probe="rademacher",
        seed=5,
        dtype=dtype,
    )

    # 台形則で積分（gridは等間隔）
    dx = grid[1] - grid[0]
    approx_int = jnp.sum((rho[:-1] + rho[1:]) * dx) * jnp.asarray(0.5, dtype=dtype)

    abs_err = jnp.abs(approx_int - jnp.asarray(1.0, dtype=dtype))
    print(
        json.dumps(
            {
                "case": "slq_spectral_density",
                "dim": dim,
                "g": g,
                "m": 90,
                "s": 24,
                "probe": "rademacher",
                "seed": 5,
                "sigma": float(sigma),
                "approx_integral": float(approx_int),
                "abs_err": float(abs_err),
            },
            ensure_ascii=False,
        )
    )
    assert float(abs_err) < 0.10


def _run_all_for_script() -> None:
    """`python test_slq.py` での補助実行を提供します。

    Notes
    -----
    - pytest はこのファイル中の `test_` 関数を収集して実行します。
    - 一方で `python path/to/test_slq.py` は pytest を起動しないため、何も実行されません。
    - 本関数は「計算結果の標準出力」を手軽に確認したい場合の補助です。
    """
    test_slq_trace_f_matches_direct_eig_for_trace_square()
    test_slq_cdf_matches_direct_ecdf()
    test_slq_density_integrates_to_one()


if __name__ == "__main__":
    _run_all_for_script()
