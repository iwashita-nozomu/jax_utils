from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.smolyak_experiment.compare_smolyak_vs_mc import _build_parser
from experiments.smolyak_experiment.cases import SUPPORTED_FAMILIES, make_family_bundle
from python.jax_util.functional.smolyak import SmolyakIntegrator


@pytest.mark.parametrize("family", SUPPORTED_FAMILIES)
def test_make_family_bundle_supports_all_declared_families(family: str) -> None:
    bundle = make_family_bundle(
        jnp=jnp,
        family=family,
        dimension=3,
        dtype=jnp.float64,
        gaussian_alpha=0.8,
        anisotropic_alpha_start=0.2,
        anisotropic_alpha_stop=1.4,
        shift_start=-0.25,
        shift_stop=0.25,
        laplace_beta_start=1.0,
        laplace_beta_stop=6.0,
        coeff_start=-1.5,
        coeff_stop=1.5,
    )
    values = bundle.integrand(jnp.zeros((3,), dtype=jnp.float64))
    assert values.shape == (1,)
    assert np.all(np.isfinite(np.asarray(values)))
    assert np.isfinite(bundle.analytic_value)
    assert bundle.metadata["family"] == family


def test_compare_parser_accepts_all_declared_families() -> None:
    parser = _build_parser()
    for family in SUPPORTED_FAMILIES:
        namespace = parser.parse_args(["--family", family])
        assert namespace.family == family


def test_balanced_exponential_eval_one_is_jittable() -> None:
    bundle = make_family_bundle(
        jnp=jnp,
        family="balanced_exponential",
        dimension=2,
        dtype=jnp.float64,
        gaussian_alpha=0.8,
        anisotropic_alpha_start=0.2,
        anisotropic_alpha_stop=1.4,
        shift_start=-0.25,
        shift_stop=0.25,
        laplace_beta_start=1.0,
        laplace_beta_stop=6.0,
        coeff_start=-1.5,
        coeff_stop=1.5,
    )
    integrator = SmolyakIntegrator(dimension=2, level=2, dtype=jnp.float64)
    compiled = eqx.filter_jit(lambda scale: bundle.eval_one(scale, integrator))
    value = compiled(jnp.asarray(1.0, dtype=jnp.float64))
    assert np.isfinite(float(np.asarray(value)))
