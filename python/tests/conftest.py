from __future__ import annotations

import os

from jax_util.xla_env import build_gpu_env

# Prevent JAX from preallocating most GPU memory during pytest runs.
os.environ.update(build_gpu_env(disable_preallocation=True))
