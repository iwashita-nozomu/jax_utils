from __future__ import annotations

import os
from pathlib import Path
import sys


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = WORKTREE_ROOT / "python"

if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from jax_util.xla_env import build_cpu_env, build_gpu_env


def _default_test_environment() -> dict[str, str]:
    """Return the default JAX/XLA env for repo-wide pytest runs."""
    if os.environ.get("JAX_UTIL_TEST_XLA_PROFILE") == "gpu":
        return build_gpu_env(disable_preallocation=True)
    return build_cpu_env()


for _key, _value in _default_test_environment().items():
    os.environ.setdefault(_key, _value)
