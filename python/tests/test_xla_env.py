from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from jax_util.xla_env import (
    build_cpu_env,
    build_env_for_profile,
    build_gpu_env,
    merge_env_vars,
)


def test_merge_env_vars_overrides_from_left_to_right() -> None:
    assert merge_env_vars(
        {"A": "1", "B": "old"},
        None,
        {"B": "new", "C": "3"},
    ) == {"A": "1", "B": "new", "C": "3"}


def test_build_cpu_env_sets_cpu_only_defaults() -> None:
    env_vars = build_cpu_env(enable_hlo_dump=True)

    assert env_vars == {
        "JAX_PLATFORMS": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "",
        "JAX_UTIL_ENABLE_HLO_DUMP": "1",
    }


def test_build_gpu_env_formats_optional_settings() -> None:
    env_vars = build_gpu_env(
        visible_devices="0,2",
        jax_platform_name="cuda",
        disable_preallocation=True,
        memory_fraction=0.4,
        allocator="platform",
        tf_gpu_allocator="cuda_malloc_async",
        use_cuda_host_allocator=False,
        enable_hlo_dump=True,
    )

    assert env_vars == {
        "CUDA_VISIBLE_DEVICES": "0,2",
        "NVIDIA_VISIBLE_DEVICES": "0,2",
        "JAX_PLATFORMS": "cuda",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.4",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",
        "TF_GPU_ALLOCATOR": "cuda_malloc_async",
        "XLA_PYTHON_CLIENT_USE_CUDA_HOST_ALLOCATOR": "false",
        "JAX_UTIL_ENABLE_HLO_DUMP": "1",
    }


def test_build_env_for_profile_dispatches_and_validates() -> None:
    assert build_env_for_profile("cpu")["JAX_PLATFORMS"] == "cpu"
    assert build_env_for_profile(
        "gpu",
        visible_devices="7",
        disable_preallocation=False,
    )["CUDA_VISIBLE_DEVICES"] == "7"

    with pytest.raises(ValueError, match="Unknown XLA env profile"):
        build_env_for_profile("tpu")


def test_xla_env_module_import_does_not_import_jax() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json, sys; "
                f"sys.path.insert(0, {str(PYTHON_ROOT)!r}); "
                "import jax_util.xla_env; "
                "print(json.dumps({'jax': 'jax' in sys.modules, 'jax_numpy': 'jax.numpy' in sys.modules}))"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout.strip())
    assert payload == {"jax": False, "jax_numpy": False}
