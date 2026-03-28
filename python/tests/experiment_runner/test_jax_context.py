from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

import pytest

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from jax_util.experiment_runner.context_utils import apply_environment_variables
from jax_util.experiment_runner.jax_context import (
    check_picklable,
    disable_jax_memory_preallocation,
    get_spawn_context,
)


@dataclass(frozen=True)
class _PicklableObject:
    value: int


def test_apply_environment_variables_sets_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)

    apply_environment_variables(
        {
            "environment_variables": {
                "CUDA_VISIBLE_DEVICES": "2",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            }
        }
    )

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"


def test_check_picklable_accepts_dataclass_and_rejects_lambda() -> None:
    check_picklable(_PicklableObject(3), name="picklable")

    with pytest.raises(ValueError, match="not picklable"):
        check_picklable(lambda value: value, name="lambda")


def test_disable_jax_memory_preallocation_sets_expected_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in (
        "XLA_FLAGS",
        "JAX_PLATFORMS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "TF_FORCE_GPU_ALLOW_GROWTH",
    ):
        monkeypatch.delenv(key, raising=False)

    disable_jax_memory_preallocation(gpu_devices=False)
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["XLA_FLAGS"] == "--xla_force_host_platform_device_count=1"

    for key in (
        "JAX_PLATFORMS",
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "TF_FORCE_GPU_ALLOW_GROWTH",
    ):
        monkeypatch.delenv(key, raising=False)

    disable_jax_memory_preallocation(gpu_devices=True)
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] == "0.9"
    assert os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] == "true"


def test_get_spawn_context_returns_spawn() -> None:
    assert get_spawn_context().get_start_method() == "spawn"
