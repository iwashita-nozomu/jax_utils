"""Import-safe XLA/JAX environment builders.

This module must remain stdlib-only so callers can prepare environment
variables before importing `jax`, `jax.numpy`, or `jax_util.base`.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "merge_env_vars",
    "build_cpu_env",
    "build_gpu_env",
    "build_env_for_profile",
]


def _format_float_env(value: float, /) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".")


def _bool_flag(enabled: bool, /) -> str:
    return "1" if enabled else "0"


def merge_env_vars(*env_sets: Mapping[str, str] | None) -> dict[str, str]:
    """Merge environment-variable mappings from left to right."""
    merged: dict[str, str] = {}
    for env_set in env_sets:
        if env_set is None:
            continue
        for key, value in env_set.items():
            merged[str(key)] = str(value)
    return merged


def build_cpu_env(*, enable_hlo_dump: bool = False) -> dict[str, str]:
    """Build a CPU-only JAX/XLA environment.

    Apply the returned mapping before importing `jax` or `jax_util`.
    """
    env_vars = {
        "JAX_PLATFORMS": "cpu",
        "CUDA_VISIBLE_DEVICES": "",
        "NVIDIA_VISIBLE_DEVICES": "",
    }
    if enable_hlo_dump:
        env_vars["JAX_UTIL_ENABLE_HLO_DUMP"] = _bool_flag(True)
    return env_vars


def build_gpu_env(
    *,
    visible_devices: str | None = None,
    jax_platform_name: str | None = None,
    disable_preallocation: bool = True,
    memory_fraction: float | None = None,
    allocator: str | None = None,
    tf_gpu_allocator: str | None = None,
    use_cuda_host_allocator: bool | None = None,
    enable_hlo_dump: bool = False,
) -> dict[str, str]:
    """Build a GPU-oriented JAX/XLA environment.

    Apply the returned mapping before importing `jax` or `jax_util`.
    """
    env_vars: dict[str, str] = {}
    if visible_devices is not None:
        env_vars["CUDA_VISIBLE_DEVICES"] = visible_devices
        env_vars["NVIDIA_VISIBLE_DEVICES"] = visible_devices
    if jax_platform_name is not None:
        env_vars["JAX_PLATFORMS"] = jax_platform_name
    if disable_preallocation:
        env_vars["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if memory_fraction is not None:
        env_vars["XLA_PYTHON_CLIENT_MEM_FRACTION"] = _format_float_env(memory_fraction)
    if allocator is not None:
        env_vars["XLA_PYTHON_CLIENT_ALLOCATOR"] = allocator
    if tf_gpu_allocator is not None:
        env_vars["TF_GPU_ALLOCATOR"] = tf_gpu_allocator
    if use_cuda_host_allocator is not None:
        env_vars["XLA_PYTHON_CLIENT_USE_CUDA_HOST_ALLOCATOR"] = (
            "true" if use_cuda_host_allocator else "false"
        )
    if enable_hlo_dump:
        env_vars["JAX_UTIL_ENABLE_HLO_DUMP"] = _bool_flag(True)
    return env_vars


def build_env_for_profile(profile: str, **kwargs: object) -> dict[str, str]:
    """Build env vars for a named standard profile."""
    if profile == "cpu":
        return build_cpu_env(
            enable_hlo_dump=bool(kwargs.get("enable_hlo_dump", False)),
        )
    if profile == "gpu":
        return build_gpu_env(
            visible_devices=(
                str(kwargs["visible_devices"])
                if kwargs.get("visible_devices") is not None
                else None
            ),
            jax_platform_name=(
                str(kwargs["jax_platform_name"])
                if kwargs.get("jax_platform_name") is not None
                else None
            ),
            disable_preallocation=bool(kwargs.get("disable_preallocation", True)),
            memory_fraction=(
                float(kwargs["memory_fraction"])
                if kwargs.get("memory_fraction") is not None
                else None
            ),
            allocator=(
                str(kwargs["allocator"])
                if kwargs.get("allocator") is not None
                else None
            ),
            tf_gpu_allocator=(
                str(kwargs["tf_gpu_allocator"])
                if kwargs.get("tf_gpu_allocator") is not None
                else None
            ),
            use_cuda_host_allocator=(
                bool(kwargs["use_cuda_host_allocator"])
                if kwargs.get("use_cuda_host_allocator") is not None
                else None
            ),
            enable_hlo_dump=bool(kwargs.get("enable_hlo_dump", False)),
        )
    raise ValueError(f"Unknown XLA env profile: {profile}")
