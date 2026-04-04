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


def _coerce_optional_float(value: object | None, /) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TypeError(f"Expected float-compatible value, got {type(value).__name__}")


def _append_xla_flags(existing: str | None, flags: list[str], /) -> str | None:
    cleaned_flags = [flag.strip() for flag in flags if flag and flag.strip()]
    if not cleaned_flags:
        return existing
    prefix = (existing or "").strip()
    if prefix:
        return f"{prefix} {' '.join(cleaned_flags)}"
    return " ".join(cleaned_flags)


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
    xla_memory_scheduler: str | None = None,
    xla_gpu_enable_while_loop_double_buffering: bool | None = None,
    xla_latency_hiding_scheduler_rerun: int | None = None,
    jax_compiler_enable_remat_pass: bool | None = None,
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
    xla_flags: list[str] = []
    if xla_memory_scheduler is not None:
        xla_flags.append(f"--xla_memory_scheduler={xla_memory_scheduler}")
    if xla_gpu_enable_while_loop_double_buffering is not None:
        xla_flags.append(
            "--xla_gpu_enable_while_loop_double_buffering="
            f"{str(xla_gpu_enable_while_loop_double_buffering).lower()}"
        )
    if xla_latency_hiding_scheduler_rerun is not None:
        xla_flags.append(
            f"--xla_latency_hiding_scheduler_rerun={int(xla_latency_hiding_scheduler_rerun)}"
        )
    combined_xla_flags = _append_xla_flags(env_vars.get("XLA_FLAGS"), xla_flags)
    if combined_xla_flags is not None:
        env_vars["XLA_FLAGS"] = combined_xla_flags
    if jax_compiler_enable_remat_pass is not None:
        env_vars["JAX_COMPILER_ENABLE_REMAT_PASS"] = (
            "true" if jax_compiler_enable_remat_pass else "false"
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
            memory_fraction=_coerce_optional_float(kwargs.get("memory_fraction")),
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
            xla_memory_scheduler=(
                str(kwargs["xla_memory_scheduler"])
                if kwargs.get("xla_memory_scheduler") is not None
                else None
            ),
            xla_gpu_enable_while_loop_double_buffering=(
                bool(kwargs["xla_gpu_enable_while_loop_double_buffering"])
                if kwargs.get("xla_gpu_enable_while_loop_double_buffering") is not None
                else None
            ),
            xla_latency_hiding_scheduler_rerun=(
                int(kwargs["xla_latency_hiding_scheduler_rerun"])
                if kwargs.get("xla_latency_hiding_scheduler_rerun") is not None
                else None
            ),
            jax_compiler_enable_remat_pass=(
                bool(kwargs["jax_compiler_enable_remat_pass"])
                if kwargs.get("jax_compiler_enable_remat_pass") is not None
                else None
            ),
            enable_hlo_dump=bool(kwargs.get("enable_hlo_dump", False)),
        )
    raise ValueError(f"Unknown XLA env profile: {profile}")
