"""experiment_runner パッケージ。

利用者にはサブモジュールからの明示的インポートを推奨します。例:

    from jax_util.experiment_runner.runner import StandardRunner
    from jax_util.experiment_runner.resource_scheduler import FullResourceCapacity

このファイルではサブモジュール自体のみをインポートし、個別シンボルをトップレベルに露出しません。
"""

from . import jax_context as jax_context
from . import runner as runner
from . import gpu_runner as gpu_runner
from . import resource_scheduler as resource_scheduler
from . import protocols as protocols

# 便宜的に最もよく使われるシンボルのみ top-level に再エクスポートします。
# - 詳細な API は引き続きサブモジュールから明示的にインポートしてください。
# - ここに置くものは変更を慎重にするため最小限に留めます。

# Runner / Scheduler / Worker (standard)
from .runner import (
    StandardRunner,
    StandardScheduler,
    StandardWorker,
    StandardResourceCapacity,
    StandardCompletion,
)

# Full resource scheduler (convenience)
from .resource_scheduler import (
    FullResourceCapacity,
    FullResourceEstimate,
    StandardFullResourceScheduler,
    GPUDeviceCapacity,
    detect_gpu_devices,
    detect_host_memory_bytes,
    detect_max_workers,
)

# GPU-specific helpers
from .gpu_runner import (
    GPUResourceCapacity,
    StandardGPUScheduler,
    visible_gpu_ids_from_environment,
)

# Protocols / constants / JAX utilities
from .protocols import (
    TaskContext,
    SUCCESS_EXIT_CODE,
    WORKER_PROTOCOL_ERROR_EXIT_CODE,
)
from .jax_context import (
    get_spawn_context,
    disable_jax_memory_preallocation,
    create_jax_safe_process_pool,
)

__all__ = [
    # runner
    "StandardRunner",
    "StandardScheduler",
    "StandardWorker",
    "StandardResourceCapacity",
    "StandardCompletion",
    # full resource
    "FullResourceCapacity",
    "FullResourceEstimate",
    "StandardFullResourceScheduler",
    "GPUDeviceCapacity",
    "detect_gpu_devices",
    "detect_host_memory_bytes",
    "detect_max_workers",
    # gpu
    "GPUResourceCapacity",
    "StandardGPUScheduler",
    "visible_gpu_ids_from_environment",
    # protocols / constants / jax utilities
    "TaskContext",
    "SUCCESS_EXIT_CODE",
    "WORKER_PROTOCOL_ERROR_EXIT_CODE",
    "get_spawn_context",
    "disable_jax_memory_preallocation",
    "create_jax_safe_process_pool",
]
