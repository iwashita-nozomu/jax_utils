"""experiment_runner パッケージ。

利用者にはサブモジュールからの明示的インポートを推奨します。例:

    from experiment_runner.runner import StandardRunner
    from experiment_runner.resource_scheduler import FullResourceCapacity

このファイルではサブモジュール自体のみをインポートし、個別シンボルをトップレベルに露出しません。
"""

from . import jax_context as jax_context
from . import runner as runner
from . import resource_scheduler as resource_scheduler
from . import protocols as protocols
from . import monitor as monitor
from . import execution_result as execution_result
from . import process_supervisor as process_supervisor
from . import child_runtime as child_runtime
from . import result_io as result_io

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
    GPUEnvironmentConfig,
    detect_gpu_devices,
    detect_host_memory_bytes,
    detect_max_workers,
)

from .result_io import (
    append_jsonl_record,
    json_compatible,
    read_jsonl_records,
)
from .monitor import (
    RuntimeMonitor,
    RuntimeMonitorConfig,
)
from .execution_result import (
    ExecutionResult,
    FailureKind,
)

# Protocols / constants / JAX utilities
from .protocols import (
    DispatchDecision,
    SkipController,
    TaskContext,
    SUCCESS_EXIT_CODE,
    WORKER_PROTOCOL_ERROR_EXIT_CODE,
)
from .jax_context import (
    check_picklable,
    create_jax_safe_process_pool,
    disable_jax_memory_preallocation,
    get_spawn_context,
)
from .context_utils import (
    apply_environment_variables,
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
    "GPUEnvironmentConfig",
    "detect_gpu_devices",
    "detect_host_memory_bytes",
    "detect_max_workers",
    # result io
    "append_jsonl_record",
    "json_compatible",
    "read_jsonl_records",
    # monitor
    "RuntimeMonitor",
    "RuntimeMonitorConfig",
    "ExecutionResult",
    "FailureKind",
    # protocols / constants / jax utilities
    "DispatchDecision",
    "SkipController",
    "TaskContext",
    "SUCCESS_EXIT_CODE",
    "WORKER_PROTOCOL_ERROR_EXIT_CODE",
    "get_spawn_context",
    "disable_jax_memory_preallocation",
    "check_picklable",
    "create_jax_safe_process_pool",
    "apply_environment_variables",
]
