"""ランナーと標準的なワーカー実装。

軽量な抽象（`StandardWorker` / `StandardCompletion` / `StandardRunner`）を提供します。
実装は fresh child process と structured completion の単純な契約に従います。

JAX fork() 互換性: StandardRunner は spawn コンテキストで child process を起動し、
ケースごとに fresh process で worker を実行します。これにより、`CUDA_VISIBLE_DEVICES`
や `JAX_PLATFORMS` のような import-sensitive な環境変数がケース間で汚染されるのを
避けます。
"""

from __future__ import annotations

from multiprocessing.connection import Connection, wait as wait_for_connections
import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, TypeVar, cast

from .child_runtime import execute_worker_in_child
from .context_utils import apply_environment_variables
from .execution_result import (
    ExecutionResult,
    FailureKind,
    build_failure_result,
    build_parent_exit_result,
    build_skipped_result,
    build_success_result,
)
from .jax_context import check_picklable, get_spawn_context
from .monitor import RuntimeMonitor
from .process_supervisor import terminate_then_kill_process
from .protocols import (
    ContextInitializer,
    ResourceEstimate,
    Scheduler,
    TaskContext,
    Worker,
)
from .result_io import json_compatible


T = TypeVar("T")
U = TypeVar("U")

# プログレス報告コールバック型定義
# ProgressCallback は実行完了ごとに呼び出される
# 引数: completed_count (完了数), total_count (全体), elapsed_time (経過時間秒), running_count (実行中数)
ProgressCallback = Callable[[int, int, float, int], None] | None

__all__ = [
    "StandardWorker",
    "StandardCompletion",
    "StandardRunner",
    "ProgressCallback",
]


def _mapping_case_id(case: object, /) -> object | None:
    if not isinstance(case, Mapping):
        return None
    if "case_id" not in case:
        return None
    return json_compatible(case["case_id"])


def _context_runner_metadata(context: TaskContext, /) -> dict[str, object]:
    metadata = context.get("runner_metadata", {})
    if not isinstance(metadata, dict):
        return {}
    return {str(key): value for key, value in metadata.items()}


def _worker_label_from_context(
    context: TaskContext,
    /,
    *,
    pid: int | None = None,
) -> str:
    metadata = _context_runner_metadata(context)
    worker_label = metadata.get("worker_label")
    if isinstance(worker_label, str) and worker_label:
        return worker_label

    env_vars = context.get("environment_variables", {})
    if isinstance(env_vars, dict):
        env_worker_label = env_vars.get("EXPERIMENT_RUNNER_WORKER_LABEL")
        if isinstance(env_worker_label, str) and env_worker_label:
            return env_worker_label

    if pid is not None:
        return f"worker-{pid}"
    return "worker"


def _gpu_ids_from_context(context: TaskContext, /) -> tuple[int, ...]:
    metadata = _context_runner_metadata(context)
    gpu_ids = metadata.get("gpu_ids")
    if isinstance(gpu_ids, (list, tuple)):
        return tuple(int(gpu_id) for gpu_id in gpu_ids)

    env_vars = context.get("environment_variables", {})
    if not isinstance(env_vars, dict):
        return ()
    gpu_ids_text = env_vars.get("gpu_ids")
    if isinstance(gpu_ids_text, str) and gpu_ids_text:
        return tuple(
            int(gpu_id.strip())
            for gpu_id in gpu_ids_text.split(",")
            if gpu_id.strip()
        )
    gpu_id_text = env_vars.get("gpu_id")
    if isinstance(gpu_id_text, str) and gpu_id_text:
        return (int(gpu_id_text),)
    return ()


class StandardWorker(Generic[T, U]):
    """ワーカー呼び出しラッパー。

    - `initializer(context)` を先頭で実行して process-local 環境を整える。
    - `task` が `ExecutionResult` を返せばそのまま返す。
    - `task` が通常の payload を返した場合は `status="ok"` の結果へ正規化する。
    - 例外の structured diagnostics 化は child runtime 側が担当する。
    - 任意で `resource_estimator` を注入して `resource_estimate` を提供できる。
    """

    def __init__(
        self,
        task: Callable[[T, TaskContext], U],
        resource_estimator: Callable[[T], ResourceEstimate] | None = None,
        initializer: ContextInitializer = apply_environment_variables,
    ) -> None:
        self.task = task
        self._resource_estimator = resource_estimator
        self.initializer = initializer

    def __call__(self, case: T, context: TaskContext) -> ExecutionResult:
        self.initializer(context)
        task_result = self.task(case, context)
        if isinstance(task_result, ExecutionResult):
            return task_result
        return build_success_result()

    def resource_estimate(self, case: T) -> ResourceEstimate:
        # NOTE: _resource_estimator は Worker インスタンスが from_worker()
        #       で生成される場合のみ呼び出される。そのため常に None ではない。
        if self._resource_estimator is None:
            raise ValueError("resource_estimator is not configured for this worker.")
        return self._resource_estimator(case)


@dataclass(frozen=True)
class StandardCompletion(Generic[T]):
    case: T
    context: TaskContext
    result: ExecutionResult


@dataclass
class _RunningProcess(Generic[T]):
    case: T
    context: TaskContext
    process: Any
    receiver: Connection
    started_at: float


def _run_worker_in_child(
    sender: Connection,
    worker: Worker[T],
    case: T,
    context: TaskContext,
) -> None:
    """Execute one worker invocation in a fresh spawned child process."""
    result = execute_worker_in_child(worker, case, context)
    try:
        sender.send(result)
    except (BrokenPipeError, OSError):
        pass
    finally:
        sender.close()


class StandardRunner(Generic[T]):
    """スケジューラとワーカーを使ってケースを並列実行するランナー。

    - spawn コンテキストでケースごとに fresh child process を起動
    - GPU / JAX の import-sensitive な環境変数がケース間で漏れない
    - 完了ごとに `scheduler.on_finish` を呼び、次のケースを投入する。
    - プログレス報告コールバックをサポート
    """

    def __init__(
        self,
        scheduler: Scheduler[T],
        progress_callback: ProgressCallback = None,
        case_timeout_seconds: float | None = None,
        termination_grace_seconds: float = 5.0,
        monitor: RuntimeMonitor | None = None,
        on_case_started: Callable[[T, TaskContext, int], None] | None = None,
        on_case_finished: Callable[
            [T, TaskContext, ExecutionResult, int | None],
            None,
        ]
        | None = None,
    ) -> None:
        """
        ランナーを初期化する。

        Parameters
        ----------
        scheduler : Scheduler[T]
            ケースをスケジューリングするスケジューラ
        progress_callback : ProgressCallback, optional
            実行完了ごとに (completed, total, elapsed, running) を受け取るコールバック
        """
        if case_timeout_seconds is not None and case_timeout_seconds <= 0:
            raise ValueError("case_timeout_seconds must be positive when provided.")
        if termination_grace_seconds <= 0:
            raise ValueError("termination_grace_seconds must be positive.")
        self.scheduler = scheduler
        self.progress_callback = progress_callback
        self.case_timeout_seconds = case_timeout_seconds
        self.termination_grace_seconds = termination_grace_seconds
        self.monitor = monitor
        self.on_case_started = on_case_started
        self.on_case_finished = on_case_finished

    def run(self, worker: Worker[T]) -> None:
        """ケースを並列実行する。

        spawn コンテキストで fresh child process を起動し、
        JAX の import-sensitive な process state をケース単位で分離する。

        Parameters
        ----------
        worker : Worker
            各ケースを実行するワーカー

        Raises
        ------
        ValueError
            ワーカーが pickle 化不可能な場合。spawn child process へ
            別プロセスに送出できる必要があります。
        """
        # ワーカーが pickle 化可能であることを確認
        check_picklable(worker, name="Worker")
        self._execute_with_spawned_processes(worker, self.scheduler.total_case_count)

    def _execute_with_spawned_processes(
        self,
        worker: Worker[T],
        total_cases: int,
    ) -> None:
        """
        spawn child process を使用してケースを実行する。

        プログレスコールバックが登録されている場合は、
        ケース完了ごとに進捗状況を報告する。

        Parameters
        ----------
        worker : Worker[T, U]
            各ケースを実行するワーカー
        total_cases : int
            全体のケース数（既完了 + 未実行）
        """
        max_workers = self.scheduler.resource_capacity.max_workers
        spawn_context = get_spawn_context()
        running: dict[Connection, _RunningProcess[T]] = {}
        start_time = time.time()
        self._update_monitor_state(total_cases, len(running))
        try:
            while not self.scheduler.is_completed() or running:
                # max_workers 本まで fresh child process を起動する。
                while len(running) < max_workers:
                    job = self.scheduler.next_case()
                    if job is None:
                        break
                    case, context = job
                    skip_reason = self._resolve_skip_reason(case, context)
                    if skip_reason is not None:
                        result = build_skipped_result(skip_reason, source="runner")
                        self.scheduler.on_finish(
                            case,
                            context,
                            result,
                        )
                        self._notify_case_finished(case, context, result, pid=None)
                        self._update_monitor_state(total_cases, len(running))
                        self._report_progress(start_time, total_cases, len(running))
                        continue
                    receiver, sender = spawn_context.Pipe(duplex=False)
                    process = spawn_context.Process(
                        target=_run_worker_in_child,
                        args=(sender, worker, case, context),
                    )
                    process.start()
                    sender.close()
                    running[receiver] = _RunningProcess(
                        case=case,
                        context=context,
                        process=process,
                        receiver=receiver,
                        started_at=time.monotonic(),
                    )
                    self._notify_case_started(case, context, process.pid)
                    self._update_monitor_state(total_cases, len(running))

                self._finish_timed_out_processes(running, start_time, total_cases)

                if not running:
                    continue

                ready_receivers = cast(
                    list[Connection],
                    wait_for_connections(list(running.keys()), timeout=0.1),
                )
                if not ready_receivers:
                    ready_receivers = [
                        receiver
                        for receiver, child in running.items()
                        if not child.process.is_alive()
                    ]
                    if not ready_receivers:
                        continue

                for receiver in ready_receivers:
                    child = running.pop(receiver)
                    result = self._receive_child_result(child)
                    self.scheduler.on_finish(child.case, child.context, result)
                    self._notify_case_finished(
                        child.case,
                        child.context,
                        result,
                        pid=child.process.pid,
                    )
                    self._update_monitor_state(total_cases, len(running))
                    self._report_progress(start_time, total_cases, len(running))
        finally:
            for child in running.values():
                child.receiver.close()
                terminate_then_kill_process(
                    child.process,
                    grace_seconds=self.termination_grace_seconds,
                )
                self._notify_case_finished(
                    child.case,
                    child.context,
                    build_failure_result(
                        failure_kind=FailureKind.NO_COMPLETION,
                        message="Child process was terminated during runner cleanup.",
                        raw_exit_code=child.process.exitcode,
                        source="parent",
                    ),
                    pid=child.process.pid,
                )
            self._update_monitor_state(total_cases, 0)

    def _resolve_skip_reason(
        self,
        case: T,
        context: TaskContext,
    ) -> str | None:
        skip_controller = self.scheduler.skip_controller
        if skip_controller is None:
            return None
        skip_reason = cast(object, skip_controller.should_skip(case, context))
        if skip_reason is not None and not isinstance(skip_reason, str):
            raise TypeError("skip_controller.should_skip must return str | None.")
        return skip_reason

    def _finish_timed_out_processes(
        self,
        running: dict[Connection, _RunningProcess[T]],
        start_time: float,
        total_cases: int,
    ) -> None:
        if self.case_timeout_seconds is None:
            return

        now = time.monotonic()
        timed_out_receivers = [
            receiver
            for receiver, child in running.items()
            if now - child.started_at > self.case_timeout_seconds
        ]
        for receiver in timed_out_receivers:
            child = running.pop(receiver)
            child.receiver.close()
            terminate_then_kill_process(
                child.process,
                grace_seconds=self.termination_grace_seconds,
            )
            result = build_failure_result(
                failure_kind=FailureKind.TIMEOUT,
                message=(
                    "Child process exceeded "
                    f"case_timeout_seconds={self.case_timeout_seconds}."
                ),
                raw_exit_code=child.process.exitcode,
                source="parent",
            )
            self.scheduler.on_finish(child.case, child.context, result)
            self._notify_case_finished(
                child.case,
                child.context,
                result,
                pid=child.process.pid,
            )
            self._update_monitor_state(total_cases, len(running))
            self._report_progress(start_time, total_cases, len(running))

    def _receive_child_result(
        self,
        child: _RunningProcess[T],
    ) -> ExecutionResult:
        result: ExecutionResult | None = None
        try:
            received = child.receiver.recv()
            if isinstance(received, ExecutionResult):
                result = received
        except (EOFError, OSError, TypeError, ValueError):
            result = None
        finally:
            child.receiver.close()
            child.process.join()

        if result is not None:
            if child.process.exitcode in {0, None}:
                return result

        return build_parent_exit_result(
            child.process.exitcode,
            message=(
                "Child process exited without a usable completion record."
                if result is None
                else "Child process exited abnormally after reporting completion."
            ),
        )

    def _report_progress(
        self,
        start_time: float,
        total_cases: int,
        running_count: int,
    ) -> None:
        if self.progress_callback is None:
            return
        elapsed = time.time() - start_time
        completed = len(self.scheduler.completions)
        self.progress_callback(
            completed,
            total_cases,
            elapsed,
            running_count,
        )

    def _update_monitor_state(self, total_cases: int, running_count: int) -> None:
        if self.monitor is None:
            return
        completed = len(self.scheduler.completions)
        pending = max(total_cases - completed - running_count, 0)
        self.monitor.update_runner_state(
            pending_cases=pending,
            running_cases=running_count,
            completed_cases=completed,
            max_workers=self.scheduler.resource_capacity.max_workers,
        )

    def _notify_case_started(
        self,
        case: T,
        context: TaskContext,
        pid: int | None,
    ) -> None:
        if pid is None:
            return
        if self.monitor is not None:
            self.monitor.register_worker(
                case_id=_mapping_case_id(case),
                worker_label=_worker_label_from_context(context, pid=pid),
                pid=pid,
                gpu_ids=_gpu_ids_from_context(context),
            )
        if self.on_case_started is not None:
            self.on_case_started(case, context, pid)

    def _notify_case_finished(
        self,
        case: T,
        context: TaskContext,
        result: ExecutionResult,
        *,
        pid: int | None,
    ) -> None:
        if self.monitor is not None and pid is not None:
            self.monitor.complete_worker(pid=pid, result=result.to_dict())
        if self.on_case_finished is not None:
            self.on_case_finished(case, context, result, pid)
