"""ランナーと標準的なスケジューラ/ワーカーの実装。

軽量な抽象（`StandardScheduler` / `StandardRunner` / `StandardWorker`）を提供します。
実装は並列実行と完了ハンドリングの単純な契約に従います。

JAX fork() 互換性: StandardRunner は spawn コンテキストで child process を起動し、
ケースごとに fresh process で worker を実行します。これにより、`CUDA_VISIBLE_DEVICES`
や `JAX_PLATFORMS` のような import-sensitive な環境変数がケース間で汚染されるのを
避けます。
"""

from __future__ import annotations

from multiprocessing.connection import Connection, wait as wait_for_connections
import time
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar, cast

from .child_runtime import execute_worker_in_child
from .execution_result import (
    ExecutionResult,
    FailureKind,
    build_failure_result,
    build_parent_exit_result,
    coerce_execution_result,
)
from .jax_context import check_picklable, get_spawn_context
from .process_supervisor import terminate_then_kill_process
from .protocols import (
    ResourceEstimate,
    Scheduler,
    SUCCESS_EXIT_CODE,
    TaskContext,
    Worker,
)


T = TypeVar("T")
U = TypeVar("U")

# プログレス報告コールバック型定義
# ProgressCallback は実行完了ごとに呼び出される
# 引数: completed_count (完了数), total_count (全体), elapsed_time (経過時間秒), running_count (実行中数)
ProgressCallback = Callable[[int, int, float, int], None] | None

__all__ = [
    "StandardWorker",
    "StandardResourceCapacity",
    "StandardCompletion",
    "StandardScheduler",
    "StandardRunner",
    "ProgressCallback",
]


class StandardWorker(Generic[T, U]):
    """ワーカー呼び出しラッパー。

    - `task` を実行し、正常終了時は成功コードを返す。
    - 例外の structured diagnostics 化は child runtime 側が担当する。
    - 任意で `resource_estimator` を注入して `resource_estimate` を提供できる。
    """

    def __init__(
        self,
        task: Callable[[T, TaskContext], U],
        resource_estimator: Callable[[T], ResourceEstimate] | None = None,
    ) -> None:
        self.task = task
        self._resource_estimator = resource_estimator

    def __call__(self, case: T, context: TaskContext) -> int:
        self.task(case, context)
        return SUCCESS_EXIT_CODE

    def resource_estimate(self, case: T) -> ResourceEstimate:
        # NOTE: _resource_estimator は Worker インスタンスが from_worker()
        #       で生成される場合のみ呼び出される。そのため常に None ではない。
        if self._resource_estimator is None:
            raise ValueError("resource_estimator is not configured for this worker.")
        return self._resource_estimator(case)


@dataclass(frozen=True)
class StandardResourceCapacity:
    max_workers: int

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive.")


@dataclass(init=False)
class StandardCompletion(Generic[T]):
    case: T
    context: TaskContext
    result: ExecutionResult

    def __init__(
        self,
        case: T,
        context: TaskContext,
        result: ExecutionResult | int = 0,
        *,
        exit_code: int | None = None,
    ) -> None:
        self.case = case
        self.context = context
        self.result = (
            coerce_execution_result(exit_code)
            if exit_code is not None
            else coerce_execution_result(result)
        )

    @property
    def exit_code(self) -> int:
        return self.result.exit_code


@dataclass
class _RunningProcess(Generic[T]):
    case: T
    context: TaskContext
    process: Any
    receiver: Connection
    started_at: float


def _run_worker_in_child(
    sender: Connection,
    worker: Worker[T, U],
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


class StandardScheduler(Generic[T]):
    """FIFO ベースのシンプルなスケジューラ実装。

    .. deprecated::
        `StandardFullResourceScheduler` を使ってください。
        リソース管理（GPU メモリ、ワーカースロット、ホストメモリ）を
        統合的に管理し、より堅牢な並列実行が可能です。

    - `next_case()` で次のケースとその `TaskContext` を返す。
    - `on_finish()` で `completions` に結果を記録する。
    """

    def __init__(
        self,
        resource_capacity: StandardResourceCapacity,
        cases: list[T],
        context_builder: Callable[[T], TaskContext] | None = None,
    ) -> None:
        self._resource_capacity = resource_capacity
        self._pending_cases = list(cases)
        self._context_builder = context_builder
        self.completions: list[StandardCompletion[T]] = []

    @property
    def resource_capacity(self) -> StandardResourceCapacity:
        return self._resource_capacity

    @property
    def total_case_count(self) -> int:
        return len(self.completions) + len(self._pending_cases)

    def _build_context(self, case: T) -> TaskContext:
        if self._context_builder is None:
            return {}
        return dict(self._context_builder(case))

    def next_case(self) -> tuple[T, TaskContext] | None:
        """待機中ケースの先頭を取り出して (case, context) を返す。存在しなければ None を返す。"""
        if self._pending_cases:
            case = self._pending_cases.pop(0)
            return case, self._build_context(case)
        return None

    def on_finish(
        self,
        case: T,
        context: TaskContext,
        result: ExecutionResult | int,
    ) -> None:
        """完了時に `completions` に結果を記録する（context は複製して保存）。"""
        self.completions.append(
            StandardCompletion(
                case=case,
                context=dict(context),
                result=result,
            )
        )

    def is_completed(self) -> bool:
        return not self._pending_cases


class StandardRunner(Generic[T, U]):
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

    def run(self, worker: Worker[T, U]) -> None:
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
        self._execute_with_spawned_processes(worker, self._resolve_total_cases())

    def _resolve_total_cases(self) -> int:
        total_case_count = getattr(self.scheduler, "total_case_count", None)
        if isinstance(total_case_count, int) and total_case_count >= 0:
            return total_case_count

        total_cases = len(self.scheduler.completions)
        if hasattr(self.scheduler, "_pending_cases"):
            return total_cases + len(self.scheduler._pending_cases)  # type: ignore[attr-defined]
        if hasattr(self.scheduler, "_pending_entries"):
            return total_cases + len(self.scheduler._pending_entries)  # type: ignore[attr-defined]
        return total_cases

    def _execute_with_spawned_processes(
        self,
        worker: Worker[T, U],
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
        try:
            while not self.scheduler.is_completed() or running:
                # max_workers 本まで fresh child process を起動する。
                while len(running) < max_workers:
                    job = self.scheduler.next_case()
                    if job is None:
                        break
                    case, context = job
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
                    self._report_progress(start_time, total_cases, len(running))
        finally:
            for child in running.values():
                child.receiver.close()
                terminate_then_kill_process(
                    child.process,
                    grace_seconds=self.termination_grace_seconds,
                )

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
            self._report_progress(start_time, total_cases, len(running))

    def _receive_child_result(
        self,
        child: _RunningProcess[T],
    ) -> ExecutionResult:
        result: ExecutionResult | int | None = None
        try:
            result = cast(ExecutionResult | int, child.receiver.recv())
        except (EOFError, OSError, TypeError, ValueError):
            result = None
        finally:
            child.receiver.close()
            child.process.join()

        if result is not None:
            normalized = coerce_execution_result(result)
            if child.process.exitcode in {0, None}:
                return normalized

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
