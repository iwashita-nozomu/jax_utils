"""ランナーと標準的なスケジューラ/ワーカーの実装。

軽量な抽象（`StandardScheduler` / `StandardRunner` / `StandardWorker`）を提供します。
実装は並列実行と完了ハンドリングの単純な契約に従います。
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import traceback
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from .protocols import (
    ResourceEstimate,
    Scheduler,
    SUCCESS_EXIT_CODE,
    TaskContext,
    WORKER_PROTOCOL_ERROR_EXIT_CODE,
    Worker,
)


T = TypeVar("T")
U = TypeVar("U")


class StandardWorker(Generic[T, U]):
    """ワーカー呼び出しラッパー。

    - `task` を実行し、例外が発生した場合はトレースを出力してエラーコードを返す。
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
        try:
            self.task(case, context)
            return SUCCESS_EXIT_CODE
        except Exception:
            traceback.print_exc()
            return WORKER_PROTOCOL_ERROR_EXIT_CODE

    def resource_estimate(self, case: T) -> ResourceEstimate:
        if self._resource_estimator is None:
            raise ValueError("resource_estimator is not configured.")
        return self._resource_estimator(case)


@dataclass(frozen=True)
class StandardResourceCapacity:
    max_workers: int

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive.")


@dataclass(frozen=True)
class StandardCompletion(Generic[T]):
    case: T
    context: TaskContext
    exit_code: int


class StandardScheduler(Generic[T]):
    """FIFO ベースのシンプルなスケジューラ実装。

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

    def on_finish(self, case: T, context: TaskContext, exit_code: int) -> None:
        """完了時に `completions` に結果を記録する（context は複製して保存）。"""
        self.completions.append(
            StandardCompletion(
                case=case,
                context=dict(context),
                exit_code=exit_code,
            )
        )

    def is_completed(self) -> bool:
        return not self._pending_cases


class StandardRunner(Generic[T, U]):
    """スケジューラとワーカーを使ってケースを並列実行するランナー。

    - `ProcessPoolExecutor` を使って `max_workers` 並列プロセスで実行する。
    - 完了ごとに `scheduler.on_finish` を呼び、次のケースを投入する。
    """

    def __init__(self, scheduler: Scheduler[T]) -> None:
        self.scheduler = scheduler

    def run(self, worker: Worker[T, U]) -> None:
        with ProcessPoolExecutor(
            max_workers=self.scheduler.resource_capacity.max_workers
        ) as ex:
            running: dict[Future[int], tuple[T, TaskContext]] = {}
            while not self.scheduler.is_completed() or running:
                while True:
                    job = self.scheduler.next_case()
                    if job is None:
                        break
                    case, context = job
                    fut = ex.submit(worker, case, context)
                    running[fut] = job

                if not running:
                    continue

                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in done:
                    case, context = running.pop(fut)
                    self.scheduler.on_finish(case, context, fut.result())
