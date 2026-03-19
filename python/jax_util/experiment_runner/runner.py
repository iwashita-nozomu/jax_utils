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
        # 待機中のケースから FIFO 順で次のケースを取り出します。
        # context_builder が指定されている場合は context を生成し、
        # 指定がなければ空の context を返します。
        # 待機中のケースがなくなったときだけ None を返します。
        if self._pending_cases:
            case = self._pending_cases.pop(0)
            return case, self._build_context(case)
        return None

    def on_finish(self, case: T, context: TaskContext, exit_code: int) -> None:
        # ケース完了時に結果を completions リストへ記録します。
        # context の不変性を確保するため、dict() で複製してから保存します。
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
    def __init__(self, scheduler: Scheduler[T]) -> None:
        self.scheduler = scheduler

    def run(self, worker: Worker[T, U]) -> None:
        # ProcessPoolExecutor で max_workers 本のプロセスを起動し、ケースを並列実行します。
        # scheduler から次のケースを取得し、完了時に on_finish を呼び出して状態を更新します。
        # FIRST_COMPLETED で done future を順次処理し、新しいケースを投入します。
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
