from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import traceback
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from .protocols import (
    Scheduler,
    SUCCESS_EXIT_CODE,
    TaskContext,
    WORKER_PROTOCOL_ERROR_EXIT_CODE,
    Worker,
)


T = TypeVar("T")
U = TypeVar("U")


class StandardWorker(Generic[T, U]):
    def __init__(self, task: Callable[[T, TaskContext], U]) -> None:
        self.task = task

    def __call__(self, case: T, context: TaskContext) -> int:
        try:
            self.task(case, context)
            return SUCCESS_EXIT_CODE
        except Exception:
            traceback.print_exc()
            return WORKER_PROTOCOL_ERROR_EXIT_CODE


@dataclass(frozen=True)
class StandardResourceCapacity:
    max_workers: int


class StandardScheduler(Generic[T]):
    def __init__(
        self,
        resource_capacity: StandardResourceCapacity,
        cases: list[T],
    ) -> None:
        self._resource_capacity = resource_capacity
        self.cases = cases

    @property
    def resource_capacity(self) -> StandardResourceCapacity:
        return self._resource_capacity

    def next_case(self) -> tuple[T, TaskContext] | None:
        if self.cases:
            case = self.cases.pop(0)
            return case, {}
        return None

    def on_finish(self, case: T, context: TaskContext, exit_code: int) -> None:
        del case, context, exit_code

    def is_completed(self) -> bool:
        return not self.cases


class StandardRunner(Generic[T, U]):
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
