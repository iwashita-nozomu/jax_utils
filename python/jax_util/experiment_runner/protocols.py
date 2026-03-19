from __future__ import annotations

from typing import Callable, Protocol, TypeAlias, TypeVar

T = TypeVar("T")
U = TypeVar("U")

TaskContext: TypeAlias = dict[str, str]


class ResourceEstimate(Protocol):
    # Concrete schedulers may use per-case estimates internally.
    ...


class ResourceCapacity(Protocol):
    @property
    def max_workers(self) -> int: ...


class Worker(Protocol[T, U]):
    task: Callable[[T, TaskContext], U]

    # Task を実行して終了コードを返します。ログ出力は worker 側で扱います。
    def __call__(self, case: T, context: TaskContext) -> int: ...


class Scheduler(Protocol[T]):
    @property
    def resource_capacity(self) -> ResourceCapacity: ...

    def next_case(self) -> tuple[T, TaskContext] | None: ...
    def on_finish(self, case: T, context: TaskContext, exit_code: int) -> None: ...
    def is_completed(self) -> bool: ...


class Runner(Protocol[T, U]):
    scheduler: Scheduler[T]

    # ワーカーへケースを割り当て、全ケースが完了するまで管理します。
    def run(self, worker: Worker[T, U]) -> None: ...


SUCCESS_EXIT_CODE = 0
WORKER_PROTOCOL_ERROR_EXIT_CODE = 1
