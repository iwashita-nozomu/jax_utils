from __future__ import annotations

from typing import Callable, Protocol, TypeAlias, TypeVar


T = TypeVar("T")
U = TypeVar("U")

TaskContext: TypeAlias = dict[str, str]


class ResourceEstimate(Protocol):
    # 実際の見積もり実装は task を定義する実験コードと同じ場所に置き、
    # scheduler 側はその結果だけを消費する方針です。
    ...


class ResourceCapacity(Protocol):
    @property
    def max_workers(self) -> int: ...


class Worker(Protocol[T, U]):
    task: Callable[[T, TaskContext], U]

    # Task を実行して終了コードを返します。ログ出力は worker 側で扱います。
    def __call__(self, case: T, context: TaskContext) -> int: ...


class ResourceEstimatingWorker(Worker[T, U], Protocol[T, U]):
    # task 実装と同じ場所で定義された per-case の資源見積もりを返します。
    def resource_estimate(self, case: T) -> ResourceEstimate: ...


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
