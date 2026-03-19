"""Experiment runner の軽量プロトコル定義。

このモジュールは、スケジューラ・ランナー・ワーカー間で共有する型と小さな契約を定義します。
実装は最小限に留め、ドキュメントコメントで期待される振る舞いを示します。
"""

from __future__ import annotations

from typing import Callable, Protocol, TypeAlias, TypeVar


T = TypeVar("T")
U = TypeVar("U")

# `TaskContext` はワーカーへ渡す環境や設定を表す辞書です（文字列値のみを想定）。
TaskContext: TypeAlias = dict[str, str]


class ResourceEstimate(Protocol):
    """ケースごとのリソース見積もりを表すプロトコル。

    - 実際のデータ構造（dataclass 等）は実験側で定義し、スケジューラはその値を消費する。
    """
    ...


class ResourceCapacity(Protocol):
    """利用可能なリソースの上限を表すプロトコル。"""

    @property
    def max_workers(self) -> int: ...


class Worker(Protocol[T, U]):
    """ワーカーはタスク（case）を受け取り `TaskContext` 下で実行するコール可能オブジェクトを要求する。

    - エラー時は例外を投げても構わないが、ランナーは終了コードで結果を受け取る契約になっている。
    """

    task: Callable[[T, TaskContext], U]

    def __call__(self, case: T, context: TaskContext) -> int: ...


class ResourceEstimatingWorker(Worker[T, U], Protocol[T, U]):
    """`resource_estimate(case)` を提供するワーカーの拡張プロトコル。"""

    def resource_estimate(self, case: T) -> ResourceEstimate: ...


class Scheduler(Protocol[T]):
    """スケジューラは次に実行すべきケースの選択と完了通知を受け取る責務を負う。"""

    @property
    def resource_capacity(self) -> ResourceCapacity: ...

    def next_case(self) -> tuple[T, TaskContext] | None: ...

    def on_finish(self, case: T, context: TaskContext, exit_code: int) -> None: ...

    def is_completed(self) -> bool: ...


class Runner(Protocol[T, U]):
    """ランナーはスケジューラとワーカーを使って全ケースを実行する役割を持つ。"""

    scheduler: Scheduler[T]

    def run(self, worker: Worker[T, U]) -> None: ...


SUCCESS_EXIT_CODE = 0
WORKER_PROTOCOL_ERROR_EXIT_CODE = 1
