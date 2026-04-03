"""Experiment runner の軽量プロトコル定義。

このモジュールは、スケジューラ・ランナー・ワーカー間で共有する型と小さな契約を定義します。
実装は最小限に留め、ドキュメントコメントで期待される振る舞いを示します。
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any, Callable, Protocol, TypeAlias, TypeVar

if TYPE_CHECKING:
    from .execution_result import ExecutionResult


T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)
__all__ = [
    "TaskContext",
    "ContextInitializer",
    "SkipController",
    "ResourceEstimate",
    "ResourceCapacity",
    "Worker",
    "Scheduler",
    "Runner",
]

# `TaskContext` はワーカーへ渡す環境や設定を表す。
# 環境変数辞書のような構造化データも流せるよう Any を許容する。
TaskContext: TypeAlias = dict[str, Any]
ContextInitializer: TypeAlias = Callable[[TaskContext], None]


class SkipController(Protocol[T_contra]):
    """起動前 skip と完了後 state 更新を持つ controller protocol。"""

    def should_skip(self, case: T_contra, context: TaskContext) -> str | None: ...

    def update(
        self,
        case: T_contra,
        context: TaskContext,
        result: "ExecutionResult",
    ) -> None: ...


class ResourceEstimate(Protocol):
    """ケースごとのリソース見積もりを表すプロトコル。

    - 実際のデータ構造（dataclass 等）は実験側で定義し、スケジューラはその値を消費する。
    """
    ...


class ResourceCapacity(Protocol):
    """利用可能なリソースの上限を表すプロトコル。"""

    @property
    def max_workers(self) -> int: ...


class Worker(Protocol[T]):
    """ワーカーはタスク（case）を受け取り `TaskContext` 下で実行するコール可能オブジェクトを要求する。

    - 正常系では `ExecutionResult(status="ok")` を返す。
    - 失敗系を自前で扱う worker は `ExecutionResult(status="failed")` を返してよい。
    - 例外は child runtime が structured diagnostics に正規化する。
    """

    task: Callable[[T, TaskContext], object]
    initializer: ContextInitializer

    def __call__(self, case: T, context: TaskContext) -> "ExecutionResult": ...

    def resource_estimate(self, case: T) -> ResourceEstimate: ...


class Scheduler(Protocol[T]):
    """スケジューラは次に実行すべきケースの選択と完了通知を受け取る責務を負う。"""

    @property
    def resource_capacity(self) -> ResourceCapacity: ...

    @property
    def completions(self) -> list[Any]: ...

    @property
    def total_case_count(self) -> int: ...

    @property
    def skip_controller(self) -> SkipController[T] | None: ...

    def next_case(self) -> tuple[T, TaskContext] | None: ...

    def on_finish(
        self,
        case: T,
        context: TaskContext,
        result: "ExecutionResult",
    ) -> None: ...

    def is_completed(self) -> bool: ...


class Runner(Protocol[T]):
    """ランナーはスケジューラとワーカーを使って全ケースを実行する役割を持つ。"""

    scheduler: Scheduler[T]

    def run(self, worker: Worker[T]) -> None: ...
