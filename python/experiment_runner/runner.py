"""ランナーと標準的なスケジューラ/ワーカーの実装。

軽量な抽象（`StandardScheduler` / `StandardRunner` / `StandardWorker`）を提供します。
実装は並列実行と完了ハンドリングの単純な契約に従います。

JAX fork() 互換性: StandardRunner は spawn コンテキストでワーカープロセスを起動
することで、fork() ベースの multiprocessing 問題を回避できます。
use_spawn_context=True（デフォルト）で有効です。
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
import traceback
import time
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from .jax_context import check_picklable, create_jax_safe_process_pool
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
        # NOTE: _resource_estimator は Worker インスタンスが from_worker()
        #       で生成される場合のみ呼び出される。そのため常に None ではない。
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

    - spawn コンテキストで ProcessPoolExecutor を起動（CPU-only でも）
    - JAX fork() 互換性問題を完全に回避
    - 完了ごとに `scheduler.on_finish` を呼び、次のケースを投入する。
    - プログレス報告コールバックをサポート
    """

    def __init__(
        self,
        scheduler: Scheduler[T],
        progress_callback: ProgressCallback = None,
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
        self.scheduler = scheduler
        self.progress_callback = progress_callback

    def run(self, worker: Worker[T, U]) -> None:
        """ケースを並列実行する。

        spawn コンテキストで ProcessPoolExecutor を起動し、
        JAX fork() 互換性を確保しながら並列実行する。

        Parameters
        ----------
        worker : Worker
            各ケースを実行するワーカー

        Raises
        ------
        ValueError
            ワーカーが pickle 化不可能な場合。ProcessPoolExecutor で
            別プロセスに送出できる必要があります。
        """
        # ワーカーが pickle 化可能であることを確認
        check_picklable(worker, name="Worker")

        # 総ケース数をここで計算（scheduler の種類に依存しない抽象的方法）
        # 既完了分 + 未実行分を合算
        total_cases = len(self.scheduler.completions)
        pending_count = 0

        # scheduler の属性で未実行ケース数を取得（StandardScheduler と
        # StandardFullResourceScheduler の両方に対応）
        if hasattr(self.scheduler, "_pending_cases"):
            # StandardScheduler の場合
            pending_count = len(self.scheduler._pending_cases)  # type: ignore[attr-defined]
        elif hasattr(self.scheduler, "_pending_entries"):
            # StandardFullResourceScheduler の場合
            pending_count = len(self.scheduler._pending_entries)  # type: ignore[attr-defined]

        total_cases += pending_count

        # 常に spawn コンテキストで executor を起動
        with create_jax_safe_process_pool(
            max_workers=self.scheduler.resource_capacity.max_workers
        ) as ex:
            self._execute_with_executor(ex, worker, total_cases)

    def _execute_with_executor(
        self,
        executor: ProcessPoolExecutor,
        worker: Worker[T, U],
        total_cases: int,
    ) -> None:
        """
        Executor を使用してケースを実行する。

        プログレスコールバックが登録されている場合は、
        ケース完了ごとに進捗状況を報告する。

        Parameters
        ----------
        executor : ProcessPoolExecutor
            実行用の Executor
        worker : Worker[T, U]
            各ケースを実行するワーカー
        total_cases : int
            全体のケース数（既完了 + 未実行）
        """
        running: dict[Future[int], tuple[T, TaskContext]] = {}
        start_time = time.time()

        while not self.scheduler.is_completed() or running:
            # 新しいケースを投入できるまで試行
            while True:
                job = self.scheduler.next_case()
                if job is None:
                    break
                case, context = job
                fut = executor.submit(worker, case, context)
                running[fut] = job

            if not running:
                continue

            # 最初に完了したタスクを待つ
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for fut in done:
                case, context = running.pop(fut)
                self.scheduler.on_finish(case, context, fut.result())

                # プログレス報告を実行
                if self.progress_callback is not None:
                    elapsed = time.time() - start_time
                    completed = len(self.scheduler.completions)
                    self.progress_callback(
                        completed,
                        total_cases,
                        elapsed,
                        len(running),
                    )
