from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
from typing import Callable, Generic, Mapping, Protocol, Sequence, TypeVar, cast

from .protocols import TaskContext, Worker
from .runner import StandardResourceCapacity, StandardScheduler


# ジェネリック型定義
T = TypeVar("T")
U = TypeVar("U")

# 環境変数名の候補。ユーザーがGPUの可視性を制御するために使う。
_GPU_ENV_NAMES = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")

# モジュールの前提環境 (整理済み)
# - GPU の可視性は環境変数 `NVIDIA_VISIBLE_DEVICES` または `CUDA_VISIBLE_DEVICES`
#   によって制御されることを前提とする。形式はカンマ区切りの整数列、
#   空文字 / "-1" / "none" / "void" は "GPU を使わない" を意味し、
#   "all" は検出された全 GPU を意味する。
# - システムの GPU メモリ検出には `nvidia-smi --query-gpu=index,memory.total`
#   を利用する。出力は CSV (noheader, nounits) で MiB 単位と仮定する。
def _validate_non_negative_int(value: int, field_name: str, /) -> int:
    """
    値を正規化して非負整数を返すヘルパー。

    - 可能なら int に変換して返す（OS 由来の値や外部入力を正規化する目的）。
    - bool は意図しない混入を避けるため拒否する。
    - 変換できない場合は TypeError、負なら ValueError を投げる。
    """
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be int, bool is not allowed.")
    try:
        iv = int(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"{field_name} must be int-convertible.") from exc
    if iv < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return iv


def _validate_positive_int(value: int, field_name: str, /) -> int:
    """
    値を正規化して正の整数（>=1）を返すヘルパー。

    - int に変換可能な値は int に変換して返す。
    - bool は拒否する。
    - 変換不能や 1 未満の値は例外となる。
    """
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be int, bool is not allowed.")
    try:
        iv = int(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"{field_name} must be int-convertible.") from exc
    if iv < 1:
        raise ValueError(f"{field_name} must be positive.")
    return iv


@dataclass(frozen=True)
class GPUDeviceCapacity:
    gpu_id: int
    memory_bytes: int
    max_slots: int = 1

    def __post_init__(self) -> None:
        # GPU ID とメモリ量は非負、スロット数は正であることを検証する。
        _validate_non_negative_int(self.gpu_id, "gpu_id")
        _validate_non_negative_int(self.memory_bytes, "memory_bytes")
        _validate_positive_int(self.max_slots, "max_slots")


@dataclass(frozen=True)
class FullResourceCapacity(StandardResourceCapacity):
    host_memory_bytes: int = 0
    gpu_devices: tuple[GPUDeviceCapacity, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        # ホストメモリは非負であることを検証
        _validate_non_negative_int(self.host_memory_bytes, "host_memory_bytes")

        # GPU の ID は重複がないことを保証する
        seen_gpu_ids: set[int] = set()
        for gpu_device in self.gpu_devices:
            if gpu_device.gpu_id in seen_gpu_ids:
                raise ValueError("gpu_ids must be unique.")
            seen_gpu_ids.add(gpu_device.gpu_id)

    @classmethod
    def from_system(
        cls,
        *,
        max_workers: int | None = None,
        host_memory_bytes: int | None = None,
        gpu_devices: Sequence[GPUDeviceCapacity] | None = None,
        environ: Mapping[str, str] | None = None,
        gpu_query_rows: Sequence[tuple[int, int]] | None = None,
        gpu_max_slots: int = 1,
    ) -> FullResourceCapacity:
        """
        システム環境から FullResourceCapacity を構築するユーティリティ。

        - 引数で明示的に与えられた値は優先される。
        - 与えられなければ自動検出関数（CPU数、ホストメモリ、GPU 情報）を使う。
        - GPU の検出は `nvidia-smi` を呼び出すため、環境によっては失敗する可能性がある。
        """
        resolved_max_workers = (
            detect_max_workers() if max_workers is None else max_workers
        )
        resolved_host_memory_bytes = (
            detect_host_memory_bytes()
            if host_memory_bytes is None
            else host_memory_bytes
        )
        resolved_gpu_devices = (
            detect_gpu_devices(
                environ=environ,
                query_rows=gpu_query_rows,
                max_slots=gpu_max_slots,
            )
            if gpu_devices is None
            else tuple(gpu_devices)
        )
        return cls(
            max_workers=resolved_max_workers,
            host_memory_bytes=resolved_host_memory_bytes,
            gpu_devices=tuple(resolved_gpu_devices),
        )



@dataclass(frozen=True)
class FullResourceEstimate:
    host_memory_bytes: int = 0
    gpu_count: int = 0
    gpu_memory_bytes: int = 0
    gpu_slots: int = 1

    def __post_init__(self) -> None:
        # ケースごとの資源見積もりの妥当性検査。
        # - ホストメモリ、GPU 数、GPU メモリはいずれも非負である必要がある。
        # - GPU スロットは少なくとも 1 である必要がある（GPU を使わないケースでもデフォルト 1）。
        # - GPU 数が 0 のときは GPU メモリは 0 でないと不整合になるため拒否する。
        _validate_non_negative_int(self.host_memory_bytes, "host_memory_bytes")
        _validate_non_negative_int(self.gpu_count, "gpu_count")
        _validate_non_negative_int(self.gpu_memory_bytes, "gpu_memory_bytes")
        _validate_positive_int(self.gpu_slots, "gpu_slots")

        if self.gpu_count == 0 and self.gpu_memory_bytes != 0:
            raise ValueError("gpu_memory_bytes must be 0 when gpu_count is 0.")


def detect_max_workers(cpu_count: int | None = None, /) -> int:
    """
    システムの CPU コア数からワーカー数を決める。

    - 引数 `cpu_count` が与えられていればそれを使い、なければ `os.cpu_count()` を利用する。
    - 返り値は 1 以上の正の整数であることを保証する。
    """
    resolved_cpu_count = os.cpu_count() if cpu_count is None else cpu_count
    return _validate_positive_int(resolved_cpu_count, "cpu_count")


def detect_host_memory_bytes(
    page_size: int | None = None,
    phys_pages: int | None = None,
    /,
) -> int:
    """
    ホストの物理メモリをバイト単位で返す。

    - `os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')` を用いて計算する。
    - テストのために引数でページサイズ/ページ数を注入できるようにしている。
    """
    resolved_page_size = os.sysconf("SC_PAGE_SIZE") if page_size is None else page_size
    resolved_phys_pages = (
        os.sysconf("SC_PHYS_PAGES") if phys_pages is None else phys_pages
    )
    return _validate_positive_int(
        resolved_page_size,
        "page_size",
    ) * _validate_positive_int(
        resolved_phys_pages,
        "phys_pages",
    )

'''
review:
_visible_gpu_ids_from_environment と _query_gpu_rows_from_system は、環境変数の解釈とシステムからの GPU 情報取得を分担して行う関数で、detect_gpu_devices 内で呼び出される。
この分割は責務の分離という観点で適切に見える。
この段階で正規化すれば、のちの検証は不要
'''


# Note: 上の詳細説明に従って、環境変数の解析は各呼び出し側で簡潔に行います。
# 旧来の `_visible_gpu_ids_from_environment` は単純化のため削除しました。


def _query_gpu_rows_from_system(
    environ: Mapping[str, str] | None = None,
    /,
) -> list[tuple[int, int]]:
    """
    システム上の GPU 情報を `nvidia-smi` で取得し、(index, memory_bytes) のリストを返す。

    - `nvidia-smi` が存在しない、または呼び出しに失敗した場合は、環境変数で GPU 指定が
      なされているかを確認し、必要な場合は RuntimeError を投げる。
    - 出力パースで期待される形式でない場合は ValueError を投げる。
    """
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        # nvidia-smi が無い環境では GPU メモリ量を検出できない。
        # ただし環境変数で明示的に GPU を指定している場合、メモリ量の確認が必須なので失敗させる。
        source = os.environ if environ is None else environ
        # 簡潔に環境変数を解釈する（詳細はモジュール先頭コメント参照）
        specified = None
        for env_name in _GPU_ENV_NAMES:
            raw = source.get(env_name)
            if raw is None:
                continue
            s = raw.strip().lower()
            if s in {"", "-1", "none", "void"}:
                specified = ()
                break
            if s == "all":
                specified = None
                break
            specified = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
            break
        if specified not in {None, ()}:
            raise RuntimeError("nvidia-smi is required to detect GPU memory.") from exc
        return []

    gpu_rows: list[tuple[int, int]] = []
    for line in completed.stdout.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue
        index_text, memory_mib_text = [part.strip() for part in stripped_line.split(",")]
        if not index_text.isdigit() or not memory_mib_text.isdigit():
            raise ValueError("nvidia-smi output has unexpected format.")
        # 出力は MiB 単位で返ってくるため、バイト単位へ変換する
        gpu_rows.append((int(index_text), int(memory_mib_text) * 1024 * 1024))
    return gpu_rows


def detect_gpu_devices(
    *,
    environ: Mapping[str, str] | None = None,
    query_rows: Sequence[tuple[int, int]] | None = None,
    max_slots: int = 1,
) -> tuple[GPUDeviceCapacity, ...]:
    _validate_positive_int(max_slots, "max_slots")
    # 環境変数で明示的に GPU を無効にしている場合は空のタプルを返す。
    source = os.environ if environ is None else environ
    visible_gpu_ids = None
    for env_name in _GPU_ENV_NAMES:
        raw = source.get(env_name)
        if raw is None:
            continue
        s = raw.strip().lower()
        if s in {"", "-1", "none", "void"}:
            visible_gpu_ids = ()
            break
        if s == "all":
            visible_gpu_ids = None
            break
        visible_gpu_ids = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
        break
    if visible_gpu_ids == ():
        return ()

    # query_rows が与えられていればテスト用の注入値を使い、そうでなければシステムに問い合わせる
    resolved_rows = list(
        _query_gpu_rows_from_system(environ) if query_rows is None else query_rows
    )

    # GPU 行情報を辞書にしてインデックスで参照できるようにする
    rows_by_gpu_id: dict[int, int] = {}
    for gpu_id, memory_bytes in resolved_rows:
        # 正規化して int を得る（外部注入値や sysconf 由来の値を安全に扱う）
        gpu_id = _validate_non_negative_int(gpu_id, "gpu_id")
        memory_bytes = _validate_non_negative_int(memory_bytes, "memory_bytes")
        rows_by_gpu_id[gpu_id] = memory_bytes

    if visible_gpu_ids is None:
        # 環境変数で明示的に指定がない場合、検出された順序で全 GPU を選択対象とする
        selected_gpu_ids = [gpu_id for gpu_id, _ in resolved_rows]
    else:
        # 環境変数で指定がある場合は、その ID が検出結果に含まれていることを確認する
        missing_gpu_ids = [
            gpu_id for gpu_id in visible_gpu_ids if gpu_id not in rows_by_gpu_id
        ]
        if missing_gpu_ids:
            raise ValueError("visible GPU ids are not present in detected GPU rows.")
        selected_gpu_ids = list(visible_gpu_ids)

    # 最終的に GPUDeviceCapacity のタプルを返す
    return tuple(
        GPUDeviceCapacity(
            gpu_id=gpu_id,
            memory_bytes=rows_by_gpu_id[gpu_id],
            max_slots=max_slots,
        )
        for gpu_id in selected_gpu_ids
    )


@dataclass(frozen=True)
class _PendingCase(Generic[T]):
    case: T
    estimate: FullResourceEstimate


@dataclass(frozen=True)
class _RunningAllocation:
    host_memory_bytes: int
    gpu_ids: tuple[int, ...]
    gpu_memory_bytes: int
    gpu_slots: int


class FullResourceWorker(Worker[T, U], Protocol[T, U]):
    def resource_estimate(self, case: T) -> FullResourceEstimate: ...


class StandardFullResourceScheduler(StandardScheduler[T], Generic[T]):
    @classmethod
    def from_worker(
        cls,
        resource_capacity: FullResourceCapacity,
        cases: list[T],
        worker: FullResourceWorker[T, U],
        context_builder: Callable[[T], TaskContext] | None = None,
        disable_gpu_preallocation: bool = False,
    ) -> StandardFullResourceScheduler[T]:
        return cls(
            resource_capacity=resource_capacity,
            cases=cases,
            estimate_builder=worker.resource_estimate,
            context_builder=context_builder,
            disable_gpu_preallocation=disable_gpu_preallocation,
        )

    def __init__(
        self,
        resource_capacity: FullResourceCapacity,
        cases: list[T],
        estimate_builder: Callable[[T], FullResourceEstimate],
        context_builder: Callable[[T], TaskContext] | None = None,
        disable_gpu_preallocation: bool = False,
    ) -> None:
        """
        スケジューラの内部状態を初期化する。

        - `estimate_builder` は各ケースに対して `FullResourceEstimate` を返す関数（あるいは bound method）である。
        - 待機中ケースを `_pending_entries` として、リソース割当可能かを評価できる形で保存する。
        - 利用可能なワーカースロット、ホストメモリ、GPU メモリ、GPU スロットを追跡する辞書を作る。
        """
        # 親クラスは cases を受け取るが、ここでは独自に pending を管理するため空リストで初期化する
        super().__init__(
            resource_capacity=resource_capacity,
            cases=[],
            context_builder=context_builder,
        )
        self._estimate_builder = estimate_builder
        self._disable_gpu_preallocation = disable_gpu_preallocation

        # 各ケースを資源見積り付きの保留エントリへ変換する
        self._pending_entries = [
            _PendingCase(
                case=case,
                estimate=self._build_estimate(case),
            )
            for case in cases
        ]

        # 初期の利用可能リソースを記録
        self._available_worker_slots = resource_capacity.max_workers
        self._available_host_memory_bytes = resource_capacity.host_memory_bytes
        self._available_gpu_memory_bytes = {
            gpu_device.gpu_id: gpu_device.memory_bytes
            for gpu_device in resource_capacity.gpu_devices
        }
        self._available_gpu_slots = {
            gpu_device.gpu_id: gpu_device.max_slots
            for gpu_device in resource_capacity.gpu_devices
        }

        # 実行中割当を context の id をキーにして保持（on_finish で参照して解放する）
        self._active_allocations: dict[int, _RunningAllocation] = {}

    @property
    def resource_capacity(self) -> FullResourceCapacity:
        return cast(FullResourceCapacity, self._resource_capacity)

    def _build_estimate(self, case: T) -> FullResourceEstimate:
        estimate = self._estimate_builder(case)
        if not self._fits_capacity(estimate):
            raise ValueError("case resource estimate exceeds FullResourceCapacity.")
        return estimate

    def _fits_capacity(self, estimate: FullResourceEstimate) -> bool:
        """
        与えられた見積りが全体の容量で満たせるかをチェックする。

        - ホストメモリが足りなければ False。
        - GPU を要求しないケース（gpu_count == 0）はホストメモリチェックのみで OK とする。
        - GPU が必要な場合は、要求されるスロット数とメモリ量を満たす GPU が必要数以上存在するかを確認する。
        """
        if estimate.host_memory_bytes > self.resource_capacity.host_memory_bytes:
            return False
        if estimate.gpu_count == 0:
            return True

        compatible_gpu_count = 0
        for gpu_device in self.resource_capacity.gpu_devices:
            # その GPU が要求するスロット数・メモリ量を満たすかを確認する
            if gpu_device.max_slots < estimate.gpu_slots:
                continue
            if gpu_device.memory_bytes < estimate.gpu_memory_bytes:
                continue
            compatible_gpu_count += 1
            if compatible_gpu_count >= estimate.gpu_count:
                return True
        return False

    def _select_gpu_ids(self, estimate: FullResourceEstimate) -> tuple[int, ...] | None:
        if estimate.gpu_count == 0:
            return ()
        # 現在の利用可能リソースから、実際に割り当て可能な GPU ID の組を選ぶ。
        # 最初に条件を満たす順で選んでいき、要求数に達したらその組を返す。
        selected_gpu_ids: list[int] = []
        for gpu_device in self.resource_capacity.gpu_devices:
            gpu_id = gpu_device.gpu_id
            if self._available_gpu_slots[gpu_id] < estimate.gpu_slots:
                continue
            if self._available_gpu_memory_bytes[gpu_id] < estimate.gpu_memory_bytes:
                continue
            selected_gpu_ids.append(gpu_id)
            if len(selected_gpu_ids) == estimate.gpu_count:
                return tuple(selected_gpu_ids)
        # 条件を満たしきれない場合は None を返して割当不可を示す
        return None

    def _try_allocate(self, estimate: FullResourceEstimate) -> _RunningAllocation | None:
        """
        実際にリソース割当を試みる。

        - ワーカースロット、ホストメモリ、GPU の順にチェックしていき、全て満たせばリソースを減算して割当情報を返す。
        - 割当不可なら None を返す。
        """
        if self._available_worker_slots < 1:
            return None
        if self._available_host_memory_bytes < estimate.host_memory_bytes:
            return None

        gpu_ids = self._select_gpu_ids(estimate)
        if gpu_ids is None:
            return None

        # リソースを減らして割当を確定する
        self._available_worker_slots -= 1
        self._available_host_memory_bytes -= estimate.host_memory_bytes
        for gpu_id in gpu_ids:
            self._available_gpu_slots[gpu_id] -= estimate.gpu_slots
            self._available_gpu_memory_bytes[gpu_id] -= estimate.gpu_memory_bytes

        return _RunningAllocation(
            host_memory_bytes=estimate.host_memory_bytes,
            gpu_ids=gpu_ids,
            gpu_memory_bytes=estimate.gpu_memory_bytes,
            gpu_slots=estimate.gpu_slots,
        )

    def next_case(self) -> tuple[T, TaskContext] | None:
        """
        次に実行可能なケースを返す。

        - 待機中エントリを先頭から順に見て、割当可能ならそのケースを取り出してコンテキストを生成する。
        - GPU を割り当てた場合は `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES` をコンテキストにセットする。
        - `XLA_PYTHON_CLIENT_PREALLOCATE` は必要に応じて無効化するためのフラグ。
        - 実際の割当情報はコンテキストの id をキーに `_active_allocations` に保存しておく。
        """
        for index, pending_entry in enumerate(self._pending_entries):
            allocation = self._try_allocate(pending_entry.estimate)
            if allocation is None:
                continue

            # 予約リストから取り除く
            self._pending_entries.pop(index)
            context = self._build_context(pending_entry.case)
            if allocation.gpu_ids:
                gpu_ids_text = ",".join(str(gpu_id) for gpu_id in allocation.gpu_ids)
                context["gpu_ids"] = gpu_ids_text
                context["CUDA_VISIBLE_DEVICES"] = gpu_ids_text
                context["NVIDIA_VISIBLE_DEVICES"] = gpu_ids_text
                if len(allocation.gpu_ids) == 1:
                    # 単一 GPU の場合は `gpu_id` も便宜的に設定する
                    context["gpu_id"] = str(allocation.gpu_ids[0])
                if self._disable_gpu_preallocation:
                    # JAX/XLA の自動メモリプリアロケーションを無効化したいとき
                    context["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

            # コンテキストをキーに割当を記録しておく（on_finish で復元する）
            self._active_allocations[id(context)] = allocation
            return pending_entry.case, context
        return None

    def on_finish(self, case: T, context: TaskContext, exit_code: int) -> None:
        super().on_finish(case, context, exit_code)

        # 実行完了後はスケジューラ側で割り当てたリソースを解放する。
        allocation = self._active_allocations.pop(id(context), None)
        if allocation is None:
            raise ValueError("context is not associated with an active allocation.")

        # ワーカースロット、ホストメモリ、GPU スロット/メモリを戻す
        self._available_worker_slots += 1
        self._available_host_memory_bytes += allocation.host_memory_bytes
        for gpu_id in allocation.gpu_ids:
            self._available_gpu_slots[gpu_id] += allocation.gpu_slots
            self._available_gpu_memory_bytes[gpu_id] += allocation.gpu_memory_bytes

    def is_completed(self) -> bool:
        return not self._pending_entries
