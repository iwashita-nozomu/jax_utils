"""複合リソース管理型スケジューラ。

このモジュールは GPU メモリ、ホストメモリ、ワーカースロットを統合的に管理し、
より堅牢な並列実行を提供します。

推奨される主要クラス:
  - StandardFullResourceScheduler: 複合リソース制約を満たすスケジューラ
  - FullResourceCapacity: リソース容量定義
  - GPUDeviceCapacity: GPU デバイス情報
  - detect_gpu_devices(): GPU 環境自動検出
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import subprocess
from typing import Callable, Generic, Mapping, Protocol, Sequence, TypeVar

from jax_util.xla_env import build_gpu_env, merge_env_vars

from .execution_result import ExecutionResult
from .protocols import SkipController, TaskContext, Worker
from .runner import StandardCompletion

T = TypeVar("T")

__all__ = [
    "GPUDeviceCapacity",
    "GPUEnvironmentConfig",
    "FullResourceCapacity",
    "FullResourceEstimate",
    "detect_max_workers",
    "detect_host_memory_bytes",
    "detect_gpu_devices",
    "StandardFullResourceScheduler",
]

# 環境変数名の候補。ユーザーがGPUの可視性を制御するために使う。
_GPU_ENV_NAMES = ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES")

# モジュールの前提環境 (整理済み)
# - GPU の可視性は環境変数 `NVIDIA_VISIBLE_DEVICES` または `CUDA_VISIBLE_DEVICES`
#   によって制御されることを前提とする。形式はカンマ区切りの整数列、
#   空文字 / "-1" / "none" / "void" は "GPU を使わない" を意味し、
#   "all" は検出された全 GPU を意味する。
# - システムの GPU メモリ検出には `nvidia-smi --query-gpu=index,memory.total`
#   を利用する。出力は CSV (noheader, nounits) で MiB 単位と仮定する。
def _validate_int(
    value: int,
    field_name: str,
    /,
    *,
    minimum: int = 0,
) -> int:
    """値を正規化し、指定範囲内の整数を返す。"""
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be int, bool is not allowed.")
    try:
        iv = int(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError(f"{field_name} must be int-convertible.") from exc
    if iv < minimum:
        if minimum > 1:
            raise ValueError(f"{field_name} must be >= {minimum}.")
        if minimum == 1:
            raise ValueError(f"{field_name} must be positive.")
        raise ValueError(f"{field_name} must be non-negative.")
    return iv


def _available_cpu_indices() -> list[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    cpu_count = os.cpu_count() or 1
    return list(range(cpu_count))


def _partition_cpu_indices(num_workers: int, /) -> list[list[int]]:
    cpu_indices = _available_cpu_indices()
    if num_workers < 1:
        raise ValueError("num_workers must be positive.")
    if not cpu_indices:
        return [[] for _ in range(num_workers)]

    base = len(cpu_indices) // num_workers
    extra = len(cpu_indices) % num_workers
    groups: list[list[int]] = []
    start = 0
    for worker_index in range(num_workers):
        group_size = base + (1 if worker_index < extra else 0)
        if group_size <= 0:
            groups.append([cpu_indices[worker_index % len(cpu_indices)]])
            continue
        stop = start + group_size
        groups.append(cpu_indices[start:stop])
        start = stop
    return groups


@dataclass(frozen=True)
class GPUDeviceCapacity:
    gpu_id: int
    memory_bytes: int
    max_slots: int = 1

    def __post_init__(self) -> None:
        # GPU ID とメモリ量は非負、スロット数は正であることを検証する。
        _validate_int(self.gpu_id, "gpu_id", minimum=0)
        _validate_int(self.memory_bytes, "memory_bytes", minimum=0)
        _validate_int(self.max_slots, "max_slots", minimum=1)


@dataclass(frozen=True)
class GPUEnvironmentConfig:
    disable_preallocation: bool = False
    memory_fraction: float | None = None
    xla_client_allocator: str | None = None
    tf_gpu_allocator: str | None = None
    use_cuda_host_allocator: bool | None = None

    def __post_init__(self) -> None:
        if self.memory_fraction is not None:
            if isinstance(self.memory_fraction, bool):
                raise TypeError("memory_fraction must be numeric, bool is not allowed.")
            if not 0.0 < float(self.memory_fraction) <= 1.0:
                raise ValueError("memory_fraction must be within (0, 1].")
        if self.xla_client_allocator is not None and not self.xla_client_allocator:
            raise ValueError("xla_client_allocator must not be empty.")
        if self.tf_gpu_allocator is not None and not self.tf_gpu_allocator:
            raise ValueError("tf_gpu_allocator must not be empty.")

    def build_environment_variables(self) -> dict[str, str]:
        return build_gpu_env(
            disable_preallocation=self.disable_preallocation,
            memory_fraction=self.memory_fraction,
            allocator=self.xla_client_allocator,
            tf_gpu_allocator=self.tf_gpu_allocator,
            use_cuda_host_allocator=self.use_cuda_host_allocator,
        )


def _merge_gpu_environment_config(
    *,
    disable_gpu_preallocation: bool,
    gpu_environment_config: GPUEnvironmentConfig | None,
) -> GPUEnvironmentConfig | None:
    if gpu_environment_config is None:
        if not disable_gpu_preallocation:
            return None
        return GPUEnvironmentConfig(disable_preallocation=True)
    return GPUEnvironmentConfig(
        disable_preallocation=(
            gpu_environment_config.disable_preallocation or disable_gpu_preallocation
        ),
        memory_fraction=gpu_environment_config.memory_fraction,
        xla_client_allocator=gpu_environment_config.xla_client_allocator,
        tf_gpu_allocator=gpu_environment_config.tf_gpu_allocator,
        use_cuda_host_allocator=gpu_environment_config.use_cuda_host_allocator,
    )


@dataclass(frozen=True)
class FullResourceCapacity:
    max_workers: int
    host_memory_bytes: int = 0
    gpu_devices: tuple[GPUDeviceCapacity, ...] = ()

    def __post_init__(self) -> None:
        _validate_int(self.max_workers, "max_workers", minimum=1)
        # ホストメモリは非負であることを検証
        _validate_int(self.host_memory_bytes, "host_memory_bytes", minimum=0)

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
        _validate_int(self.host_memory_bytes, "host_memory_bytes", minimum=0)
        _validate_int(self.gpu_count, "gpu_count", minimum=0)
        _validate_int(self.gpu_memory_bytes, "gpu_memory_bytes", minimum=0)
        _validate_int(self.gpu_slots, "gpu_slots", minimum=1)

        if self.gpu_count == 0 and self.gpu_memory_bytes != 0:
            raise ValueError("gpu_memory_bytes must be 0 when gpu_count is 0.")


def detect_max_workers(cpu_count: int | None = None, /) -> int:
    """
    システムの CPU コア数からワーカー数を決める。

    - 引数 `cpu_count` が与えられていればそれを使い、なければ `os.cpu_count()` を利用する。
    - 返り値は 1 以上の正の整数であることを保証する。
    """
    # 引数が None のときはシステムの値を使う。システムの判定不能（None）の場合は
    # スケジューラは動作できないため明示的に例外を投げる。
    if cpu_count is None:
        sys_count = os.cpu_count()
        if sys_count is None:
            raise RuntimeError("unable to determine CPU count from os.cpu_count()")
        return _validate_int(sys_count, "cpu_count", minimum=1)
    return _validate_int(cpu_count, "cpu_count", minimum=1)


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
    # ページサイズ / ページ数の解決をインライン化して直接乗算する。
    page_size_value = _validate_int(
        os.sysconf("SC_PAGE_SIZE") if page_size is None else page_size,
        "page_size",
        minimum=1,
    )
    phys_pages_value = _validate_int(
        os.sysconf("SC_PHYS_PAGES") if phys_pages is None else phys_pages,
        "phys_pages",
        minimum=1,
    )
    # 実効メモリを 0.8 倍に制限して、常用時の安定性を優先する。
    return int(page_size_value * phys_pages_value * 0.8)

def _resolve_visible_gpu_ids(
    environ: Mapping[str, str] | None = None,
    /,
) -> tuple[int, ...] | None:
    """環境変数から可視 GPU 一覧を解決する。

    - 明示指定がなければ `None` を返す。
    - 空文字 / "-1" / "none" / "void" は「GPU を使わない」とみなして空タプルを返す。
    - "all" は検出された全 GPU を使う指定として `None` を返す。
    """
    source = os.environ if environ is None else environ
    for env_name in _GPU_ENV_NAMES:
        raw = source.get(env_name)
        if raw is None:
            continue
        stripped = raw.strip().lower()
        if stripped in {"", "-1", "none", "void"}:
            return ()
        if stripped == "all":
            return None
        return tuple(int(token.strip()) for token in raw.split(",") if token.strip())
    return None


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
        # 環境変数で明示的に GPU を指定している場合、メモリ量の確認が必須なので失敗させる。
        specified = _resolve_visible_gpu_ids(environ)
        if specified not in {None, ()}:
            raise RuntimeError(
                f"GPU devices specified in environment variables ({_GPU_ENV_NAMES}) "
                "but nvidia-smi is not found or failed. "
                "Ensure nvidia-smi is available or set CUDA_VISIBLE_DEVICES='' to disable GPU."
            ) from exc
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
    _validate_int(max_slots, "max_slots", minimum=1)
    # 環境変数で明示的に GPU を無効にしている場合は空のタプルを返す。
    visible_gpu_ids = _resolve_visible_gpu_ids(environ)
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
        gpu_id = _validate_int(gpu_id, "gpu_id", minimum=0)
        memory_bytes = _validate_int(memory_bytes, "memory_bytes", minimum=0)
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

    # 最終的に GPUDeviceCapacity のタプルを返す。
    # GPU メモリも 0.8 倍を上限として扱い、長時間実験時の OOM を減らす。
    return tuple(
        GPUDeviceCapacity(
            gpu_id=gpu_id,
            memory_bytes=int(rows_by_gpu_id[gpu_id] * 0.8),
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
    worker_slot_id: int
    cpu_affinity: tuple[int, ...]
    host_memory_bytes: int
    gpu_ids: tuple[int, ...]
    gpu_memory_bytes: int
    gpu_slots: int
    gpu_slot_assignments: tuple[tuple[int, tuple[int, ...]], ...]


class FullResourceWorker(Worker[T], Protocol[T]):
    def resource_estimate(self, case: T) -> FullResourceEstimate: ...


class StandardFullResourceScheduler(Generic[T]):
    @classmethod
    def from_worker(
        cls,
        cases: list[T],
        worker: FullResourceWorker[T],
        context_builder: Callable[[T], TaskContext] | None = None,
        skip_controller: "SkipController[T] | None" = None,
        disable_gpu_preallocation: bool = False,
        gpu_environment_config: GPUEnvironmentConfig | None = None,
        resource_capacity: FullResourceCapacity | None = None,
    ) -> StandardFullResourceScheduler[T]:
        resolved_capacity = (
            FullResourceCapacity.from_system(gpu_max_slots=1)
            if resource_capacity is None
            else resource_capacity
        )
        return cls(
            resource_capacity=resolved_capacity,
            cases=cases,
            estimate_builder=worker.resource_estimate,
            context_builder=context_builder,
            skip_controller=skip_controller,
            disable_gpu_preallocation=disable_gpu_preallocation,
            gpu_environment_config=gpu_environment_config,
        )

    def __init__(
        self,
        resource_capacity: FullResourceCapacity,
        cases: list[T],
        estimate_builder: Callable[[T], FullResourceEstimate],
        context_builder: Callable[[T], TaskContext] | None = None,
        skip_controller: "SkipController[T] | None" = None,
        disable_gpu_preallocation: bool = False,
        gpu_environment_config: GPUEnvironmentConfig | None = None,
    ) -> None:
        """
        スケジューラの内部状態を初期化する。

        - `estimate_builder` は各ケースに対して `FullResourceEstimate` を返す関数（あるいは bound method）である。
        - 待機中ケースを `_pending_entries` として、リソース割当可能かを評価できる形で保存する。
        - 利用可能なワーカースロット、ホストメモリ、GPU メモリ、GPU スロットを追跡する辞書を作る。
        """
        self._resource_capacity = resource_capacity
        self._context_builder = context_builder
        self.skip_controller = skip_controller
        self.completions: list[StandardCompletion[T]] = []
        self._estimate_builder = estimate_builder
        self._gpu_environment_config = _merge_gpu_environment_config(
            disable_gpu_preallocation=disable_gpu_preallocation,
            gpu_environment_config=gpu_environment_config,
        )

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
        self._available_worker_slot_ids = deque(range(resource_capacity.max_workers))
        cpu_groups = _partition_cpu_indices(resource_capacity.max_workers)
        self._worker_cpu_affinities = {
            worker_slot_id: tuple(cpu_groups[worker_slot_id])
            for worker_slot_id in range(resource_capacity.max_workers)
        }
        self._available_host_memory_bytes = resource_capacity.host_memory_bytes
        self._available_gpu_memory_bytes = {
            gpu_device.gpu_id: gpu_device.memory_bytes
            for gpu_device in resource_capacity.gpu_devices
        }
        self._available_gpu_slots = {
            gpu_device.gpu_id: gpu_device.max_slots
            for gpu_device in resource_capacity.gpu_devices
        }
        self._available_gpu_slot_ids = {
            gpu_device.gpu_id: list(range(gpu_device.max_slots))
            for gpu_device in resource_capacity.gpu_devices
        }

        # 実行中割当を context の id をキーにして保持（on_finish で参照して解放する）
        self._active_allocations: dict[int, _RunningAllocation] = {}

    @property
    def resource_capacity(self) -> FullResourceCapacity:
        return self._resource_capacity

    @property
    def total_case_count(self) -> int:
        return (
            len(self.completions)
            + len(self._pending_entries)
            + len(self._active_allocations)
        )

    def _build_context(self, case: T) -> TaskContext:
        if self._context_builder is None:
            return {}
        return dict(self._context_builder(case))

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
        eligible_gpu_ids = [
            gpu_device.gpu_id
            for gpu_device in self.resource_capacity.gpu_devices
            if self._available_gpu_slots[gpu_device.gpu_id] >= estimate.gpu_slots
            and self._available_gpu_memory_bytes[gpu_device.gpu_id] >= estimate.gpu_memory_bytes
        ]
        if len(eligible_gpu_ids) < estimate.gpu_count:
            return None
        sorted_gpu_ids = sorted(
            eligible_gpu_ids,
            key=lambda gpu_id: self._available_gpu_slots[gpu_id],
            reverse=True,
        )
        return tuple(sorted_gpu_ids[: estimate.gpu_count])

    def _try_allocate(self, estimate: FullResourceEstimate) -> _RunningAllocation | None:
        """
        実際にリソース割当を試みる。

        - ワーカースロット、ホストメモリ、GPU の順にチェックしていき、全て満たせばリソースを減算して割当情報を返す。
        - 割当不可なら None を返す。
        """
        if self._available_worker_slots < 1:
            return None
        if not self._available_worker_slot_ids:
            return None
        if self._available_host_memory_bytes < estimate.host_memory_bytes:
            return None

        gpu_ids = self._select_gpu_ids(estimate)
        if gpu_ids is None:
            return None

        gpu_slot_assignments: list[tuple[int, tuple[int, ...]]] = []
        for gpu_id in gpu_ids:
            available_slot_ids = self._available_gpu_slot_ids[gpu_id]
            if len(available_slot_ids) < estimate.gpu_slots:
                return None

        # リソースを減らして割当を確定する
        worker_slot_id = self._available_worker_slot_ids.popleft()
        cpu_affinity = self._worker_cpu_affinities.get(worker_slot_id, ())
        self._available_worker_slots -= 1
        self._available_host_memory_bytes -= estimate.host_memory_bytes
        for gpu_id in gpu_ids:
            selected_slot_ids = tuple(self._available_gpu_slot_ids[gpu_id][: estimate.gpu_slots])
            del self._available_gpu_slot_ids[gpu_id][: estimate.gpu_slots]
            self._available_gpu_slots[gpu_id] -= estimate.gpu_slots
            self._available_gpu_memory_bytes[gpu_id] -= estimate.gpu_memory_bytes
            gpu_slot_assignments.append((gpu_id, selected_slot_ids))

        return _RunningAllocation(
            worker_slot_id=worker_slot_id,
            cpu_affinity=cpu_affinity,
            host_memory_bytes=estimate.host_memory_bytes,
            gpu_ids=gpu_ids,
            gpu_memory_bytes=estimate.gpu_memory_bytes,
            gpu_slots=estimate.gpu_slots,
            gpu_slot_assignments=tuple(gpu_slot_assignments),
        )

    def _worker_label_for_allocation(self, allocation: _RunningAllocation, /) -> str:
        if not allocation.gpu_slot_assignments:
            return f"cpu-w{allocation.worker_slot_id}"

        labels: list[str] = []
        for gpu_id, slot_ids in allocation.gpu_slot_assignments:
            if len(slot_ids) == 1:
                labels.append(f"gpu-{gpu_id}-w{slot_ids[0]}")
            else:
                slot_ids_text = "-".join(str(slot_id) for slot_id in slot_ids)
                labels.append(f"gpu-{gpu_id}-w{slot_ids_text}")
        return "+".join(labels)

    def _single_gpu_slot(self, allocation: _RunningAllocation, /) -> int | None:
        if len(allocation.gpu_slot_assignments) != 1:
            return None
        _, slot_ids = allocation.gpu_slot_assignments[0]
        if len(slot_ids) != 1:
            return None
        return int(slot_ids[0])

    def next_case(self) -> tuple[T, TaskContext] | None:
        """
        次に実行可能なケースを返す。

        - 待機中エントリを先頭から順に見て、割当可能ならそのケースを取り出してコンテキストを生成する。
        - GPU を割り当てた場合は、実行環境を `context["environment_variables"]` にまとめて格納する。
        - 実際の割当情報はコンテキストの id をキーに `_active_allocations` に保存しておく。
        """
        for index, pending_entry in enumerate(self._pending_entries):
            allocation = self._try_allocate(pending_entry.estimate)
            if allocation is None:
                continue

            # 予約リストから取り除く
            self._pending_entries.pop(index)
            context = self._build_context(pending_entry.case)
            inherited_env_vars = context.get("environment_variables", {})
            if not isinstance(inherited_env_vars, dict):
                raise TypeError("context['environment_variables'] must be a dict.")
            env_vars = {
                str(key): str(value) for key, value in inherited_env_vars.items()
            }
            if allocation.gpu_ids:
                gpu_ids_text = ",".join(str(gpu_id) for gpu_id in allocation.gpu_ids)
                allocation_env_vars = build_gpu_env(
                    visible_devices=gpu_ids_text,
                    disable_preallocation=False,
                )
                allocation_env_vars["gpu_ids"] = gpu_ids_text
                if len(allocation.gpu_ids) == 1:
                    allocation_env_vars["gpu_id"] = str(allocation.gpu_ids[0])
                if self._gpu_environment_config is not None:
                    env_vars = merge_env_vars(
                        env_vars,
                        allocation_env_vars,
                        self._gpu_environment_config.build_environment_variables(),
                    )
                else:
                    env_vars = merge_env_vars(env_vars, allocation_env_vars)

            worker_label = self._worker_label_for_allocation(allocation)
            env_vars["EXPERIMENT_RUNNER_WORKER_LABEL"] = worker_label
            env_vars["EXPERIMENT_RUNNER_CPU_AFFINITY"] = ",".join(
                str(cpu) for cpu in allocation.cpu_affinity
            )
            if allocation.gpu_ids:
                env_vars["EXPERIMENT_RUNNER_ASSIGNED_GPU_IDS"] = ",".join(
                    str(gpu_id) for gpu_id in allocation.gpu_ids
                )
            gpu_slot = self._single_gpu_slot(allocation)
            if gpu_slot is not None:
                env_vars["EXPERIMENT_RUNNER_GPU_SLOT"] = str(gpu_slot)
            context["environment_variables"] = env_vars
            context["runner_metadata"] = {
                "worker_label": worker_label,
                "worker_slot_id": allocation.worker_slot_id,
                "cpu_affinity": list(allocation.cpu_affinity),
                "gpu_ids": list(allocation.gpu_ids),
                "gpu_slot": gpu_slot,
            }

            # コンテキストをキーに割当を記録しておく（on_finish で復元する）
            self._active_allocations[id(context)] = allocation
            return pending_entry.case, context
        return None

    def on_finish(
        self,
        case: T,
        context: TaskContext,
        result: ExecutionResult,
    ) -> None:
        self.completions.append(
            StandardCompletion(
                case=case,
                context=dict(context),
                result=result,
            )
        )
        if self.skip_controller is not None:
            self.skip_controller.update(case, context, result)

        # 実行完了後はスケジューラ側で割り当てたリソースを解放する。
        allocation = self._active_allocations.pop(id(context), None)
        if allocation is None:
            raise ValueError("context is not associated with an active allocation.")

        # ワーカースロット、ホストメモリ、GPU スロット/メモリを戻す
        self._available_worker_slots += 1
        self._available_worker_slot_ids.append(allocation.worker_slot_id)
        self._available_host_memory_bytes += allocation.host_memory_bytes
        for gpu_id, slot_ids in allocation.gpu_slot_assignments:
            self._available_gpu_slots[gpu_id] += allocation.gpu_slots
            self._available_gpu_memory_bytes[gpu_id] += allocation.gpu_memory_bytes
            self._available_gpu_slot_ids[gpu_id].extend(slot_ids)
            self._available_gpu_slot_ids[gpu_id].sort()

    def is_completed(self) -> bool:
        return not self._pending_entries
