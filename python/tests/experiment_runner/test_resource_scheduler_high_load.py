from __future__ import annotations

from dataclasses import dataclass
import json
import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping, cast

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from jax_util.experiment_runner.resource_scheduler import (
    FullResourceCapacity,
    FullResourceEstimate,
    GPUDeviceCapacity,
    StandardFullResourceScheduler,
)
from jax_util.experiment_runner.runner import (
    StandardRunner,
    StandardWorker,
)
from jax_util.experiment_runner.protocols import TaskContext


def _case_id(case: Mapping[str, object], /) -> int:
    value = case.get("case_id")
    if not isinstance(value, int):
        raise TypeError("case_id must be int.")
    return value


def _sleep_seconds(case: Mapping[str, object], /) -> float:
    value = case.get("sleep_seconds")
    if not isinstance(value, (int, float)):
        raise TypeError("sleep_seconds must be numeric.")
    return float(value)


def _resource_estimate(case: Mapping[str, object], /) -> FullResourceEstimate:
    host_memory_bytes = case.get("host_memory_bytes", 0)
    gpu_count = case.get("gpu_count", 0)
    gpu_memory_bytes = case.get("gpu_memory_bytes", 0)
    if not isinstance(host_memory_bytes, int):
        raise TypeError("host_memory_bytes must be int.")
    if not isinstance(gpu_count, int):
        raise TypeError("gpu_count must be int.")
    if not isinstance(gpu_memory_bytes, int):
        raise TypeError("gpu_memory_bytes must be int.")
    return FullResourceEstimate(
        host_memory_bytes=host_memory_bytes,
        gpu_count=gpu_count,
        gpu_memory_bytes=gpu_memory_bytes,
    )


def _write_json(path: Path, record: Mapping[str, object], /) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=True, sort_keys=True), encoding="utf-8")


@dataclass(frozen=True)
class FullResourceRecordingTask:
    records_dir: Path

    def __call__(self, case: dict[str, object], context: TaskContext) -> dict[str, object]:
        case_id = _case_id(case)
        started_at = time.time()
        time.sleep(_sleep_seconds(case))
        finished_at = time.time()

        record: dict[str, object] = {
            "case_id": case_id,
            "pid": os.getpid(),
            "gpu_ids": context.get("gpu_ids", ""),
            "cuda_visible_devices": context.get("CUDA_VISIBLE_DEVICES", ""),
            "started_at": started_at,
            "finished_at": finished_at,
        }
        _write_json(self.records_dir / f"case_{case_id}.json", record)
        return record


@dataclass(frozen=True)
class EstimatedCaseTask:
    def __call__(self, case: dict[str, object], context: TaskContext) -> dict[str, object]:
        return {"case_id": _case_id(case), "context": dict(context)}

    def resource_estimate(self, case: Mapping[str, object], /) -> FullResourceEstimate:
        return _resource_estimate(case)


def test_high_load_resource_scheduler(tmp_path: Path) -> None:
    # 大量の短時間ケースを生成しスケジューラに負荷をかける
    count = 200
    cases: list[dict[str, object]] = []
    for i in range(count):
        # ほとんどが短い sleep、ランダム性は不要で単純に負荷を作る
        has_gpu = (i % 3) == 0
        cases.append({
            "case_id": i,
            "sleep_seconds": 0.005,
            "host_memory_bytes": 1,
            "gpu_count": 1 if has_gpu else 0,
            "gpu_memory_bytes": 1 if has_gpu else 0,
        })

    scheduler = StandardFullResourceScheduler(
        resource_capacity=FullResourceCapacity(
            max_workers=8,
            host_memory_bytes=16,
            gpu_devices=(
                GPUDeviceCapacity(gpu_id=0, memory_bytes=8, max_slots=1),
                GPUDeviceCapacity(gpu_id=1, memory_bytes=8, max_slots=1),
            ),
        ),
        cases=cases,
        estimate_builder=_resource_estimate,
    )

    runner = StandardRunner(scheduler)
    with TemporaryDirectory() as tmpdir:
        records_dir = Path(tmpdir) / "records"
        worker = StandardWorker(FullResourceRecordingTask(records_dir))

        start = time.perf_counter()
        runner.run(worker)
        elapsed = time.perf_counter() - start

        # 全ケースが完了していること
        assert len(scheduler.completions) == count
        # 実行時間は過度に長くないこと（マシンやCIで差があるため緩めに設定）
        assert elapsed < 10.0


def _run_all_tests() -> None:
    # 単体実行用ラッパー: テストを実行して 1 行 JSON サマリを出力する。
    count = 200
    with TemporaryDirectory() as tmpdir:
        records_dir = Path(tmpdir) / "records"
        try:
            cases: list[dict[str, object]] = []
            for i in range(count):
                has_gpu = (i % 3) == 0
                cases.append({
                    "case_id": i,
                    "sleep_seconds": 0.005,
                    "host_memory_bytes": 1,
                    "gpu_count": 1 if has_gpu else 0,
                    "gpu_memory_bytes": 1 if has_gpu else 0,
                })

            scheduler = StandardFullResourceScheduler(
                resource_capacity=FullResourceCapacity(
                    max_workers=8,
                    host_memory_bytes=16,
                    gpu_devices=(
                        GPUDeviceCapacity(gpu_id=0, memory_bytes=8, max_slots=1),
                        GPUDeviceCapacity(gpu_id=1, memory_bytes=8, max_slots=1),
                    ),
                ),
                cases=cases,
                estimate_builder=_resource_estimate,
            )

            runner = StandardRunner(scheduler)
            worker = StandardWorker(FullResourceRecordingTask(records_dir))

            start = time.perf_counter()
            runner.run(worker)
            elapsed = time.perf_counter() - start

            result = {
                "case": "high_load_resource_scheduler",
                "expected": {"count": count, "elapsed_lt_seconds": 10.0},
                "actual": {"count": len(scheduler.completions), "elapsed_seconds": elapsed},
            }
            print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        except Exception as exc:  # pragma: no cover - manual runs
            print(json.dumps({"case": "high_load_resource_scheduler", "error": str(exc)}))
            raise


if __name__ == "__main__":
    _run_all_tests()
