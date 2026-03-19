from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping, cast

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(PYTHON_ROOT))

import pytest

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


if os.environ.get("RUN_HEAVY_TESTS") not in {"1", "true", "True"}:
    pytest.skip("heavy tests disabled; set RUN_HEAVY_TESTS=1 to enable", allow_module_level=True)


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


@dataclass(frozen=True)
class FullResourceRecordingTask:
    records_dir: Path

    def __call__(self, case: dict[str, object], context: TaskContext) -> dict[str, object]:
        case_id = _case_id(case)
        started_at = time.time()
        time.sleep(_sleep_seconds(case))
        finished_at = time.time()

        return {"case_id": case_id, "started_at": started_at, "finished_at": finished_at, "gpu_ids": context.get("gpu_ids", "")}


def test_heavy_resource_scheduler(tmp_path: Path) -> None:
    # 非常に多数のケースでスケジューラ/ランナーを負荷検証する
    count = 2000
    cases: list[dict[str, object]] = []
    for i in range(count):
        has_gpu = (i % 4) == 0
        cases.append({
            "case_id": i,
            "sleep_seconds": 0.001,
            "host_memory_bytes": 1,
            "gpu_count": 1 if has_gpu else 0,
            "gpu_memory_bytes": 1 if has_gpu else 0,
        })

    scheduler = StandardFullResourceScheduler(
        resource_capacity=FullResourceCapacity(
            max_workers=16,
            host_memory_bytes=64,
            gpu_devices=(
                GPUDeviceCapacity(gpu_id=0, memory_bytes=8, max_slots=1),
                GPUDeviceCapacity(gpu_id=1, memory_bytes=8, max_slots=1),
                GPUDeviceCapacity(gpu_id=2, memory_bytes=8, max_slots=1),
            ),
        ),
        cases=cases,
        estimate_builder=_resource_estimate,
    )

    runner = StandardRunner(scheduler)
    worker = StandardWorker(FullResourceRecordingTask(tmp_path))

    start = time.perf_counter()
    runner.run(worker)
    elapsed = time.perf_counter() - start

    assert len(scheduler.completions) == count
    assert elapsed < 120.0


def _run_all_tests() -> None:
    # 単体実行用のラッパー: テストを実行して結果を 1 行 JSON で出力する。
    count = 2000
    with TemporaryDirectory() as tmp_dir:
        try:
            # 再利用しやすくするため、テスト関数と同様のロジックをここでも実行する
            cases: list[dict[str, object]] = []
            for i in range(count):
                has_gpu = (i % 4) == 0
                cases.append({
                    "case_id": i,
                    "sleep_seconds": 0.001,
                    "host_memory_bytes": 1,
                    "gpu_count": 1 if has_gpu else 0,
                    "gpu_memory_bytes": 1 if has_gpu else 0,
                })

            scheduler = StandardFullResourceScheduler(
                resource_capacity=FullResourceCapacity(
                    max_workers=16,
                    host_memory_bytes=64,
                    gpu_devices=(
                        GPUDeviceCapacity(gpu_id=0, memory_bytes=8, max_slots=1),
                        GPUDeviceCapacity(gpu_id=1, memory_bytes=8, max_slots=1),
                        GPUDeviceCapacity(gpu_id=2, memory_bytes=8, max_slots=1),
                    ),
                ),
                cases=cases,
                estimate_builder=_resource_estimate,
            )

            runner = StandardRunner(scheduler)
            worker = StandardWorker(FullResourceRecordingTask(Path(tmp_dir)))

            start = time.perf_counter()
            runner.run(worker)
            elapsed = time.perf_counter() - start

            result = {
                "case": "heavy_resource_scheduler",
                "expected": {"count": count, "elapsed_lt_seconds": 120.0},
                "actual": {"count": len(scheduler.completions), "elapsed_seconds": elapsed},
            }
            print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        except Exception as exc:  # pragma: no cover - for manual runs
            print(json.dumps({"case": "heavy_resource_scheduler", "error": str(exc)}))
            raise


if __name__ == "__main__":
    _run_all_tests()
