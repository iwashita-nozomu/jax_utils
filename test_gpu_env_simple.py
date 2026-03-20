#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
environment_variables dict のみテスト
スケジューラの GPU 検出部分をスキップ
"""

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "python"))

from jax_util.experiment_runner import (
    FullResourceCapacity,
    GPUDeviceCapacity,
    StandardFullResourceScheduler,
    TaskContext,
)
from jax_util.experiment_runner import FullResourceEstimate
from jax_util.experiment_runner.protocols import Worker, ResourceEstimatingWorker
from typing import Any


class TestWorker(Worker[dict[str, Any], int]):
    """環境変数をチェック"""

    def __call__(self, case: dict[str, Any], context: TaskContext) -> int:
        print(f"\n  Case: {case['case_id']}")
        print(f"  environment_variables in context: {context.get('environment_variables', {})}")
        return 0

    def resource_estimate(self, case: dict[str, Any]) -> FullResourceEstimate:
        return FullResourceEstimate(
            host_memory_bytes=100 * 1024 * 1024,
            gpu_count=1,
            gpu_memory_bytes=500 * 1024 * 1024,
            gpu_slots=1,
        )


def main():
    """テスト実行"""
    print("=" * 70)
    print("Test: environment_variables dict setup (manual GPU setup)")
    print("=" * 70)

    # ケース
    cases = [
        {"case_id": "case_1", "index": 0},
        {"case_id": "case_2", "index": 1},
    ]

    # GPU デバイスを手動指定（from_system() をスキップ）
    gpu_devices = [
        GPUDeviceCapacity(gpu_id=0, memory_bytes=16 * 1024 * 1024 * 1024),
        GPUDeviceCapacity(gpu_id=1, memory_bytes=16 * 1024 * 1024 * 1024),
        GPUDeviceCapacity(gpu_id=2, memory_bytes=16 * 1024 * 1024 * 1024),
    ]

    resource_capacity = FullResourceCapacity(
        max_workers=2,
        host_memory_bytes=16 * 1024 * 1024 * 1024,
        gpu_devices=tuple(gpu_devices),
    )

    print(f"\nGPU Devices: {len(resource_capacity.gpu_devices)}")
    for dev in gpu_devices:
        print(f"  GPU {dev.gpu_id}: {dev.memory_bytes / (1024**3):.1f} GB")

    # コンテキストビルダー
    def context_builder(case: dict[str, Any]) -> TaskContext:
        return {"case_id": case["case_id"]}

    # スケジューラ生成
    print("\nCreating scheduler...")
    scheduler = StandardFullResourceScheduler.from_worker(
        resource_capacity=resource_capacity,
        cases=cases,
        worker=TestWorker(),
        context_builder=context_builder,
    )

    print("Scheduler created successfully")

    # ケースを1つ取得
    print("\nCalling next_case()...")
    job = scheduler.next_case()
    if job:
        case, context = job
        print(f"Got case: {case['case_id']}")
        print(f"Context keys: {list(context.keys())}")
        if "environment_variables" in context:
            print(f"✓ environment_variables found: {context['environment_variables']}")
        else:
            print("✗ environment_variables NOT found")

    print("\nTest complete!")


if __name__ == "__main__":
    main()
