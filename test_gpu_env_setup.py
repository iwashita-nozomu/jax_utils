#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
environment_variables の dict セットアップをテスト

スケジューラから返される context に environment_variables が
正しく格納されているか、ワーカーが正しく読み取れるかを確認。
"""

import os
import sys
from pathlib import Path

# ワークスペース構成
WORKSPACE_ROOT = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "python"))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from jax_util.experiment_runner import (
    FullResourceCapacity,
    StandardFullResourceScheduler,
    TaskContext,
)
from jax_util.experiment_runner.protocols import Worker, ResourceEstimatingWorker
from jax_util.experiment_runner import FullResourceEstimate
from typing import Any


class TestWorker(Worker[dict[str, Any], int]):
    """environment_variables を検査するテストワーカー"""

    def __call__(self, case: dict[str, Any], context: TaskContext) -> int:
        """環境変数を表示"""
        print(f"\n=== Case: {case['case_id']} ===")
        print(f"TaskContext keys: {list(context.keys())}")
        
        # environment_variables dict を読み取り
        env_vars = context.get("environment_variables", {})
        print(f"environment_variables: {env_vars}")
        
        # 実際に os.environ に設定
        for key, value in env_vars.items():
            os.environ[key] = str(value)
            print(f"  set os.environ[{key!r}] = {value!r}")
        
        # 確認
        if "CUDA_VISIBLE_DEVICES" in env_vars:
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET")
            print(f"  verified: CUDA_VISIBLE_DEVICES = {cuda_visible!r}")
        
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
    print("Test: environment_variables dict setup")
    print("=" * 70)

    # 簡単なケースリスト
    cases = [
        {"case_id": "case_1", "index": 0},
        {"case_id": "case_2", "index": 1},
    ]

    # リソース容量（GPU 2個）
    resource_capacity = FullResourceCapacity.from_system(
        max_workers=2,
        host_memory_bytes=16 * 1024 * 1024 * 1024,
        gpu_devices=None,
        gpu_max_slots=1,
    )

    print(f"\nResource capacity: GPU={len(resource_capacity.gpu_devices)} devices")
    for device in resource_capacity.gpu_devices:
        print(f"  GPU {device.gpu_id}: {device.gpu_name}")

    # コンテキストビルダー
    def context_builder(case: dict[str, Any]) -> TaskContext:
        return {"case_id": case["case_id"]}

    # スケジューラ生成
    scheduler = StandardFullResourceScheduler.from_worker(
        resource_capacity=resource_capacity,
        cases=cases,
        worker=TestWorker(),
        context_builder=context_builder,
    )

    # ケースを取得して環境変数を確認
    print("\n--- Scheduler.next_case() output ---")
    for i in range(len(cases)):
        job = scheduler.next_case()
        if job is None:
            print(f"Case {i}: NONE")
            break

        case, context = job
        print(f"\nCase {i}: {case['case_id']}")
        print(f"  context keys: {list(context.keys())}")
        print(f"  environment_variables: {context.get('environment_variables', {})}")

        # ワーカーで処理
        worker = TestWorker()
        exit_code = worker(case, context)
        scheduler.on_finish(case, context, exit_code)


if __name__ == "__main__":
    main()
