#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ランナーの spawn 方式 と GPU リソース管理をテストする。

StandardRunner が spawn コンテキストで動作し、
リソーススケジューラが GPU を適切に検出・管理していることを確認する。

Test cases:
  - Worker pickle 可能性確認
  - Spawn コンテキスト動作確認
  - GPU 検出と割当
  - リソース監視（メモリ、ワーカースロット、GPU）
"""

import sys
from pathlib import Path

# ワークスペース構成を解決
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
WORKSPACE_ROOT = SCRIPT_DIR
if str(WORKSPACE_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "python"))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import time
from typing import Any

from jax_util.experiment_runner import (
    FullResourceCapacity,
    FullResourceEstimate,
    StandardFullResourceScheduler,
    StandardRunner,
    TaskContext,
    SUCCESS_EXIT_CODE,
)
from jax_util.experiment_runner.jax_context import check_picklable, get_spawn_context
from jax_util.experiment_runner.protocols import Worker


# テスト用ワーカー：簡単なカウンタータスク
class SimpleCounterWorker(Worker[dict[str, Any], int]):
    """テスト用ワーカー：case_id とワーカー ID をログに記録して実行"""

    def __init__(self, task_duration: float = 0.1) -> None:
        """
        Parameters
        ----------
        task_duration : float
            各タスクの実行時間（秒）
        """
        self.task_duration = task_duration

    def __call__(self, case: dict[str, Any], context: TaskContext) -> int:
        """ケースを実行し、指定時間スリープ"""
        try:
            case_id = case.get("case_id", "unknown")
            worker_id = context.get("worker_id", "?")

            # GPU デバイスが割り当てられているか確認
            gpu_ids = context.get("gpu_ids", None)
            cuda_visible = context.get("CUDA_VISIBLE_DEVICES", "none")

            # 簡単なログ出力
            time.sleep(self.task_duration)

            # 結果をコンテキストに保存
            context["completed_at"] = time.time()
            context["gpu_assigned"] = gpu_ids is not None
            context["cuda_visible_devices"] = cuda_visible

            return SUCCESS_EXIT_CODE
        except Exception as e:
            context["error"] = str(e)
            return 1

    def resource_estimate(self, case: dict[str, Any]) -> FullResourceEstimate:
        """ケースのリソース見積もり（固定値）"""
        # 軽量なテストケース：ホストメモリ 256MB、スロット1
        return FullResourceEstimate(
            host_memory_bytes=256 * 1024 * 1024,  # 256 MB
            gpu_count=0,  # GPU 不使用
            gpu_memory_bytes=0,
            gpu_slots=1,
        )


def test_worker_picklability() -> None:
    """テスト1: Worker が pickle 化可能であることを確認"""
    print("\n" + "=" * 70)
    print("Test 1: Worker Picklability Check")
    print("=" * 70)

    worker = SimpleCounterWorker()
    try:
        check_picklable(worker, name="SimpleCounterWorker")
        print("✓ Worker is picklable (can be sent to worker processes)")
    except ValueError as e:
        print(f"✗ Worker is NOT picklable: {e}")
        raise


def test_spawn_context() -> None:
    """テスト2: Spawn コンテキストが取得できることを確認"""
    print("\n" + "=" * 70)
    print("Test 2: Spawn Context Availability")
    print("=" * 70)

    try:
        ctx = get_spawn_context()
        print(f"✓ Spawn context obtained: {ctx}")
    except Exception as e:
        print(f"✗ Failed to get spawn context: {e}")
        raise


def test_resource_aware_scheduling() -> None:
    """テスト3: リソース認識スケジューリング with プログレス監視"""
    print("\n" + "=" * 70)
    print("Test 3: Resource-Aware Scheduling with Progress Monitoring")
    print("=" * 70)

    # テストケース生成（10ケース）
    test_cases = [
        {
            "case_id": f"test_case_{i:03d}",
            "dimension": i + 1,
            "value": i * 0.1,
        }
        for i in range(10)
    ]

    # リソース容量設定
    resource_capacity = FullResourceCapacity(
        max_workers=4,
        host_memory_bytes=2 * 1024 * 1024 * 1024,  # 2 GB
        gpu_devices=(),  # GPU なし（テスト用）
    )

    # コンテキストビルダー
    def context_builder(case: dict[str, Any]) -> TaskContext:
        return {
            "case_id": case["case_id"],
            "worker_id": case["case_id"],
        }

    # プログレス監視用コールバック
    last_report_time = [0.0]  # クロージャで時刻を記録

    def progress_callback(
        completed: int, total: int, elapsed: float, running: int
    ) -> None:
        # 1秒ごとにレポート
        if elapsed - last_report_time[0] >= 1.0:
            if total > 0:
                pct = (completed / total) * 100
                throughput = completed / elapsed if elapsed > 0 else 0
                print(
                    f"  Progress: [{completed:2d}/{total:2d}] {pct:5.1f}% "
                    f"| Throughput: {throughput:5.2f} cases/s | Running: {running}"
                )
            last_report_time[0] = elapsed

    # ワーカー
    worker = SimpleCounterWorker(task_duration=0.2)

    # スケジューラ
    scheduler = StandardFullResourceScheduler.from_worker(
        resource_capacity=resource_capacity,
        cases=test_cases,
        worker=worker,
        context_builder=context_builder,
    )

    # ランナー（プログレスコールバック付き）
    runner = StandardRunner(scheduler, progress_callback=progress_callback)

    print(f"Running {len(test_cases)} test cases...")
    print(f"Resource capacity: {resource_capacity.max_workers} workers, "
          f"{resource_capacity.host_memory_bytes / (1024**3):.1f} GB host memory")

    start_time = time.time()
    runner.run(worker)
    elapsed = time.time() - start_time

    # 結果集計
    completions = scheduler.completions
    success_count = sum(1 for c in completions if c.exit_code == 0)
    failed_count = len(completions) - success_count

    print(f"\n✓ Execution completed")
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Cases: {len(completions)} total, {success_count} success, {failed_count} failed")
    print(f"  Throughput: {len(completions) / elapsed:.2f} cases/s")

    assert success_count == len(test_cases), f"Expected all cases to succeed"


def test_gpu_detection() -> None:
    """テスト4: GPU 自動検出"""
    print("\n" + "=" * 70)
    print("Test 4: GPU Auto-Detection")
    print("=" * 70)

    from jax_util.experiment_runner import detect_gpu_devices

    gpu_devices = detect_gpu_devices()
    if gpu_devices:
        print(f"✓ GPU devices detected: {len(gpu_devices)}")
        for dev in gpu_devices:
            print(
                f"  GPU {dev.gpu_id}: "
                f"{dev.memory_bytes / (1024**3):.1f} GB, "
                f"max_slots={dev.max_slots}"
            )
    else:
        print("✓ No GPU devices available")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("StandardRunner Spawn & Resource Management Tests")
    print("=" * 70)

    try:
        test_worker_picklability()
        test_spawn_context()
        test_gpu_detection()
        test_resource_aware_scheduling()

        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
