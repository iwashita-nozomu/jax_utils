# -*- coding: utf-8 -*-
"""
スケジューラのリソース追跡デバッグ

StandardFullResourceScheduler が gpu_memory_bytes を正しく減算しているか、
ステップバイステップで検証するテスト。
"""

from jax_util.experiment_runner import (
    FullResourceCapacity,
    FullResourceEstimate,
    StandardFullResourceScheduler,
    TaskContext,
)
from jax_util.experiment_runner.protocols import Worker
from typing import Any


class DebugWorker(Worker[dict[str, Any], int]):
    """デバッグ用ワーカー"""
    
    def __call__(self, case: dict[str, Any], context: TaskContext) -> int:
        return 0
    
    def resource_estimate(self, case: dict[str, Any]) -> FullResourceEstimate:
        """各タスク：5GB GPU メモリ要求、ホストメモリ最小限"""
        return FullResourceEstimate(
            host_memory_bytes=int(0.1 * 1024 * 1024 * 1024),  # 100MB ホストメモリ（最小限）
            gpu_count=1,
            gpu_memory_bytes=int(5.0 * 1024 * 1024 * 1024),  # 5GB GPU メモリ（48GB÷5 ≈ 9タスク限界）
            gpu_slots=1,
        )


def debug_scheduler_memory_tracking() -> None:
    """スケジューラのメモリ追跡をデバッグ"""
    print("=" * 70)
    print("Scheduler Memory Tracking Debug (50 tasks × 5GB)")
    print("=" * 70)
    
    # 50 個のタスク（メモリリソースを超過）
    cases = [{"task_id": f"task_{i:02d}"} for i in range(50)]
    
    def context_builder(case: dict[str, Any]) -> TaskContext:
        return {"task_id": case["task_id"]}
    
    # スケジューラ作成（max_slots=50、max_workers=60で CPU 制約を外す）
    resource_capacity = FullResourceCapacity.from_system(
        gpu_max_slots=50,
        max_workers=60  # CPU ワーカースロット制約を超過して、GPU メモリ制約を検証
    )
    
    scheduler = StandardFullResourceScheduler.from_worker(
        cases=cases,
        worker=DebugWorker(),
        context_builder=context_builder,
        resource_capacity=resource_capacity,
    )
    
    # 初期状態表示
    capacity = scheduler.resource_capacity
    print(f"\nInitial Resource Capacity:")
    print(f"  Host Memory: {capacity.host_memory_bytes / (1024**3):.2f}GB")
    print(f"  GPU Devices: {len(capacity.gpu_devices)}")
    for gpu_dev in capacity.gpu_devices:
        print(f"    GPU {gpu_dev.gpu_id}: {gpu_dev.memory_bytes / (1024**3):.2f}GB")
    
    # スケジューラ内の利用可能メモリを直接確認
    print(f"\nScheduler's _available_gpu_memory_bytes (before allocation):")
    for gpu_id, mem_bytes in scheduler._available_gpu_memory_bytes.items():
        print(f"  GPU {gpu_id}: {mem_bytes / (1024**3):.2f}GB")
    
    print(f"\nScheduler's _available_gpu_slots (before allocation):")
    for gpu_id, slots in scheduler._available_gpu_slots.items():
        print(f"  GPU {gpu_id}: {slots} slots")
    
    # タスク割り当てをシミュレート
    print(f"\n" + "=" * 70)
    print("Allocating tasks one by one:")
    print("=" * 70)
    
    allocated_count = 0
    failed_count = 0
    
    for i in range(50):
        result = scheduler.next_case()
        if result is None:
            print(f"\nNo more cases available (allocated {allocated_count}, failed {failed_count})")
            break
        case, context = result
        
        allocated_count += 1
        task_id = case["task_id"]
        env_vars = context.get("environment_variables", {})
        gpu_id = env_vars.get("gpu_id", "?")
        
        # ログ出力は最初の 5 個と最後の 5 個のみ
        if i < 5 or i >= 45:
            print(f"[{allocated_count}] {task_id} allocated to GPU {gpu_id}")
            # 割り当て直後のメモリ状態を表示
            mem_str = ", ".join([f"GPU{gid}: {mem_bytes/(1024**3):.2f}GB" 
                                for gid, mem_bytes in scheduler._available_gpu_memory_bytes.items()])
            print(f"    {mem_str}")
        elif i == 5:
            print(f"    ... (tasks 5-44 omitted) ...")
    
    print(f"\n" + "=" * 70)
    print(f"Final Status:")
    print(f"  Allocated: {allocated_count} tasks")
    print(f"  Failed: {failed_count} tasks")
    print(f"  Total: {len(cases)} tasks")
    print(f"  Remaining GPU memory:")
    for gpu_id, mem_bytes in scheduler._available_gpu_memory_bytes.items():
        print(f"    GPU {gpu_id}: {mem_bytes / (1024**3):.2f}GB")
    
    print(f"\n" + "=" * 70)
    print("Analysis:")
    print(f"  GPU capacity: 3 × 16GB = 48GB")
    print(f"  Task demand: 50 × 5GB = 250GB")
    print(f"  Expected max: 48GB ÷ 5GB = 9-10 tasks")
    print(f"  Actual allocated: {allocated_count} tasks")
    
    if allocated_count <= 10:
        print(f"  ✓ CORRECT: GPU memory constraint enforced (≤10 tasks)")
    else:
        print(f"  ✗ ERROR: GPU memory constraint violated!")
        print(f"     Allocated {allocated_count} tasks > 9-10 expected max")
    print("=" * 70)


if __name__ == "__main__":
    debug_scheduler_memory_tracking()
