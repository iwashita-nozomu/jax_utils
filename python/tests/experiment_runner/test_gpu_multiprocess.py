# -*- coding: utf-8 -*-
"""
GPU メモリ共有テスト（小メモリ・多並列版）

max_slots=10 に設定して、小さいメモリのプロセスを多数同時実行するテスト。

テスト設定：
- 20 タスク × 100MB = 2GB 総メモリ需要
- 3 GPU × 16GB = 48GB 搭載
- max_slots=10 により、最大 10 プロセスが同じ GPU で実行可能
- 期待値：20 タスク / 3 GPU ≈ 7 タスク同時実行 × 3 秒 ≈ 9 秒
"""

from __future__ import annotations

import time
import sys
from typing import Any
import psutil

# JAX メモリ先取り無効化（GPU デバイスも含めて無効化）
from jax_util.experiment_runner import disable_jax_memory_preallocation
disable_jax_memory_preallocation(gpu_devices=True)

from jax_util.experiment_runner import (
    FullResourceCapacity,
    FullResourceEstimate,
    StandardFullResourceScheduler,
    StandardRunner,
    TaskContext,
    SUCCESS_EXIT_CODE,
    WORKER_PROTOCOL_ERROR_EXIT_CODE,
    apply_environment_variables,
)
from jax_util.experiment_runner.protocols import Worker


class GPUMultiProcessWorker(Worker[dict[str, Any], int]):
    """複数プロセス同時実行テスト用ワーカー
    
    複数タスクが同じ GPU 上で並列実行されることを確認。
    JAX はプリアロケなしなので、実際の使用量に応じてメモリを割り当てる。
    """

    def __call__(self, case: dict[str, Any], context: TaskContext) -> int:
        """ケースを実行"""
        try:
            import jax.numpy as jnp
            import os
            
            task_id = case["task_id"]
            
            # environment_variables を os.environ に適用
            apply_environment_variables(context)
            
            env_vars = context.get("environment_variables", {})
            gpu_id = env_vars.get("gpu_id", "unknown")
            
            proc = psutil.Process()
            process_name = f"[{task_id}]"
            
            print(f"\n{process_name} Starting on GPU {gpu_id}...")
            sys.stdout.flush()
            
            # 1GB 行列を GPU にアロケート
            print(f"{process_name} Allocating 1GB on GPU...")
            sys.stdout.flush()
            
            matrix_size = 16384
            large_matrix = jnp.ones((matrix_size, matrix_size), dtype=jnp.float32)
            
            # メモリが実際に割り当てられるようにアクセス
            _ = large_matrix[0, 0]
            mem_allocated_gb = large_matrix.nbytes / (1024**3)
            
            # ホストメモリ使用量
            host_mem_mb = proc.memory_info().rss / (1024**2)
            
            print(f"{process_name} GPU mem allocated: {mem_allocated_gb:.2f}GB, Host mem: {host_mem_mb:.1f}MB")
            print(f"{process_name} Waiting 5 seconds...")
            sys.stdout.flush()
            
            # 5 秒待機（その間に他のタスクが同じ GPU で実行される）
            time.sleep(5)
            
            # クリーンアップ
            del large_matrix
            print(f"{process_name} Completed")
            sys.stdout.flush()
            
            return SUCCESS_EXIT_CODE
        except Exception as e:
            print(f"[{case['task_id']}] ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            return WORKER_PROTOCOL_ERROR_EXIT_CODE

    def resource_estimate(self, case: dict[str, Any]) -> FullResourceEstimate:
        """このタスクのメモリ見積もり"""
        return FullResourceEstimate(
            host_memory_bytes=int(1.2 * 1024 * 1024 * 1024),  # 1.2GB ホスト
            gpu_count=1,                                        # 1 GPU
            gpu_memory_bytes=int(1.0 * 1024 * 1024 * 1024),   # 1GB GPU
            gpu_slots=1,                                        # 1 スロット（複数プロセスが共有可能）
        )


def test_gpu_multi_process() -> None:
    """複数プロセス同時実行テスト"""
    print("=" * 70)
    print("GPU Multi-Process Test (複数タスク同時実行)")
    print("=" * 70)
    print("\nSetup: 10 tasks × 1GB = 10GB total")
    print("GPU mode: max_slots=4 (複数プロセスが同じ GPU で共有可能)")
    print("Strategy: 貪欲割り当て（メモリに余裕があれば複数プロセスを同じ GPU に割り当て）")
    print("\nExpected behavior:")
    print("  - Multiple tasks run in parallel on same GPU")
    print("  - Memory efficiently shared across processes")
    print("  - Faster total execution time (parallel instead of sequential)\n")
    
    # テストケース生成
    cases = [
        {"task_id": f"task_{i:02d}"} 
        for i in range(10)
    ]
    
    # コンテキストビルダー
    def context_builder(case: dict[str, Any]) -> TaskContext:
        return {"task_id": case["task_id"]}
    
    # ★重要★: max_slots=4 に設定して、複数プロセスを同じ GPU で実行可能にする
    # resource_capacity を明示的に作成して max_slots を指定
    resource_capacity = FullResourceCapacity.from_system(gpu_max_slots=4)
    
    # スケジューラ作成
    scheduler = StandardFullResourceScheduler.from_worker(
        cases=cases,
        worker=GPUMultiProcessWorker(),
        context_builder=context_builder,
        resource_capacity=resource_capacity,
    )
    
    # リソース容量を表示
    capacity = scheduler.resource_capacity
    print("=== Detected Resource Capacity ===")
    print(f"Host Memory: {capacity.host_memory_bytes / (1024**3):.2f}GB")
    print(f"GPU Devices: {len(capacity.gpu_devices)}")
    for gpu_dev in capacity.gpu_devices:
        print(f"  GPU {gpu_dev.gpu_id}: {gpu_dev.memory_bytes / (1024**3):.2f}GB, max_slots={gpu_dev.max_slots}")
    
    print(f"\n=== Expected Allocation ===")
    print(f"With max_slots={capacity.gpu_devices[0].max_slots}:")
    print(f"  - Multiple tasks can run simultaneously on the same GPU")
    print(f"  - Greedy allocation: pack as many tasks as possible per GPU")
    print(f"  - Total throughput should be ~(10 tasks × 5s) / 3 parallel ≈ ~17 seconds")
    
    # プログレス表示
    def progress(completed: int, total: int, elapsed: float, running: int) -> None:
        print(f"\r[Progress] Completed: {completed}/{total} | Running: {running} | Elapsed: {elapsed:.1f}s", end="", flush=True)
    
    # ランナー実行
    runner = StandardRunner(scheduler, progress_callback=progress)
    print(f"\nRunning {len(cases)} tasks (1GB each, max_slots=4)...\n")
    print("During execution in another terminal: watch -n 1 nvidia-smi\n")
    
    t0 = time.time()
    runner.run(GPUMultiProcessWorker())
    elapsed = time.time() - t0
    
    # 結果統計
    print(f"\n\n" + "=" * 70)
    print(f"Completed in {elapsed:.1f}s")
    print(f"Throughput: {len(cases) / elapsed:.2f} tasks/s")
    print(f"Expected: ~0.59 tasks/s (10 tasks / 17 seconds theoretical)")
    
    print(f"\nTotal GPU memory requested: {len(cases) * 1.0:.1f}GB")
    print(f"Total GPU memory available: {sum(gpu.memory_bytes for gpu in capacity.gpu_devices) / (1024**3):.1f}GB")
    
    print(f"\nScheduler allocated tasks using max_slots={capacity.gpu_devices[0].max_slots}")
    print(f"Check nvidia-smi output to verify actual GPU utilization")
    print("Expected: Multiple process running on same GPU simultaneously")
    print("=" * 70)


if __name__ == "__main__":
    test_gpu_multi_process()
