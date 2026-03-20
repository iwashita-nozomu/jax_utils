# -*- coding: utf-8 -*-
"""
GPU メモリ検出の検証

detect_gpu_devices() が正しくメモリを報告しているか確認するテスト。
"""

from jax_util.experiment_runner import detect_gpu_devices, FullResourceCapacity

# GPU メモリを検出
capacity = FullResourceCapacity.from_system(gpu_max_slots=50)

print("=" * 70)
print("GPU Memory Detection Verification")
print("=" * 70)
print(f"\nDetected by FullResourceCapacity.from_system(gpu_max_slots=50):")
print(f"Host Memory: {capacity.host_memory_bytes / (1024**3):.2f}GB")
print(f"GPU Devices: {len(capacity.gpu_devices)}")

for gpu_dev in capacity.gpu_devices:
    print(f"\n  GPU {gpu_dev.gpu_id}:")
    print(f"    Memory: {gpu_dev.memory_bytes / (1024**3):.2f}GB")
    print(f"    max_slots: {gpu_dev.max_slots}")
    print(f"    Max tasks if 1GB each: {gpu_dev.memory_bytes // (1024 * 1024 * 1024)}")

# 実際のスケジューラ容量確認
print(f"\nTotal GPU memory available: {sum(gpu.memory_bytes for gpu in capacity.gpu_devices) / (1024**3):.2f}GB")
print(f"Total tasks (50 × 1GB) demand: 50.0GB")
print(f"Host memory available: {capacity.host_memory_bytes / (1024**3):.2f}GB")
print(f"Total tasks (50 × 1.2GB host) demand: 60.0GB")

print("\n" + "=" * 70)
print("Analysis:")
if sum(gpu.memory_bytes for gpu in capacity.gpu_devices) < 50 * (1024**3):
    print("✗ GPU memory insufficient for all 50 tasks (expected)")
else:
    print("✗ ERROR: GPU memory reported as sufficient (unexpected)")

if capacity.host_memory_bytes < 60 * (1024**3):
    print("✓ Host memory insufficient for all 50 tasks (expected)")
else:
    print("✗ Host memory sufficient (why are tasks failing on GPU mem?)")
print("=" * 70)
