from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from jax_util.experiment_runner import (
    CHILD_COMPLETE_PREFIX,
    append_jsonl_record,
    apply_worker_environment,
    json_compatible,
    worker_slot_from_mapping,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", required=True)
    parser.add_argument("--run-config-json", required=True)
    parser.add_argument("--worker-slot-json", required=True)
    parser.add_argument("--jsonl-output", type=Path, required=True)
    args = parser.parse_args()

    case = json.loads(args.case_json)
    run_config = json.loads(args.run_config_json)
    worker_slot = worker_slot_from_mapping(json.loads(args.worker_slot_json))

    apply_worker_environment(
        platform=str(run_config["platform"]),
        worker_slot=worker_slot,
        disable_gpu_preallocation=bool(run_config.get("disable_gpu_preallocation", True)),
    )

    import jax
    import jax.numpy as jnp

    platform = str(run_config["platform"])
    target_device = jax.devices("gpu")[0] if platform == "gpu" else jax.devices("cpu")[0]
    matrix_size = int(case.get("matrix_size", 2048))
    min_work_seconds = float(case.get("min_work_seconds", 2.5))

    with jax.default_device(target_device):
        lhs = jnp.ones((matrix_size, matrix_size), dtype=jnp.float32)
        rhs = jnp.eye(matrix_size, dtype=jnp.float32)
        acc = lhs

    started = time.perf_counter()
    iterations = 0
    while time.perf_counter() - started < min_work_seconds:
        acc = jnp.tanh(acc @ rhs)
        jax.block_until_ready(acc)
        iterations += 1

    work_seconds = time.perf_counter() - started
    visible_gpu_ids = [int(device.id) for device in jax.devices("gpu")] if platform == "gpu" else []

    result = {
        "status": "ok",
        "case_id": int(case["case_id"]),
        "assigned_gpu_index": worker_slot.gpu_index,
        "worker_label": worker_slot.worker_label,
        "gpu_slot": worker_slot.gpu_slot,
        "cpu_affinity": list(worker_slot.cpu_affinity),
        "backend": jax.default_backend(),
        "device_kind": target_device.device_kind,
        "visible_device_id": int(target_device.id),
        "gpu_device_count": len(visible_gpu_ids),
        "visible_gpu_ids": visible_gpu_ids,
        "matrix_size": matrix_size,
        "iterations": iterations,
        "work_seconds": work_seconds,
        "pid": os.getpid(),
        "checksum": float(jax.device_get(acc[0, 0])),
    }

    append_jsonl_record(args.jsonl_output, result)
    print(f"{CHILD_COMPLETE_PREFIX}{json.dumps(json_compatible(result), ensure_ascii=True)}", flush=True)


if __name__ == "__main__":
    main()
