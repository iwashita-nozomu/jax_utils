#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smolyak 積分器スケーリング実験

次元・レベル・データ型の全組み合わせで Smolyak 積分器の
初期化時間・実行時間を測定する。

実験コードはワーカー定義とケース定義に専念。
リソース管理、スケジューリング、結果集計はランナー側で自動処理。
"""

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
import subprocess

try:
    import psutil
except Exception:
    psutil = None

# ワークスペース構成
SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
WORKSPACE_ROOT = SCRIPT_DIR
if str(WORKSPACE_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "python"))
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# JAX メモリ先取り無効化（最初のインポート前）
from jax_util.experiment_runner import disable_jax_memory_preallocation
# JAX の GPU メモリ先取りはワーカーごとの環境変数で制御するため無効化する
disable_jax_memory_preallocation(gpu_devices=True)

from jax_util.experiment_runner import (
    FullResourceEstimate,
    StandardFullResourceScheduler,
    StandardRunner,
    TaskContext,
    SUCCESS_EXIT_CODE,
    WORKER_PROTOCOL_ERROR_EXIT_CODE,
    apply_environment_variables,
)
from jax_util.experiment_runner.protocols import Worker
from experiments.smolyak_experiment import cases, runner_config


# ============================================================================
# ワーカー定義
# ============================================================================

class SmolyakWorker(Worker[dict[str, Any], int]):
    """Smolyak 積分器を実行するワーカー
    
    ワーカープロセス起動時の環境変数設定に対応。
    TaskContext から environment_variables dict を取得し、
    JAX インポート前に os.environ にセットする。
    
    型パラメータ:
    - T = dict[str, Any] （ケース）
    - U = int （終了コード）
    """

    def __call__(self, case: dict[str, Any], context: TaskContext) -> int:
        """ケースを実行し、結果を JSONL に記録
        
        Process:
        1. TaskContext から GPU 割り当て情報を取得
        2. apply_environment_variables で os.environ に環境変数をセット
        3. JAX をインポート
        4. ケース実行
        5. 結果を JSONL に記録
        """
        try:
            # 1. TaskContext から environment_variables を取得して os.environ にセット
            apply_environment_variables(context)

            # デバッグ: 環境変数設定を確認
            env_vars = context.get("environment_variables", {})
            cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')
            gpu_ids_val = env_vars.get("CUDA_VISIBLE_DEVICES", "N/A")
            sys.stderr.write(f"[Worker] Case {case['case_id']}: env_vars keys={list(env_vars.keys())}, CUDA_VISIBLE_DEVICES={cuda_visible}\n")
            sys.stderr.flush()

            # 軽量メトリクス収集ヘルパー
            def _collect_metrics() -> dict[str, Any]:
                metrics: dict[str, Any] = {}
                try:
                    if psutil is not None:
                        proc = psutil.Process()
                        metrics["host_rss_bytes"] = int(proc.memory_info().rss)
                    else:
                        metrics["host_rss_bytes"] = None
                except Exception:
                    metrics["host_rss_bytes"] = None

                # GPU メモリ使用量（nvidia-smi が利用可能な場合）
                metrics["gpu_mem_bytes"] = None
                try:
                    gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES")
                    if gpu_env and gpu_env != "NOT SET":
                        gpu_ids = [int(x) for x in gpu_env.split(",") if x.strip().isdigit()]
                        try:
                            out = subprocess.check_output([
                                "nvidia-smi",
                                "--query-gpu=memory.used",
                                "--format=csv,nounits,noheader",
                            ])
                            lines = out.decode().strip().splitlines()
                            total = 0
                            for gid in gpu_ids:
                                if 0 <= gid < len(lines):
                                    total += int(lines[gid].strip()) * 1024 * 1024
                            metrics["gpu_mem_bytes"] = total
                        except Exception:
                            metrics["gpu_mem_bytes"] = None
                    else:
                        metrics["gpu_mem_bytes"] = None
                except Exception:
                    metrics["gpu_mem_bytes"] = None

                return metrics

            # 事前メトリクス
            pre_metrics = _collect_metrics()

            # 2. ここで JAX をインポート（GPU ID が既に設定されている）
            import jax.numpy as jnp
            from jax_util.functional.smolyak import SmolyakIntegrator

            sys.stderr.write(f"[Worker] Case {case['case_id']}: JAX imported successfully\n")
            sys.stderr.flush()

            # 3. ケース実行
            result = self._run_case(case, jnp, SmolyakIntegrator)

            # 実行後メトリクス
            post_metrics = _collect_metrics()

            # メトリクスを結果に追加して保存
            result.setdefault("metrics", {})
            result["metrics"]["pre"] = pre_metrics
            result["metrics"]["post"] = post_metrics

            self._save_result(result, context)
            return SUCCESS_EXIT_CODE
        except Exception as e:
            context["error"] = str(e)
            sys.stderr.write(f"[Worker] Case {case['case_id']}: ERROR: {e}\n")
            sys.stderr.flush()
            return WORKER_PROTOCOL_ERROR_EXIT_CODE

    def resource_estimate(self, case: dict[str, Any]) -> FullResourceEstimate:
        """ケースのリソース見積もり"""
        return cases.estimate_case_resources(case)

    @staticmethod
    def _run_case(
        case: dict[str, Any],
        jnp: Any,
        SmolyakIntegrator: Any,
    ) -> dict[str, Any]:
        """単一ケースを実行
        
        被積分関数 f(x) = sum(x_i^2) の積分を [-0.5, 0.5]^d 上で計算。
        
        解析解：
            ∫_{[-0.5,0.5]^d} sum(x_i^2) dx
            = d × ∫_{-0.5}^{0.5} x^2 dx
            = d × [x^3/3]_{-0.5}^{0.5}
            = d × ((0.5)^3 - (-0.5)^3) / 3
            = d × (0.125 + 0.125) / 3
            = d / 12
        """
        result = {
            "case_id": case["case_id"],
            "dimension": case["dimension"],
            "level": case["level"],
            "dtype": case["dtype"],
            "trial_index": case["trial_index"],
            "status": "SUCCESS",
            "init_time_ms": 0.0,
            "integrate_time_ms": 0.0,
            "num_evaluation_points": 0,
            "integral_value": 0.0,
            "analytical_value": 0.0,
            "absolute_error": 0.0,
            "relative_error": 0.0,
            "error": None,
        }

        try:
            # dtype 取得
            jax_dtype = getattr(jnp, case["dtype"])

            # 初期化時間測定
            t0 = time.perf_counter()
            integrator = SmolyakIntegrator(
                dimension=case["dimension"],
                level=case["level"],
                dtype=jax_dtype,
            )
            result["init_time_ms"] = (time.perf_counter() - t0) * 1000.0
            # 追加メトリクス: ルール storage のバイト数（見積り）
            try:
                result["storage_bytes"] = int(integrator.storage_bytes)
            except Exception:
                result["storage_bytes"] = None
            result["num_evaluation_points"] = int(integrator.num_evaluation_points)

            # 被積分関数：f(x) = sum(x_i^2)
            def integrand(x: jnp.ndarray) -> jnp.ndarray:
                return jnp.sum(x**2, axis=-1)

            # 積分実行（2回平均）
            times = []
            integral_values = []

            

            for _ in range(2):
                t0 = time.perf_counter()
                integral_val = integrator.integrate(integrand)  # type: ignore[arg-type]
                times.append((time.perf_counter() - t0) * 1000.0)
                integral_values.append(float(integral_val))

            # プロセス CPU 時間（user + system）を収集
            try:
                if psutil is not None:
                    p = psutil.Process()
                    cpu_times = p.cpu_times()
                    result.setdefault("metrics", {})
                    result["metrics"]["cpu_user_time_s"] = float(cpu_times.user)
                    result["metrics"]["cpu_system_time_s"] = float(cpu_times.system)
            except Exception:
                pass
            
            result["integrate_time_ms"] = sum(times) / len(times)
            result["integral_value"] = sum(integral_values) / len(integral_values)
            
            # 解析解：積分領域 [-0.5, 0.5]^d 上での sum(x_i^2) の積分 = dimension / 12
            analytical_value = float(case["dimension"] / 12.0)
            result["analytical_value"] = analytical_value
            result["absolute_error"] = abs(result["integral_value"] - analytical_value)
            result["relative_error"] = (
                result["absolute_error"] / abs(analytical_value) 
                if analytical_value != 0 else 0.0
            )

        except Exception as e:
            result["status"] = "FAILURE"
            result["error"] = str(e)

        return result

    @staticmethod
    def _save_result(result: dict[str, Any], context: TaskContext) -> None:
        """結果を JSONL に追記"""
        jsonl_path = context.get("jsonl_path")
        if not jsonl_path:
            return

        path = Path(jsonl_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            # fcntl 非対応環境
            with open(path, "a") as f:
                f.write(json.dumps(result) + "\n")


# ============================================================================
# 結果集計
# ============================================================================

def _generate_final_results(
    jsonl_path: str | Path,
    config: Any,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """JSONL ファイルから統計情報を抽出して最終結果を生成
    
    Parameters
    ----------
    jsonl_path : str | Path
        ケースごとの結果を記録した JSONL ファイルパス
    config : SmolyakExperimentConfig
        実験設定
    elapsed_seconds : float
        実験全体の経過時間（秒）
        
    Returns
    -------
    dict[str, Any]
        最終結果（条件、成功/失敗統計、時間、精度指標）
    """
    jsonl_path = Path(jsonl_path)
    
    # JSONL から全結果を読み込み
    results = []
    if jsonl_path.exists():
        with open(jsonl_path, "r") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    
    # 統計情報を計算
    total_cases = len(results)
    successful_cases = sum(1 for r in results if r.get("status") == "SUCCESS")
    failed_cases = total_cases - successful_cases
    
    # タイミング情報を抽出
    init_times = [r["init_time_ms"] for r in results if r.get("init_time_ms", 0) > 0]
    integrate_times = [r["integrate_time_ms"] for r in results if r.get("integrate_time_ms", 0) > 0]
    
    # 精度情報を抽出
    absolute_errors = [
        r["absolute_error"] for r in results 
        if r.get("status") == "SUCCESS" and "absolute_error" in r
    ]
    relative_errors = [
        r["relative_error"] for r in results 
        if r.get("status") == "SUCCESS" and "relative_error" in r
    ]
    
    final_result = {
        # 実験条件
        "condition": config.to_dict(),
        # 実行統計
        "total_cases": total_cases,
        "successful_cases": successful_cases,
        "failed_cases": failed_cases,
        "success_rate": (successful_cases / total_cases * 100) if total_cases > 0 else 0.0,
        # 時間統計
        "elapsed_seconds": elapsed_seconds,
        "throughput_cases_per_second": (total_cases / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
        "init_time_stats": {
            "count": len(init_times),
            "mean_ms": (sum(init_times) / len(init_times)) if init_times else 0.0,
            "min_ms": min(init_times) if init_times else 0.0,
            "max_ms": max(init_times) if init_times else 0.0,
        },
        "integrate_time_stats": {
            "count": len(integrate_times),
            "mean_ms": (sum(integrate_times) / len(integrate_times)) if integrate_times else 0.0,
            "min_ms": min(integrate_times) if integrate_times else 0.0,
            "max_ms": max(integrate_times) if integrate_times else 0.0,
        },
        # 積分精度統計
        "accuracy_stats": {
            "absolute_error": {
                "count": len(absolute_errors),
                "mean": (sum(absolute_errors) / len(absolute_errors)) if absolute_errors else 0.0,
                "min": min(absolute_errors) if absolute_errors else 0.0,
                "max": max(absolute_errors) if absolute_errors else 0.0,
                "median": sorted(absolute_errors)[len(absolute_errors)//2] if absolute_errors else 0.0,
            },
            "relative_error": {
                "count": len(relative_errors),
                "mean": (sum(relative_errors) / len(relative_errors)) if relative_errors else 0.0,
                "min": min(relative_errors) if relative_errors else 0.0,
                "max": max(relative_errors) if relative_errors else 0.0,
                "median": sorted(relative_errors)[len(relative_errors)//2] if relative_errors else 0.0,
            },
        },
    }
    
    return final_result


# ============================================================================
# ケース定義
# ============================================================================

def get_experiment_config(size: str) -> runner_config.SmolyakExperimentConfig:
    """サイズに応じた実験設定を返す"""
    configs = {
        "smoke": runner_config.SmolyakExperimentConfig(
            min_dimension=1,
            max_dimension=3,
            min_level=1,
            max_level=2,
            dtypes=["float32"],
            num_trials=1,
            device="cpu",
        ),
        "small": runner_config.SmolyakExperimentConfig(
            min_dimension=1,
            max_dimension=5,
            min_level=1,
            max_level=5,
            dtypes=["float32"],
            num_trials=2,
            device="cpu",
        ),
        "medium": runner_config.SmolyakExperimentConfig(
            min_dimension=1,
            max_dimension=10,
            min_level=1,
            max_level=10,
            dtypes=["float32", "float64"],
            num_trials=2,
            device="gpu",
        ),
        "large": runner_config.SmolyakExperimentConfig(
            min_dimension=1,
            max_dimension=20,
            min_level=1,
            max_level=20,
            dtypes=["float16", "bfloat16", "float32", "float64"],
            num_trials=3,
            device="gpu",
        ),
    }
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")
    return configs[size]


# ============================================================================
# 実験実行
# ============================================================================

def main(size: str = "smoke") -> None:
    """実験を実行"""
    config = get_experiment_config(size)
    config.validate()

    print("=" * 70)
    print(f"Smolyak Experiment - {size.upper()}")
    print("=" * 70)
    print(f"Config: dim {config.min_dimension}-{config.max_dimension}, "
          f"level {config.min_level}-{config.max_level}, "
          f"{len(config.dtypes)} dtypes, "
          f"{config.total_cases} cases\n")

    # ケース生成
    case_list = cases.generate_cases(config)
    
    # 出力ファイル
    output_dir = Path(__file__).parent / "results" / size
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = output_dir / f"results_{int(time.time())}.jsonl"

    # コンテキストビルダー
    # NOTE: environment_variables dict は StandardFullResourceScheduler が
    #       リソース割り当てに基づいて生成する。ワーカーはこれを
    #       TaskContext から取得して os.environ に適用する。
    def context_builder(case: dict[str, Any]) -> TaskContext:
        return {
            "case_id": case["case_id"],
            "jsonl_path": str(jsonl_file),
            # environment_variables は scheduler が動的に生成して追加する
        }

    # プログレス表示
    def progress(completed: int, total: int, elapsed: float, running: int) -> None:
        if total > 0:
            pct = (completed / total) * 100
            throughput = (completed / elapsed) if elapsed > 0 else 0
            print(f"\r[{completed:4d}/{total:4d}] {pct:5.1f}% | "
                  f"Throughput: {throughput:6.2f} cases/s | "
                  f"Running: {running:d} | "
                  f"Elapsed: {elapsed:7.1f}s",
                  end="", flush=True)

    # スケジューラ（リソース自動検出、GPU 共有は自動判定）
    # ワーカーインスタンスを作成し、スケジューラはワーカーの
    # resource_estimate を使って構築する。GPU 実行時は GPU プリアロケを無効化。
    worker = SmolyakWorker()

    scheduler = StandardFullResourceScheduler.from_worker(
        cases=case_list,
        worker=worker,
        context_builder=context_builder,
        disable_gpu_preallocation=(config.device == "gpu"),
    )

    # 実行: 同じワーカーインスタンスを渡す
    runner = StandardRunner(scheduler, progress_callback=progress)
    print(f"Running {len(case_list)} cases with experiment runner\n")

    t0 = time.time()
    runner.run(worker)
    elapsed = time.time() - t0

    print(f"\n\nElapsed: {elapsed:.1f}s ({len(case_list) / elapsed:.2f} cases/s)")
    
    # JSONL から統計情報を抽出して最終 JSON を生成
    final_results = _generate_final_results(jsonl_file, config, elapsed)
    final_json_file = output_dir / f"final_results_{int(time.time())}.json"
    with open(final_json_file, "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"JSONL results: {jsonl_file}")
    print(f"Final JSON: {final_json_file}")
    print("=" * 70)
    
    # stdout に最終結果ファイルパスを出力（スクリプト外部での参照用）
    print(json.dumps({"output_jsonl": str(jsonl_file), "output_json": str(final_json_file)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smolyak experiment")
    parser.add_argument("--size", default="smoke",
                        choices=["smoke", "small", "medium", "large"],
                        help="Experiment size")
    args = parser.parse_args()

    main(args.size)
