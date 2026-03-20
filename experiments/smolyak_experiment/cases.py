# -*- coding: utf-8 -*-
"""
Smolyak 実験ケース定義モジュール

Smolyak 積分器向けの大規模実験ケースを生成・検証するロジック。
リソース見積もり機能も提供。
"""

from typing import Any
import sys
from pathlib import Path

# 相対スケジューラのインポートを解決
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
if str(WORKSPACE_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT / "python"))

from jax_util.experiment_runner import FullResourceEstimate


def estimate_case_resources(case: dict[str, Any]) -> FullResourceEstimate:
    """
    Smolyak 積分器ケースのリソース見積もり。
    
    次元とレベルから、評価点数と必要なメモリを推定する。
    保守的だが極度に過剰でない見積もりを返す。
    
    Parameters
    ----------
    case : dict
        {"dimension": int, "level": int, "dtype": str, ...}
        
    Returns
    -------
    FullResourceEstimate
        ホストメモリ見積もり（バイト単位）
    """
    dimension = case["dimension"]
    level = case["level"]
    dtype = case["dtype"]
    
    # データ型ごとのバイト数
    dtype_bytes = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
        "float64": 8,
    }
    bytes_per_element = dtype_bytes.get(dtype, 4)  # デフォルト float32
    
    # Smolyak グリッド上の評価点数（粗い推定）
    # 最大で 2^level × dimension くらいの点数に規模化
    # 実際の評価点数は (2^(level+1) - 1)^dimension だが、キャッシュや
    # JAX JIT コンパイル時のオーバーヘッドを考慮して保守的に見積もる
    max_points_multiplier = (2 ** min(level, 6))  # level > 6 では飽和
    estimated_points = max_points_multiplier ** min(dimension, 4)
    
    # メモリ見積もり（係数を下げめ）
    # factor = 2.5（入出力 + 中間計算 + JAX JIT 用領域）
    factor = 2.5
    host_memory_bytes = int(estimated_points * bytes_per_element * factor)
    
    # 最大値を 1 GB に制限（smoke/small 実験では十分）
    max_memory = 1 * 1024 * 1024 * 1024  # 1 GB
    host_memory_bytes = min(host_memory_bytes, max_memory)
    
    # CPU 上で実行（GPU を使わない）
    return FullResourceEstimate(
        host_memory_bytes=host_memory_bytes,
        gpu_count=0,
        gpu_memory_bytes=0,
        gpu_slots=1,
    )


def generate_cases(config: Any) -> list[dict[str, Any]]:
    """
    Smolyak 実験用のケースリストを生成。
    
    次元・レベル・データ型の直積からケースを生成し、各ケースに
    一意な case_id を付与する。
    
    Parameters
    ----------
    config : SmolyakExperimentConfig
        実験構成（min/max dimension, level, dtype リスト等）
        
    Returns
    -------
    list[dict[str, Any]]
        各辞書は以下を含む:
        - case_id: str - 一意識別子（"d{d}_l{l}_{dtype}_t{trial}"）
        - dimension: int
        - level: int
        - dtype: str
        - trial_index: int
        
    Examples
    --------
    >>> from experiments.smolyak_experiment import runner_config, cases
    >>> config = runner_config.SmolyakExperimentConfig(
    ...     min_dimension=1, max_dimension=3,
    ...     min_level=1, max_level=2,
    ...     dtypes=["float32"],
    ...     num_trials=1
    ... )
    >>> case_list = cases.generate_cases(config)
    >>> len(case_list)  # 3 * 2 * 1 * 1
    6
    >>> case_list[0]["case_id"]
    'd1_l1_float32_t0'
    """
    cases_list: list[dict[str, Any]] = []
    case_counter = 0
    
    # dimension -> level -> dtype -> trial の順序で生成
    for dimension in range(config.min_dimension, config.max_dimension + 1):
        for level in range(config.min_level, config.max_level + 1):
            for dtype in config.dtypes:
                for trial_index in range(config.num_trials):
                    case_id = f"d{dimension}_l{level}_{dtype}_t{trial_index}"
                    case = {
                        "case_id": case_id,
                        "dimension": dimension,
                        "level": level,
                        "dtype": dtype,
                        "trial_index": trial_index,
                        "index": case_counter,
                    }
                    cases_list.append(case)
                    case_counter += 1
    
    return cases_list
