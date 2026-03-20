# -*- coding: utf-8 -*-
"""
Smolyak 実験実行構成モジュール

大規模実験全体のパラメータ範囲、リソース制約、実行設定を管理。
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class SmolyakExperimentConfig:
    """
    Smolyak 大規模実験の実行構成。
    
    Attributes
    ----------
    min_dimension : int
        最小次元（デフォルト: 1）
    max_dimension : int
        最大次元（デフォルト: 50）
    min_level : int
        最小レベル（デフォルト: 1）
    max_level : int
        最大レベル（デフォルト: 50）
    dtypes : list[str]
        対象データ型（デフォルト: float16, bfloat16, float32, float64）
    num_trials : int
        各ケースの試行回数（デフォルト: 3）
    timeout_seconds : float
        単一ケース実行のタイムアウト（秒）（デフォルト: 300.0）
    device : str
        実行デバイス（"cpu" または "gpu"）（デフォルト: "cpu"）
    experimental : bool
        実験的フラグ（デフォルト: False）
        
    Notes
    -----
    - 総ケース数: min_dimension～max_dimension × min_level～max_level × len(dtypes) × num_trials
    - デフォルト設定: 50×50×4×3 = 30,000 ケース
    """
    
    min_dimension: int = 1
    max_dimension: int = 50
    min_level: int = 1
    max_level: int = 50
    dtypes: list[str] = field(default_factory=lambda: [
        "float16", "bfloat16", "float32", "float64"
    ])
    num_trials: int = 3
    timeout_seconds: float = 300.0
    device: Literal["cpu", "gpu"] = "cpu"
    experimental: bool = False
    
    @property
    def total_cases(self) -> int:
        """総ケース数を計算（試行なし）"""
        return (
            (self.max_dimension - self.min_dimension + 1)
            * (self.max_level - self.min_level + 1)
            * len(self.dtypes)
        )
    
    @property
    def total_tasks(self) -> int:
        """総タスク数を計算（試行を含む）"""
        return self.total_cases * self.num_trials
