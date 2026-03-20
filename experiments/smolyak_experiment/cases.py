# -*- coding: utf-8 -*-
"""
Smolyak 実験ケース定義モジュール

Smolyak 積分器向けの大規模実験ケースを生成・検証するロジック。
"""

from typing import Any


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
