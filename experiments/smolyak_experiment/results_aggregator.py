# -*- coding: utf-8 -*-
"""
Smolyak 実験結果集計モジュール

実験結果の集計、フィルタリング、分析ロジック。
"""

from typing import Any
from collections import defaultdict


def aggregate_by_dtype(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """
    実験結果をデータ型別に集計。
    
    Parameters
    ----------
    results : list[dict]
        個別の実験結果
        
    Returns
    -------
    dict[str, list[dict]]
        {dtype: [結果1, 結果2, ...]} の形式で集計
    """
    aggregated: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        dtype = result.get("dtype", "unknown")
        aggregated[dtype].append(result)
    return dict(aggregated)


def aggregate_by_dimension(results: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """
    実験結果を次元別に集計。
    
    Parameters
    ----------
    results : list[dict]
        個別の実験結果
        
    Returns
    -------
    dict[int, list[dict]]
        {dimension: [結果1, 結果2, ...]} の形式で集計
    """
    aggregated: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        dimension = result.get("dimension", -1)
        aggregated[dimension].append(result)
    return dict(sorted(aggregated.items()))


def aggregate_by_level(results: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    """
    実験結果をレベル別に集計。
    
    Parameters
    ----------
    results : list[dict]
        個別の実験結果
        
    Returns
    -------
    dict[int, list[dict]]
        {level: [結果1, 結果2, ...]} の形式で集計
    """
    aggregated: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        level = result.get("level", -1)
        aggregated[level].append(result)
    return dict(sorted(aggregated.items()))


def filter_failures(results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """
    実験結果から失敗ケースを分類・抽出。
    
    Parameters
    ----------
    results : list[dict]
        個別の実験結果
        
    Returns
    -------
    dict[str, list[dict]]
        失敗分類: {category: [失敗1, 失敗2, ...]}
        
    Notes
    -----
    失敗を以下に分類:
    - SUCCESS: 正常完了
    - TIMEOUT: 実行時間超過
    - OOM: メモリ不足
    - DIVERGENCE: 数値発散
    - NUMERICAL: 数値エラー
    """
    failures: dict[str, list[dict[str, Any]]] = {
        "SUCCESS": [],
        "TIMEOUT": [],
        "OOM": [],
        "DIVERGENCE": [],
        "NUMERICAL": [],
        "OTHER": [],
    }
    
    for result in results:
        status = result.get("status", "OTHER")
        if status in failures:
            failures[status].append(result)
        else:
            failures["OTHER"].append(result)
    
    return {k: v for k, v in failures.items() if v}  # 空きのをフィルタ


def compute_statistics(results: list[dict[str, Any]], field_name: str) -> dict[str, Any]:
    """
    結果の統計量を計算。
    
    Parameters
    ----------
    results : list[dict]
        個別の実験結果
    field_name : str
        統計対象フィールド名（例："init_time"）
        
    Returns
    -------
    dict[str, Any]
        統計量（min, max, mean, std等）
    """
    import statistics
    
    values = [r.get(field_name) for r in results if field_name in r]
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }
