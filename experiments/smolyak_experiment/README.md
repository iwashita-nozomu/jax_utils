# Smolyak Experiment Module

本モジュールは Smolyak 積分器の大規模スケーリング実験を実施するための専用パッケージです。

## 実験概要

### 目的

Smolyak 積分器の性能を多次元・多レベルにわたって系統的に測定し、以下を明らかにします：

- **初期化時間のスケーリング特性**: 次元 $d$ とレベル $\ell$ に対する計算時間の関数形
- **dtype（浮動小数点型）による精度・速度トレードオフ**: float16, bfloat16, float32, float64 の相対性能
- **実装の安定性**: 大規模パラメータ領域での数値的安定性と異常検出

### 実験パラメータ

- **次元**: $d \in [1, 50]$ （50 値）
- **レベル**: $\ell \in [1, 50]$ （50 値）
- **データ型**: float16, bfloat16, float32, float64 （4 値）
- **試行回数**: 各 (d, ℓ, dtype) の組み合わせについて 3 回
- **総ケース数**: $50 \times 50 \times 4 \times 3 = 30,000$ ケース

### 期待される実行時間

- **CPU のみ**: 約 25 時間
- **GPU 上（GPU 版 JAX, 4 並列）**: 約 6-7 時間

## ディレクトリ構造

```
smolyak_experiment/
├── __init__.py                      # パッケージ初期化
├── README.md                        # このファイル
├── cases.py                         # 実験ケース定義
├── runner_config.py                 # 実行構成（パラメータ範囲）
├── results_aggregator.py            # 結果集計・フィルタリング
└── (test_*.py)                      # 専用テスト類（実装予定）
```

## 各モジュールの説明

### `cases.py`

Smolyak 実験に特化したケース生成・検証ロジック。

責務:
- Smolyak 積分器向けのケース仕様を定義
- d × ℓ × dtype の直積からケースリストを生成
- 各ケースに一意な case_id を付与

提供予定の主要 API:
```python
class SmolyakExperimentCase:
    case_id: str
    dimension: int
    level: int
    dtype: str  # "float16", "bfloat16", "float32", "float64"
    trial_index: int
```

### `runner_config.py`

大規模実験全体の実行構成を定義。

責務:
- パラメータ範囲（d=1-50, ℓ=1-50, dtype 4値）を定義
- タイムアウト、リトライ戦略を設定
- CPU/GPU 別の実行設定を管理

提供予定の主要 API:
```python
class SmolyakExperimentConfig:
    min_dimension: int = 1
    max_dimension: int = 50
    min_level: int = 1
    max_level: int = 50
    dtypes: list[str] = ["float16", "bfloat16", "float32", "float64"]
    num_trials: int = 3
    timeout_seconds: float = 300.0
    device: Literal["cpu", "gpu"] = "cpu"
```

### `results_aggregator.py`

実験結果の集計・フィルタリング・分析。

責務:
- 生の実験結果（JSONL）を読み込み
- dtype × dimension 別に結果を集計
- 異常値・タイムアウト・失敗を分類
- 可視化用の中間ファイルを生成

## 実装ステータス

- [ ] `cases.py` - 実験ケース定義
- [ ] `runner_config.py` - 実行構成管理
- [ ] `results_aggregator.py` - 結果集計ロジック
- [ ] Unit tests - 各モジュールのテスト
- [ ] Smoke test - 小規模パラメータでの動作確認

## 使用方法（実装後）

```python
from experiments.smolyak_experiment import cases, runner_config

# 構成設定
config = runner_config.SmolyakExperimentConfig()
print(f"Total cases: {config.total_cases}")
print(f"Total tasks: {config.total_tasks}")

# ケース生成
case_spec = cases.make_smolyak_case_spec(config)

# 結果集計（実装後）
# aggregated = results_aggregator.aggregate_by_dtype(results)
```

## 参考資料

- [実験ディレクトリ計画と設計](../../notes/knowledge/experiment_directory_planning.md)
- [ベンチマーク報告書](../../python/benchmark/functional/results/BENCHMARK_REPORT.md)
- [既存実験参考: smolyak_scaling](../functional/smolyak_scaling/README.md)

---

**作成日**: 2026-03-20  
**バージョン**: 1.0
