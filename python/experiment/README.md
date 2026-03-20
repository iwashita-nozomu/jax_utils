# Python Experiment Module

Smolyak 積分器の大規模スケーリング実験を実施するための Python パッケージです。

## 概要

- **目的**: Smolyak 積分器の性能をスケーリング測定
- **対象**: 次元 d=1-50、レベル ℓ=1-50、dtype 4 型
- **総タスク数**: 50 × 50 × 4 × 3 = 30,000
- **推定実行時間**: CPU 約 25 時間、GPU 約 6-7 時間

## ディレクトリ構成

```
experiment/
├── __init__.py
├── README.md
└── smolyakexperiment/
    ├── __init__.py
    ├── README.md
    ├── cases.py
    ├── runner_config.py
    └── results_aggregator.py
```

## 実装ステータス

- [ ] ケース生成ロジック実装 (`cases.py`)
- [ ] 実行構成定義 (`runner_config.py`)
- [ ] 結果集計ロジック実装 (`results_aggregator.py`)
- [ ] ユニットテスト作成
- [ ] smoke test 実行
- [ ] 大規模実験実行

---

## 実験設計

### パラメータ範囲

| パラメータ | 範囲 | ケース数 |
|-----------|------|---------|
| **Dimension** | 1-50 | 50 |
| **Level** | 1-50 | 50 |
| **dtype** | float16, bfloat16, float32, float64 | 4 |
| **試行回数** | 3 回 | 3 |

### 総タスク数

50 × 50 × 4 × 3 = **30,000 タスク**

---

## 使用方法（実装後）

```python
from python.experiment.smolyakexperiment import cases, runner_config, results_aggregator

# 構成設定
config = runner_config.SmolyakExperimentConfig()
print(f"Total cases: {config.total_cases}")
print(f"Total tasks: {config.total_tasks}")

# ケース生成
case_spec = cases.make_smolyak_case_spec(config)

# 結果集計（実装後）
# aggregated = results_aggregator.aggregate_by_dtype(results)
```

---

## 関連ドキュメント

- [Smolyak Experiment README](./smolyakexperiment/README.md)
- [30_experiment_directory_structure.md](../../documents/conventions/python/30_experiment_directory_structure.md)
- [experiment_directory_planning.md](../../notes/knowledge/experiment_directory_planning.md)

---

## 次のステップ

1. **モジュール実装**
   - ケース生成ロジック (`cases.py`)
   - 実行構成定義 (`runner_config.py`)
   - 結果集計 (`results_aggregator.py`)

2. **テスト & 検証**
   - ユニットテスト作成
   - smoke test（d=1-3, ℓ=1-3 で動作確認）

3. **大規模実験実行**
   - フル実験実行（推定 6-25 時間）
   - 結果分析・frontier 生成

