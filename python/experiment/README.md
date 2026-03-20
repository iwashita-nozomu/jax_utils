# Python Experiment Module

このディレクトリは、大規模な実験（特に Smolyak 積分器のスケーリング実験）をモジュール化・再実行可能な形で実装するための Python パッケージです。

## 概要

- **目的**: 実験ロジック（ケース生成・実行・分析）を再利用可能なモジュール群として整理
- **対象問題**: Smolyak 積分器の大規模スケーリング実験（d=1-50, level=1-50, 4 dtype）
- **実行方式**: CPU/GPU 両対応、並列実行、JSONL 逐次保存、失敗分類

## ディレクトリ構成

```
experiment/
├── __init__.py                      # Public API 定義
├── README.md                        # このファイル
├── case_generator.py                # ケース生成ロジック（未実装）
├── protocols.py                     # Protocol/Interface 定義（未実装）
├── runner.py                        # 実験実行エンジン（未実装）
├── analysis.py                      # 結果分析・集計ロジック（未実装）
├── utils.py                         # ユーティリティ関数（未実装）
│
├── smolyak/                         # Smolyak 積分器専用モジュール
│   ├── __init__.py                  # Smolyak パッケージ API
│   ├── cases.py                     # ケース定義（未実装）
│   ├── runner_config.py             # 実行構成（未実装）
│   ├── test_cases.py                # テスト用ケース定義（未実装）
│   └── results_aggregator.py        # 結果集計（未実装）
│
├── tests/                           # ユニットテスト
│   ├── __init__.py
│   ├── test_case_generator.py       # ケース生成テスト（未実装）
│   ├── test_runner.py               # 実行エンジン smoke test（未実装）
│   ├── test_analysis.py             # 分析ロジックテスト（未実装）
│   └── test_smolyak_*.py            # Smolyak 専用テスト（未実装）
│
└── fixtures/                        # テスト用固定データ
    ├── small_cases.json             # 小規模ケース（未実装）
    └── baseline_results.json        # 期待結果（未実装）
```

## 実装ステータス

- [ ] **Phase 1**: 基礎モジュール実装
- [ ] **Phase 2**: Smolyak 専用実装
- [ ] **Phase 3**: 大規模実験実行・結果集計

### 現在

**2026-03-20**: 規約作成・ディレクトリ構造確定

- [x] `documents/conventions/python/30_experiment_directory_structure.md` 作成
- [x] `notes/knowledge/experiment_directory_planning.md` 作成
- [ ] コード実装（次フェーズ）

## 設計仕様

### 大規模 Smolyak 実験設計

実験パラメータ範囲:

| パラメータ | 範囲 | 理由 |
|-----------|------|------|
| **Dimension** | 1-50 | アルゴリズム限界確認 |
| **Level** | 1-50 | 精度-計算コスト trade-off |
| **dtype** | float16, bfloat16, float32, float64 | precision-speed 分析 |
| **Repeats** | 3 | 統計的安定性 |

### 総ケース数

- 50 × 50 × 4 = 10,000 ケース
- × 3 repeats = 30,000 単位タスク

### 推定実行時間

- CPU alone: ~25 時間
- GPU 4 並列: ~6-7 時間
- パイロット（d=1-20, level=1-20）: ~1 時間

---

## 使用方法（将来）

### 小規模テスト実行

```bash
# d=1-3, level=1-3, float32 のみで試験
python3 -m python.experiment.smolyak.runner \
  --platform cpu \
  --dimensions 1:4 \
  --levels 1:4 \
  --dtypes float32 \
  --num-repeats 1 \
  --output test_results.jsonl
```

### 本実験実行（GPU）

```bash
# フル実行: d=1-50, level=1-50, 4 dtype
python3 -m python.experiment.smolyak.runner \
  --platform gpu \
  --gpu-indices 0,1,2,3 \
  --dimensions 1:51 \
  --levels 1:51 \
  --dtypes float16,bfloat16,float32,float64 \
  --num-repeats 3 \
  --output results.jsonl
```

---

## 関連ドキュメント

- **設計仕様**: [30_experiment_directory_structure.md](../../documents/conventions/python/30_experiment_directory_structure.md)
- **計画ノート**: [experiment_directory_planning.md](../../notes/knowledge/experiment_directory_planning.md)
- **実験運用規約**: [coding-conventions-experiments.md](../../documents/coding-conventions-experiments.md)
- **既存実験参考**: [smolyak_scaling README](../../experiments/functional/smolyak_scaling/README.md)

---

## 次のステップ

1. **コード実装** (Phase 1-2)
   - `case_generator.py`: ケース生成エンジン
   - `runner.py`: 実行エンジン
   - `analysis.py`: 結果分析
   - `smolyak/`: Smolyak 専用実装

2. **テスト作成** (Phase 1-2)
   - ユニットテスト（80% target）
   - smoke test（5 分以内完了）
   - fixture データ準備

3. **大規模実験実行** (Phase 3)
   - worktree 作成
   - フル実験実行（推定 1 周間）
   - 結果分析・frontier 生成

---

## 注記

- 実装は `main` ブランチで行い、results は別 branch で管理
- 長時間実験は worktree での実行を想定
- 最終結果は `notes/` へのメモ形式で `main` に持ち帰り

**作成日**: 2026-03-20  
**ステータス**: 設計段階 → 実装予定

