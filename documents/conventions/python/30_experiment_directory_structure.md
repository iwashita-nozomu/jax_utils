# Python 実験ディレクトリ構造規約

**バージョン**: 1.0  
**作成日**: 2026-03-20  
**対象**: `python/experiment/` ディレクトリ

---

## 概要

`python/experiment/` は、実験コードを Python パッケージとして再利用・テスト可能な形で整理するディレクトリです。

- **`experiments/`** との違い: 長時間実行スクリプトと生成物を一箇所に置く場所
- **`python/experiment/`** の役割: 実験ロジックをモジュール化し、複数の実験スイートで再利用できるようにする

---

## 1. ディレクトリ構成

```
python/experiment/
├── __init__.py
├── README.md
├── case_generator.py          # ケース生成ロジック
├── protocols.py               # Protocol/Interface 定義
├── runner.py                  # 実験実行エンジン
├── analysis.py                # 結果分析・集計ロジック
├── utils.py                   # ユーティリティ関数
├── smolyak/                   # Smolyak 積分器専用モジュール
│   ├── __init__.py
│   ├── cases.py               # Smolyak 実験ケース定義
│   ├── runner_config.py       # 実行構成（次元・レベル・dtype レンジ）
│   ├── test_cases.py          # 検証用テストデータ
│   └── results_aggregator.py  # 結果集計・フィルタリング
├── tests/
│   ├── __init__.py
│   ├── test_case_generator.py
│   ├── test_runner.py
│   ├── test_analysis.py
│   └── test_smolyak_*.py      # Smolyak 専用テスト
└── fixtures/                  # テスト用の小規模ケース・期待結果
    ├── small_cases.json
    └── baseline_results.json
```

---

## 2. 各ファイルの責務

### `case_generator.py`

実験ケースを生成するロジック。

責務:
- `CaseSpec` (Protocol) を受け取り、具体的なケース列を生成
- 次元・レベル・dtype の直積を生成
- ユーザー指定の順序（例: dimension 昇順優先）に従う
- 各ケースに一意な case_id を付与

インターフェース:
```python
class CaseSpec(Protocol):
    dimensions: list[int]
    levels: list[int]
    dtypes: list[str]  # "float16", "bfloat16", "float32", "float64"
    case_ordering: Literal["dimension", "level", "dtype", "mixed"]

def generate_cases(
    spec: CaseSpec, /, seed: int = 42
) -> list[dict[str, Any]]:
    """ケース列を生成。各要素は {"case_id", "dimension", "level", "dtype", ...} を含む。"""
```

### `protocols.py`

型安全性のための Protocol 定義。

含む:
- `CaseSpec`: 実験ケース仕様
- `RunnerConfig`: 実行構成（resource, timeout, retry 等）
- `ExperimentResult`: 結果共通フォーマット
- `CaseRunner`: ケース実行インターフェース（Protocol）

### `runner.py`

実験実行エンジン。

責務:
- ケース列を受け取り、順序実行または並列実行
- 各ケースについて subprocess 実行
- 失敗分類（timeout, OOM, divergence 等）
- JSONL への逐次保存
- Host/child プロセス実行モード

### `analysis.py`

結果集計・フィルタリング。

責務:
- JSONL から final JSON 集計
- dtype 別・dimension 別・level 別の分類
- frontier 計算（precision/speed trade-off）
- 誤差統計

### `utils.py`

ユーティリティ関数。

含む:
- `dtype_to_size(dtype: str) -> int`: メモリサイズ推定
- `format_duration(sec: float) -> str`: 時間フォーマット
- `parse_case_ordering(s: str) -> list[str]`: ユーザー入力パース

### `smolyak/cases.py`

Smolyak 積分器専用のケース定義。

責務:
- Smolyak の `dimension`, `level`, `dtype` パラメータ定義
- テスト関数選択（exponential, polynomial, etc.）
- ベンチマーク vs 精度検証の区別

### `smolyak/runner_config.py`

Smolyak 実験の実行構成を定義。

含む:
```python
class SmolyakExperimentConfig:
    # 次元レンジ: 1-50
    dimension_range: tuple[int, int] = (1, 50)
    dimension_step: int = 1
    
    # レベルレンジ: 1-50
    level_range: tuple[int, int] = (1, 50)
    level_step: int = 1
    
    # dtype: 4 種類
    dtypes: list[str] = ["float16", "bfloat16", "float32", "float64"]
    
    # 実行パラメータ
    num_repeats: int = 3         # 各ケースの反復数
    num_accuracy_problems: int = 5  # 精度検証用テスト関数数
    timeout_seconds: int = 300   # ケース単位の timeout
    
    # ケース順序: "dimension" (outer loop) | "level" | "dtype" | "mixed"
    case_ordering: str = "dimension"
```

### `smolyak/results_aggregator.py`

Smolyak 専用の結果集計。

責務:
- 初期化時間の統計（mean, std, min, max）
- 積分実行時間の統計
- dtype ごとの性能比較
- frontier 計算（dimension × dtype）

---

## 3. テスト戦略

### Unit Tests (`python/tests/experiment/`)

各モジュールを独立テスト。

- `test_case_generator.py`: ケース生成ロジック
- `test_runner.py`: 実行エンジン（小規模ケースで）
- `test_analysis.py`: 集計ロジック

### Smoke Tests

軽量な実験実行テスト。

- `test_smolyak_small_cases.py`
  - d=1-3, level=1-3, 2 dtype のみで実行
  - 5 分以内に完了
  - 結果形式が valid JSON であることを確認

### Fixture

小規模期待結果。

- `fixtures/baseline_results.json`: d=2, level=2 での既知結果
- ケース実行後、baseline との差異を検証

---

## 4. 実装パターン

### ケース生成から実行まで

```python
from python.experiment.smolyak.cases import SmolyakExperimentConfig
from python.experiment.case_generator import generate_cases
from python.experiment.runner import ExperimentRunner

# 1. 構成を定義
config = SmolyakExperimentConfig(
    dimension_range=(1, 10),  # テスト用に縮小
    level_range=(1, 5),
    case_ordering="dimension",
)

# 2. ケース列を生成
cases = generate_cases(config)
print(f"Total cases: {len(cases)}")  # 10 * 5 * 4 = 200 ケース

# 3. 実行
runner = ExperimentRunner(output_file="results.jsonl")
final_results = runner.run_all(cases)

# 4. 結果分析
from python.experiment.analysis import ExperimentAnalyzer
analyzer = ExperimentAnalyzer(final_results)
print(analyzer.frontier_by_dtype())
```

### 大規模実験用 CLI

```bash
# CPU での限定実行（開発用）
python3 -m python.experiment.smolyak.runner \
  --platform cpu \
  --dimensions 1:10 \
  --levels 1:5 \
  --dtypes float32 \
  --output results_dev.jsonl

# GPU での完全実行（本実験）
python3 -m python.experiment.smolyak.runner \
  --platform gpu \
  --gpu-indices 0,1,2,3 \
  --dimensions 1:50 \
  --levels 1:50 \
  --dtypes float16,bfloat16,float32,float64 \
  --num-repeats 3 \
  --output results_full.jsonl
```

---

## 5. 大規模 Smolyak 実験の設計

### 次元範囲: 1-50

**根拠**:
- 現在のベンチマーク（Light/Heavy/Extreme）: d=1-20
- 設計限界確認で d=20 から指数爆発が観測
- 次元 50 は実用的な上限を大きく超え、アルゴリズム限界を確認

**段階**:
- d=1-10: 基本挙動（既知領域）
- d=11-30: 指数爆発中（主要分析対象）
- d=31-50: 合理的到達限界（コスト確認）

### レベル範囲: 1-50

**根拠**:
- 各次元で平均化の有用性を確認
- level=50 は評価点数 $\sim 10^{7}$ 以上（実用上限）
- 高精度 vs 計算コストの trade-off を可視化

### dtype: 4 種類

| dtype | 用途 | 期待コスト |
|-------|------|----------|
| float16 | 超低精度・高速実行 | 最小 |
| bfloat16 | 中精度・バランス型 | 低 |
| float32 | 標準精度 | 中 |
| float64 | 高精度基準 | 高 |

---

## 6. 結果ディレクトリ

実験実行時の結果格納位置（別途 worktree で実行）:

```
results/
├── raw/
│   ├── {timestamp}_run.jsonl      # 逐次保存結果
│   └── {timestamp}_metadata.json  # 実験メタデータ
├── aggregated/
│   ├── final.json                 # 完全集計結果
│   ├── by_dtype.json              # dtype 別集計
│   └── by_dimension.json          # 次元別集計
└── reports/
    ├── scaling_frontier.svg       # 次元 × dtype frontier
    ├── time_breakdown.svg         # init time vs integral time
    └── error_distribution.svg     # 誤差統計
```

---

## 7. main との統合方針

### コードの持ち帰り

- `python/experiment/` 全体を main へ統合
- テストは `python/tests/experiment/` へ
- 実行スクリプト例は `README.md` に記述

### 結果の持ち帰り

- 実行結果（JSONL、JSON）は results branch へ
- 分析レポート（SVG/HTML）は notes へのリンク
- 集計性能データ（frontier, statistics）は `notes/experiments/` へ

### メモの記述

- `notes/knowledge/experiment_directory_structure.md` に規約作成を記述
- `notes/experiments/smolyak_complete_scaling_experiment.md` に計画と進捗を記述
- branch 対応は `notes/branches/README.md` で管理

---

## 8. チェックリスト（実装時）

- [ ] `python/experiment/` ディレクトリ作成
- [ ] `__init__.py` で public API 定義
- [ ] `case_generator.py` 実装
- [ ] `protocols.py` の型定義
- [ ] `runner.py` 実装（JSONL 保存含む）
- [ ] `analysis.py` 実装
- [ ] `smolyak/` サブパッケージ実装
- [ ] ユニットテスト（`python/tests/experiment/`）
- [ ] smoke test（d=1-3, level=1-3）の実行
- [ ] 大規模実験用 worktree で実行可能な形に
- [ ] results branch 準備
- [ ] `notes/` メモ作成
- [ ] `documents/` への参照リンク張り

---

## 参考資料

- [実験環境の運用規約](../coding-conventions-experiments.md)
- [既存 Smolyak 実験](../../../experiments/functional/smolyak_scaling/README.md)
- [Experiment Runner 使い方](../../notes/knowledge/experiment_runner_usage.md)

