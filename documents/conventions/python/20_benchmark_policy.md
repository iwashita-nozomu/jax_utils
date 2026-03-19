"""Benchmark ポリシー

このドキュメントは、Python プロジェクト内のベンチマーク実装方針。

## 対象

- `python/benchmark/` 配下のベンチマークスクリプト
- 中程度の性能計測（数秒～分スケール）

## 定義

**ベンチマーク**: 単一環境での再現可能な性能計測。

- 実装変更の前後比較
- 次元・レベル・dtype による基本的なスケーリング
- サブモジュール間の相対性能
- CI/CD での性能劣化検知

## ディレクトリ構造

```
python/benchmark/
├── functional/
│   ├── __init__.py
│   ├── benchmark_smolyak_integrator.py   ← Smolyak 専用
│   ├── benchmark_monte_carlo_*.py        ← 他の方法との比較
│   └── results/                          ← JSON output（gitignore）
└── README.md                             ← benchmark 全体の説明
```

## 実装スタイル

### 関数命名
- `benchmark_*()` で始まる関数
- 戻り値は `dict[str, Any]` で JSON に変換可能
- 結果には timestamp, 条件情報を含める

### 実行時間
- 全ベンチマーク合算で数分以内（理想：1-2分以内）
- 個別テストは秒～数秒スケール
- timeout 対応は不要（対象外）

### 出力フォーマット
- JSON で統一（後段の自動分析に対応）
- 構造は `dict` で JSON serializable
- 時刻は ISO 8601 UTC

### 条件記録
- `dimension`, `level`, `dtype` など条件を明記
- `num_evaluation_points`, `num_terms` など derived quantity も追加
- 計測統計（mean, std, min, max）

## 実装例

```python
def benchmark_initialization_scaling() -> dict[str, Any]:
    """初期化時間の dimension スケーリング。"""
    results = []
    for dimension in range(1, 9):
        result = _benchmark_integration_time(
            dimension=dimension,
            level=1,
            dtype=jnp.float64,
            num_trials=2,
        )
        results.append(result)
    
    return {
        "benchmark": "initialization_scaling",
        "description": "...",
        "fixed_level": 1,
        "results": results,
    }

def run_all_benchmarks(output_file: Path | None = None):
    """すべてのベンチマークを実行。"""
    results = {
        "timestamp": time.strftime(...),
        "benchmarks": [
            benchmark_initialization_scaling(),
            benchmark_level_refinement(),
            # ...
        ],
    }
    
    if output_file:
        json.dump(results, open(output_file, "w"))
```

## 実行方法

### コマンドライン
```bash
# デフォルト import-check のみ
cd /workspace/.worktrees/work-smolyak-improvement-*
python3 python/benchmark/functional/benchmark_smolyak_integrator.py

# JSON に出力
python3 python/benchmark/functional/benchmark_smolyak_integrator.py /tmp/result.json
```

### スクリプト内から
```python
from python.benchmark.functional.benchmark_smolyak_integrator import run_all_benchmarks

results = run_all_benchmarks(output_file="benchmark_result.json")
```

## 結果の活用

### 開発中
- 実装変更の前後で実行して効果を定量化
- `git bisect` との組み合わせで regression を特定

### CI/CD
- 定期実行（push ごと or nightly）して性能劣化を検知
- タイムアウト値設定で catastrophic case を早期発見

### ドキュメント
- ベンチマーク結果を `notes/knowledge/` に定期更新
- 傾向スナップショットを theme notes に

## Benchmark vs Experiment

**Benchmark の対象外**:
- 複数条件の組み合わせ探索（→ experiment）
- 精度限界や収束性の詳細分析（→ experiment）
- GPU メモリ量や通信時間（→ profiling 専用ツール）

詳細は [benchmark_vs_experiment.md](../../notes/knowledge/benchmark_vs_experiment.md) を参照。

## コーディング規約の適用

ベンチマークスクリプトは以下を守ります：

- 型アノテーション: `dict[str, Any]` など明示
- Comments: 各関数の責務を冒頭に一行英文
- Naming: snake_case、`benchmark_*` prefix
- Imports: jax_util は `from jax_util.functional import ...` で明示
- Docstring: 1行サマリ＋Parameters+Returns

ベンチマーク自体はテストではなく、性能計測ツールのため：
- `assert` は使わない（失敗するより結果を出す）
- 例外は上げてもよい（異常条件の早期検知）
- exception handling は必要最小限
"""
