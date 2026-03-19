# ベンチマーク実行ガイド

## クイックスタート

```bash
# Light（最速検証）
cd /workspace/.worktrees/work-smolyak-improvement-20260318
python3 python/benchmark/functional/benchmark_smolyak_integrator.py --light

# Heavy（詳細分析）
python3 python/benchmark/functional/benchmark_smolyak_integrator.py --heavy \
  --output results/heavy_baseline.json

# Extreme（設計限界確認）
python3 python/benchmark/functional/benchmark_smolyak_integrator.py --extreme \
  --output results/extreme_baseline.json
```

## 各レベルの特性

### Light（軽量）
- **実行時間**: ~30 秒
- **対象**: d=1-8, level=1-5
- **用途**: デイリー検証、CI/CD pre-check
- **コマンド**: `--light`

### Heavy（中量）
- **実行時間**: ~5-10 分  
- **対象**: d=1-15, level=1-8
- **用途**: 改善検証、ベースライン記録、グラフ作成
- **コマンド**: `--heavy`

### Extreme（重量）
- **実行時間**: ~1 時間以上
- **対象**: d=1-20, level=1-10
- **用途**: 設計限界確認、3 ヶ月ごとの大規模分析
- **コマンド**: `--extreme`

## JSON 出力ヘッダ例

各レベル実行時に自動生成される JSON 形式:

```json
{
  "timestamp": "2026-03-19T12:34:57Z",
  "benchmark_level": "heavy",
  "suite": {
    "benchmarks": [
      {
        "benchmark": "initialization_scaling",
        "description": "次元ごとの初期化・積分時間のスケーリング",
        "max_dimension": 15,
        "fixed_level": 1,
        "results": [...]
      },
      {
        "benchmark": "level_refinement",
        "description": "Level 上昇時の初期化・積分コスト",
        "fixed_dimension": 3,
        "max_level": 8,
        "results": [...]
      },
      {
        "benchmark": "dtype_comparison",
        "description": "異なる精度での性能比較",
        "fixed_dimension": 8,
        "fixed_level": 3,
        "results": [...]
      }
    ]
  }
}
```

## Python API での使用

```python
from python.benchmark.functional.benchmark_smolyak_integrator import (
    run_light_benchmarks,
    run_heavy_benchmarks,
    run_extreme_benchmarks,
)

# プログラムから直接実行
light_results = run_light_benchmarks()
heavy_results = run_heavy_benchmarks()

# JSON 出力と同時実行
from python.benchmark.functional.benchmark_smolyak_integrator import (
    run_all_benchmarks,
)

run_all_benchmarks(level="heavy", output_file="my_results.json")
```

## 実装詳細

### ベンチマーク関数の入出力

各ベンチマーク関数は以下の構造で結果を返します:

```python
def run_light_benchmarks() -> dict[str, Any]:
    """
    Returns:
    {
        "benchmarks": [
            {
                "benchmark": str,       # "initialization_scaling" など
                "description": str,
                "results": [
                    {
                        "dimension": int,
                        "level": int,
                        "dtype": str,
                        "num_evaluation_points": int,
                        "num_terms": int,
                        "init_time": {"mean_sec": float, "std_sec": float, ...},
                        "integral_time": {"mean_sec": float, "std_sec": float, ...}
                    },
                    ...
                ]
            },
            ...
        ]
    }
    ```
    """
```

### パラメータカスタマイズ

`benchmark_initialization_scaling()` など個別関数の署名:

```python
def benchmark_initialization_scaling(
    max_dimension: int = 8,
    level: int = 1,
    num_trials: int = 2,
) -> dict[str, Any]:
    """次元スケーリング測定"""

def benchmark_level_refinement(
    dimension: int = 3,
    max_level: int = 5,
    num_trials: int = 2,
) -> dict[str, Any]:
    """レベル精製コスト測定"""

def benchmark_dtype_comparison(
    dimension: int = 4,
    level: int = 2,
    num_trials: int = 2,
) -> dict[str, Any]:
    """dtype 性能比較"""
```

カスタム測定:

```python
from python.benchmark.functional.benchmark_smolyak_integrator import (
    benchmark_initialization_scaling,
)

# d=1-20, level=2 の初期化時間測定
custom_result = benchmark_initialization_scaling(
    max_dimension=20,
    level=2,
    num_trials=3,
)
```

## トラブルシューティング

### エラー: `ImportError: cannot import name ...`
- ワークツリー内の Python path が正しく設定されているか確認
- Docker コンテナ内で実行していることを確認
- `sys.path` に `/workspace/.worktrees/work-smolyak-improvement-20260318` が含まれているか確認

### メモリ不足エラー
- Extreme レベルは d=20, level=10 まで測定 → メモリ使用量が増加
- Docker メモリ制限を確認: `docker stats`
- 必要に応じて `docker run --memory=4g ...`

### タイムアウト
- Extreme は 1 時間以上かかる場合がある
- `timeout` コマンドで制御: `timeout 3600 python3 ...`
- バックグラウンド実行推奨: `nohup python3 ... > log.txt 2>&1 &`

## CI/CD 統合例

### GitHub Actions

```yaml
name: Benchmark

on: [push]

jobs:
  light-benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Light Benchmark
        run: |
          cd /workspace/.worktrees/work-smolyak-improvement-20260318
          python3 python/benchmark/functional/benchmark_smolyak_integrator.py --light
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: light-benchmark
          path: results/light.json
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit に配置

cd /workspace/.worktrees/work-smolyak-improvement-20260318
python3 python/benchmark/functional/benchmark_smolyak_integrator.py --light || exit 1
```

## 参考リンク

- [ベンチマークレベル分析](./benchmark_levels_analysis.md)
- [ベンチマーク vs 実験](./benchmark_vs_experiment.md)
- [コーディング規約](../documents/conventions/python/20_benchmark_policy.md)

