# Smolyak スケーリング実験

**日付**: 2026-03-20  
**ステータス**: 実装完了（必須改善 3 件の対応中）  
**関連 worktree**: `work-smolyak-improvement-20260318`  
**関連 branch**: `work/smolyak-improvement-20260318` → `results/smolyak-experiment-201` へ統合予定

---

## 実験目的

Smolyak 積分器の**スケーリング特性分析**  
- 次元 d = 1～50, レベル ℓ = 1～50, データ型 4 種（float16/bfloat16/float32/float64）
- 初期化時間・積分実行時間の測定
- 精度伝播特性（dtype별 誤差傾向）の観測

**総ケース数**: 50 × 50 × 4 × 3 (trials) = **30,000 ケース**

---

## 実装構成

### ディレクトリ構成
```
experiments/smolyak_experiment/
├── cases.py                    # ケース生成 + リソース見積もり
├── runner_config.py            # 実験構成管理 (dataclass)
├── results_aggregator.py       # 結果集計・分析
├── run_smolyak_experiment.py   # 本実験ランナー（entry point）
├── README.md
├── __init__.py
└── results/                    # 生成物ディレクトリ
    ├── smoke/
    ├── small/
    ├── medium/
    ├── large/
    └── full/
```

### モジュール責務
- **cases.py**: ケース直積生成、リソース見積もり（host memory）
- **runner_config.py**: SmolyakExperimentConfig（5 段階サイズ設定）
- **results_aggregator.py**: by_dtype / by_dimension / by_level / statistics
- **run_smolyak_experiment.py**: メインランナー（実行、結果保存）

---

## 実行方法

### smoke test（開発用）
```bash
python3 experiments/smolyak_experiment/run_smolyak_experiment.py --size smoke
# d=1-3, level=1-2, float32 のみ
# 6 ケース、実行時間 ~3s
```

### 小～大規模実験
```bash
python3 experiments/smolyak_experiment/run_smolyak_experiment.py --size small
python3 experiments/smolyak_experiment/run_smolyak_experiment.py --size medium
python3 experiments/smolyak_experiment/run_smolyak_experiment.py --size large
python3 experiments/smolyak_experiment/run_smolyak_experiment.py --size full
```

---

## 出力仕様

### 標準出力（stdout）

実行時に以下の情報を逐次表示：

```
======================================================================
Smolyak Experiment - SMOKE
======================================================================

Config:
  Dimensions: 1-3
  Levels: 1-2
  dtypes: ['float32']
  Trials per case: 1
  Total cases: 6
  Total tasks: 6

Generated 6 cases

Resource Capacity:
  Max workers: 4
  Host memory: 16.0 GB
  GPU devices: 0

Running cases (sequential execution with resource awareness):

  [   6/   6] elapsed=    2.7s  throughput=  2.2 cases/s

======================================================================
Results Summary
======================================================================

Successful: 6 / 6
Failed: 0 / 6
Total elapsed time: 2.7 s (0.0 m)
Throughput: 2.2 cases/s

Initialization time statistics (ms):
  Min: 1.88
  Max: 319.23
  Mean: 59.92
  Std: 116.05

Integration time statistics (ms):
  Min: 132.17
  Max: 303.99
  Mean: 196.02

Initialization time by dimension (ms):
  d= 1: mean= 165.75, min=  12.27, max= 319.23
  d= 2: mean=   7.20, min=   1.88, max=  12.52
  d= 3: mean=   6.81, min=   2.21, max=  11.40

Results saved to: .../experiments/smolyak_experiment/results/smoke/results_1773985645.json
```

### JSON 出力ファイル

**パス**: `experiments/smolyak_experiment/results/{size}/results_{timestamp}.json`

**スキーマ**:

```json
{
  "config": {
    "min_dimension": 1,
    "max_dimension": 3,
    "min_level": 1,
    "max_level": 2,
    "dtypes": ["float32"],
    "num_trials": 1,
    "timeout_seconds": 300.0,
    "device": "cpu",
    "num_accuracy_problems": 9,
    "coeff_start": -0.55,
    "coeff_stop": 0.65,
    "total_cases": 6,
    "total_tasks": 6,
    "experimental": false
  },
  "resource_capacity": {
    "max_workers": 4,
    "host_memory_bytes": 17179869184,  // 16 GB
    "gpu_devices": 0
  },
  "results": [
    {
      "case_id": "d1_l1_float32_t0",
      "dimension": 1,
      "level": 1,
      "dtype": "float32",
      "trial_index": 0,
      "status": "SUCCESS",
      "init_time_ms": 121.33,
      "integrate_time_ms": 255.14,
      "num_evaluation_points": 3,
      "error": null
    },
    // ... 5 件分
  ],
  "summary": {
    "total": 6,
    "success": 6,
    "failed": 0,
    "elapsed_seconds": 2.7,
    "throughput_cases_per_sec": 2.22
  }
}
```

### 出力アイテム詳細

#### 各ケースの結果（results[]）

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `case_id` | str | ケース識別子（d{dim}_l{level}_{dtype}_t{trial}） |
| `dimension` | int |空間次元 |
| `level` | int | Smolyak レベル |
| `dtype` | str | データ型（float16/bfloat16/float32/float64） |
| `trial_index` | int | 試行番号（0 から num_trials-1） |
| `status` | str | 実行結果（SUCCESS/FAILURE） |
| `init_time_ms` | float | Smolyak 積分器初期化時間（ミリ秒） |
| `integrate_time_ms` | float | 積分実行時間（ミリ秒、2 回測定の平均） |
| `num_evaluation_points` | int | グリッド上の評価点数 |
| `error` | str \| null | エラーメッセージ（失敗時のみ） |

#### 統計サマリー（summary）

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `total` | int | 総実行ケース数 |
| `success` | int | 成功したケース数 |
| `failed` | int | 失敗したケース数 |
| `elapsed_seconds` | float | 全実行時間（秒） |
| `throughput_cases_per_sec` | float | スループット（ケース/秒） |

#### リソース容量（resource_capacity）

| フィールド | 型 | 説明 |
|-----------|-----|------|
| `max_workers` | int | 最大並列ワーカー数（現在未使用、将来拡張用） |
| `host_memory_bytes` | int | ホストメモリ容量（バイト） |
| `gpu_devices` | int | GPU デバイス数（0 = CPU only） |

---

## 主要な観測結果（smoke test）

✅ **Smoke test 実行確認**:
- 全 6 ケース成功
- 次元効果が明確
  - d=1: 初期化時間で顕著な差（165ms 平均）→ JIT compile overhead
  - d=2,3: 安定（~7ms）
- 被積分関数評価点数: d=1, l=1 で 3 点、増加に伴い指数関数的増加

---

## 今後の課題（優先度順）

### 🔴 **必須対応（実装品質）**

#### 1. JAX dtype 変換の例外処理
```python
# run_single_case() で
try:
    jax_dtype = getattr(jnp, dtype)
except AttributeError:
    result["status"] = "FAILURE"
    result["error"] = f"Unsupported dtype: {dtype}"
    return result
```
**理由**: dtype が無効な場合の例外を予防  
**優先度**: P0（エラー現象の封じ込め）

#### 2. generate_cases() 内での config 検証
```python
def generate_cases(config: Any) -> list[dict[str, Any]]:
    config.validate()  # 追加
    ...
```
**理由**: 呼び出し側の検証忘れを防ぐ  
**優先度**: P0（安全性）

#### 3. 実験スクリプト先頭に results branch コメント
```python
# results branch: results/smolyak-experiment-201
# Generated results should be stored in this branch, not main.
```
**理由**: 規約 [coding-conventions-experiments.md](documents/coding-conventions-experiments.md) §5 に準拠  
**優先度**: P0（ガバナンス）

---

### 🟡 **推奨改善（保守性・拡張性）**

#### 4. TypedDict でスキーマ明示化
```python
from typing import TypedDict

class CaseDict(TypedDict):
    case_id: str
    dimension: int
    level: int
    dtype: str
    trial_index: int
    index: int
```
**理由**: 型安全性向上、IDE サポート改善  
**優先度**: P1（開発体験）

#### 5. メモリ見積もり根拠を Comment に明記
```python
# メモリ見積もり計算
# - Smolyak グリッド点数: 粗い上限値で保護
# - 係数 2.5: 入出力 + 中間計算 + JAX JIT cache（経験値）
# - 最大値制限 1GB: smoke/small 実験での OOM 予防
```
**理由**: 将来の tuning/デバッグ時の根拠明確化  
**優先度**: P2（ドキュメント）

#### 6. リソース容量を from_system() で自動検出
```python
resource_capacity = FullResourceCapacity.from_system(
    gpu_max_slots=1
)
```
**理由**: 環境依存性を減らす  
**優先度**: P2（ポータビリティ）

#### 7. run_experiment() を関数分割
- `_select_config(size)` - config 選択
- `_run_cases(case_list, worker)` - 実行
- `_save_results(results, output_dir)` - 保存

**理由**: テスト可能性向上、単一責務化  
**優先度**: P2（テスト駆動開発）

---

### 🟢 **将来的な拡張（デザイン改善）**

#### 8. 被積分関数の外部化
```python
class SmolyakWorker(Worker):
    def __init__(self, integrand_fn=None):
        self.integrand = integrand_fn or self._default_integrand
```
**利点**: 被積分関数を plug-in 可能に  
**優先度**: P3（flexibility）

#### 9. 並列スケジューリングの再実装
- 現在: シーケンシャル実行（JAX fork() 非互換のため）
- 将来: thread pool ベースの実装、または `jax.config.update("jax_enable_custom_prng", True)` で fork 回避

**優先度**: P3（パフォーマンス最適化）

#### 10. GPU サポート
```python
resource_capacity = FullResourceCapacity.from_system(
    device="gpu"  # または自動検出
)
```
**優先度**: P3（スケーラビリティ）

#### 11. Progress JSONL 出力
- ケース完了時点で逐次 JSONL へ追記
- 長時間実験で途中結果確認可能に

**優先度**: P3（監視可視化）

---

## 実装インデックス

| ファイル | 行数 | 役割 |
|---------|------|------|
| `cases.py` | 130 | ケース生成 + リソース見積もり |
| `runner_config.py` | 100 | 構成管理（dataclass） |
| `results_aggregator.py` | 105 | 統計集計 |
| `run_smolyak_experiment.py` | 363 | メインランナー |

---

## テスト状況

| ケース | 実行 | 結果 |
|--------|------|------|
| smoke | ✅ 実施 | 6/6 成功 |
| small | ✅ 実装完了 | 未実施（時間制約） |
| medium | 🟡 実装完了 | 実行中…（timeout で中断、150/400 ケース） |
| large | ✅ 実装完了 | 未実施 |
| full | ✅ 実装完了 | 未実施 |

---

## 関連リソース

- **コードレビュー**: [reviews/CODE_REVIEW__smolyak_experiment_201.md](reviews/CODE_REVIEW__smolyak_experiment_201.md)
- **規約**: 
  - [documents/coding-conventions-experiments.md](documents/coding-conventions-experiments.md)
  - [documents/coding-conventions-project.md](documents/coding-conventions-project.md)
- **実装**: `experiments/smolyak_experiment/`
- **生成物サンプル**: `experiments/smolyak_experiment/results/smoke/results_1773985645.json`

---

## 次のアクション

1. **必須改善 3 件**を実施（P0）
2. **smoke test を再実行**して出力確認
3. **small test を完全実行**
4. 推奨改善 4 件を反復的に実施（P1-P2）
5. 本実験実行（full, 30,000 ケース）

**Idea: 実験結果の可視化**
- 初期化時間 vs 次元の 2D プロット
- dtype 別の誤差傾向
- level スケーリング曲線

**Consideration: 実行時間の最適化**
- Smoke → Small → Medium と段階的に検証
- Medium 実行時間が 150s/150 ケース = 1s/ケース → full では 30,000s = 8.3 時間
- 並列化実装で大幅削減可能（estimate: 4 時間 → 1 時間）
