# コードレビュー: Smolyak 実験ランナー実装
**日付**: 2026-03-20  
**対象**: `experiments/smolyak_experiment/` (cases.py, runner_config.py, results_aggregator.py, run_smolyak_experiment.py)  
**レビュー視点**: コード品質、設計、保守性、規約遵守

---

## 概要

Smolyak 積分器の大規模スケーリング実験を行うための統合実験ランナーの実装。5 段階のサイズ設定、リソース見積もり、自動結果保存機能を備える。

**実装の強み**:
- モジュール化が適切で責務が明確
- 包括的なドキュメント（docstring）
- エラー検証が充実
- リソース認識型設計

---

## 詳細レビュー

### 1. **ドキュメント品質** ⭐⭐⭐⭐ (優秀)

#### 強み
- **modulle-level docstring**: 全ファイルに記載あり、目的が明確
- **関数レベルのdocstring**: `generate_cases()` に Parameters/Returns/Examples を完備
- **クラスドキュメント**: `SmolyakExperimentConfig` の Attributes セクション充実
- **複雑ロジックへの注記**: `estimate_case_resources()` で 2^level 飽和化の理由を記載

#### 改善余地
- `run_single_case()` のdocstring で JAX dtype 変換の例外処理について記載がない
- `SmolyakWorker` クラスに resource_estimate() のドキュメントが簡潔すぎる
- `run_experiment()` は大きな関数だが、内部ロジック（config選択、ワーカー実行、結果保存）の説明が込み入っている

**推奨**: `run_experiment()` を複数の小関数に分割し、各関数にドキュメント追加

---

### 2. **型ヒント** ⭐⭐⭐ (良好)

#### 強み
- `SmolyakExperimentConfig` で dataclass + Literal型を正しく使用
- 戻り値型多くの関数で指定
- Import: `from typing import Any, Literal`

#### 改善余地
- `run_single_case()` と `run_experiment()` で `dict[str, Any]` が多用
  - 構造化TypingDict の導入で型安全性向上可能
- `context_builder` ラムダが型注釈なし
- `results` list のジェネリック関数が `list[dict[str, Any]]`

**推奨**:
```python
from typing import TypedDict

class CaseDict(TypedDict):
    case_id: str
    dimension: int
    level: int
    dtype: str
    trial_index: int
    index: int

def generate_cases(config: SmolyakExperimentConfig) -> list[CaseDict]:
    ...
```

---

### 3. **エラーハンドリング** ⭐⭐⭐⭐ (優秀)

#### 強み
- `SmolyakExperimentConfig.validate()`: 全フィールド検証、明確なエラーメッセージ
- `run_single_case()`: 例外をキャッチし status/error に記録
- `estimate_case_resources()`: max_memory 制限で OOM 予防
- `run_experiment()`: size が無効な場合 sys.exit(1)

#### 改善余地
- `run_single_case()` で `getattr(jnp, dtype)` 失敗時のハンドリングなし
  - AttributeError が発生する可能性
- `generate_cases()` で config 検証を明示的に呼んでいない（呼び出し側に依存）
- JSON ファイルの書き込み失敗時の処理なし

**推奨**:
```python
def run_single_case(case: dict[str, Any]) -> dict[str, Any]:
    ...
    try:
        jax_dtype = getattr(jnp, dtype)
    except AttributeError:
        result["status"] = "FAILURE"
        result["error"] = f"Unsupported dtype: {dtype}"
        return result
    
    # その後の処理
```

---

### 4. **メモリ・リソース管理** ⭐⭐⭐ (良好)

#### 強み
- FullResourceCapacity / FullResourceEstimate を適切に統合
- リソース見積もり関数（estimate_case_resources）で粗な計算だが保守的
- 結果を JSON で逐次保存（メモリ溢れ対策）
- シーケンシャル実行で JAX マルチスレッド問題を回避

#### 改善余地
- **メモリ見積もりの精度**: 
  - max_points_multiplier = 2^level で level > 6 で飽和 → 大規模ケースで見積もりが不正確になる可能性
  - factor = 2.5 は経験値？ 根拠を Comment に明記すべき
  - 実際の Smolyak グリッド点数式 (2^(level+1) - 1)^dimension との乖離が大きい

- **リソース容量の固定**: 16GB ホストメモリ、ワーカー 4 は hardcode
  - `FullResourceCapacity.from_system()` で自動検出すべき

- **GPU サポートなし**: GPU device 空のまま、future 拡張を想定している？

**推奨**:
```python
# リソース見積もりをより正確に
def estimate_case_resources(case: dict[str, Any]) -> FullResourceEstimate:
    dimension = case["dimension"]
    level = case["level"]
    dtype = case["dtype"]
    
    dtype_bytes = {
        "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8
    }
    bytes_per_element = dtype_bytes.get(dtype, 4)
    
    # 実際の Smolyak グリッド評価点数
    # 安全な上限を計算
    num_points_per_dim = (2 ** (min(level, 8) + 1)) - 1
    num_points_upper = num_points_per_dim ** min(dimension, 3)
    
    # スケーリング因子（経験値、実測に基づく）
    # JAX JIT compilation cache + array allocation overhead
    memory_factor = 3.0  # 3x for buffers + overhead
    
    host_memory_bytes = int(num_points_upper * bytes_per_element * memory_factor)
    
    # 安全化：最大を 2GB に
    MAX_MEMORY = 2 * (1024 ** 3)
    host_memory_bytes = min(host_memory_bytes, MAX_MEMORY)
    
    return FullResourceEstimate(
        host_memory_bytes=host_memory_bytes,
        gpu_count=0,
        gpu_memory_bytes=0,
        gpu_slots=1,
    )

# リソース容量を自動検出
resource_capacity = FullResourceCapacity.from_system(
    max_workers=4,  # or detect from system
    gpu_max_slots=1
)
```

---

### 5. **可読性・名前付け** ⭐⭐⭐⭐ (優秀)

#### 強み
- 関数・変数名が明確：`run_single_case`, `aggregate_by_dimension`
- 定数に意味のある名前：`configs_by_size`
- ループ変量の命名が適切
- 出力フォーマット（print 文）が視覚的に整理

#### 改善余地
- `SmolyakWorker` クラスが `Worker` Protocol を実装するが、Protocol 明示がコメントのみ
- `context_builder` ラムダが inline すぎる → 命名された関数へ
- `results` list の各要素が部分的に Optional フィールド持つ → スキーマ不明確

**推奨**:
```python
def make_context(case: dict[str, Any]) -> TaskContext:
    """各ケース用のコンテキストを作成"""
    return {"case_id": case["case_id"]}

scheduler = StandardFullResourceScheduler.from_worker(
    resource_capacity=resource_capacity,
    cases=case_list,
    worker=worker,
    context_builder=make_context,
)
```

---

### 6. **テスト可能性** ⭐⭐⭐ (良好)

#### 強み
- `generate_cases()` 純粋関数、単体テスト容易
- `SmolyakExperimentConfig.validate()` デカップリング可能
- `results_aggregator.*` 関数が副作用なし

#### 改善余地
- `run_single_case()` が SmolyakIntegrator に依存 → 被積分関数が暗黙的
- `run_experiment()` は sys.exit() や print() に依存 → 単体テスト困難
- config のデフォルト値が configs_by_size に hardcode されている

**推奨**:
```python
# テスト用に config を引数化
def run_experiment(config: SmolyakExperimentConfig, 
                   output_dir: Path | None = None) -> dict[str, Any]:
    """
    実験を実行して結果を返す。
    
    Parameters
    ----------
    config : SmolyakExperimentConfig
    output_dir : Path | None
        結果保存先。None の場合は標準出置
    
    Returns
    -------
    dict[str, Any]
        実験結果
    """
    ...
```

---

### 7. **パフォーマンス** ⭐⭐⭐ (良好)

#### 強み
- `generate_cases()` で O(N) 線形な直積生成
- `aggregate_by_*()` で defaultdict 使用（O(N)）
- 結果を JSON で逐次保存し、メモリ蓄積なし

#### 改善余地
- `run_experiment()` で全ケースをループ
  - 10 万ケースなら loop overhead が無視できるか確認
  - 表示（print）の頻度調整可能（現在は 10 ケースごと）

- `SmolyakWorker.__call__()` で毎回 `run_single_case()` 呼ぶ
  - 被積分関数の JIT compile がケースごとに再実行される可能性あり
  - → キャッシング戦略を検討

**推奨**: ケース 500+ では `print()` 頻度を 100 ごと、1000+ では 500 ごとに

---

### 8. **設計・アーキテクチャ** ⭐⭐⭐⭐ (優秀)

#### 強み
- **責務分離**: cases → runner_config → run_experiment の依存関係が明確
- **モジュル化**: results_aggregator の統計機能が独立
- **拡張性**: worker protocol で将来の別 worker 実装余地あり
- **リソース認識**: StandardFullResourceScheduler の統合

#### 改善余地
- `SmolyakWorker` が `resource_estimate()` を提供するが、実際の並列スケジューリングは使われていない（シーケンシャル実行）
  - リソース認識設計が活かされていない
  - コメントで「JAX fork() 非互換」と記載あるが、将来の改善で並列化可能性を記述すべき

- 被積分関数が hardcode されている
  - 外部化して plug-in 可能にする設計も検討

**推奨**:
```python
class SmolyakWorker(Worker):
    """Smolyak リソース認識ワーカー"""
    
    def __init__(self, integrand=None):
        self.integrand = integrand or self._default_integrand
    
    @staticmethod
    def _default_integrand(x: jnp.ndarray) -> jnp.ndarray:
        """デフォルト被積分関数: f(x) = sum(x_i^2)"""
        return jnp.sum(x**2, axis=-1)
    
    def __call__(self, case, context):
        result = run_single_case(case, self.integrand)
        context["result"] = result
```

---

### 9. **規約遵守【重要】** ⭐⭐⭐ (良好)

#### チェック項目

| 項目 | 状態 | 備考 |
|------|------|------|
| import パス | ✅ `jax_util.*` で統一 | OK |
| PYTHONPATH 設定 | ✅ sys.path.insert | OK |
| Docker 対応 | ✅ Dockerfile 準拠実装 | 追加パッケージなし |
| ドキュメント位置 | ✅ `documents/` に指定なし | experiments 段階で OK |
| worktree 規約 | ⚠️ 未記載 | 実験スクリプト先頭にコメント追加すべき |
| 結果保存ルール | ✅ `results/{size}/` に集約 | OK |
| JSON 形式 | ✅ config/results/summary 含む | OK |

#### 改善项目
1. **実験スクリプト先頭にコメント追加**（規約 coding-conventions-experiments.md §5）
   ```python
   # results branch: results/smolyak-experiment-201
   # Generated results should be committed to this branch, not main.
   ```

2. **`experiments/smolyak_experiment/results/.gitkeep` が必要**
   - ディレクトリ構造保持のため

3. **`notes/experiments/smolyak_experiment.md` を main に作成**
   - 実験目的、期待値、主要設定を記載

---

## 出力形式評価

### JSON スキーマの適切性

✅ **強み**:
- config, resource_capacity, results, summary が適切に分離
- 再現性のため config 全体を保存
- タイムスタンプで複数実行結果を識別可能

⚠️ **改善余地**:
- metadata（worktree/branch/commit SHA）を含めるとよい
  ```json
  "metadata": {
    "timestamp": "2026-03-20T...",
    "worktree": "/workspace/.worktrees/work-smolyak-improvement-20260318",
    "branch": "work/smolyak-improvement-20260318",
    "commit": "0d78ea0..."
  }
  ```

---

## 動作確認結果评估

| テスト | 結果 | 発見事項 |
|--------|------|----------|
| smoke test (d=1-3, level=1-2) | ✅ 6/6 成功 | 初期化時間: d=1 で 165ms は妥当 |
| medium (d=1-10, level=1-10 x2) | 🟡 実行中 | 時間がかかるが expected |

---

## 優先度別改善提案

### 🔴 **必須（実装品質に影響**）
1. `getattr(jnp, dtype)` の例外処理追加
2. `generate_cases()` 内部で `config.validate()` 呼び出し
3. 実験スクリプト先頭に results branch コメント追加

### 🟡 **推奨（保守性向上）**
1. TypedDict で case/result スキーマ明示化
2. memory 見積もり根拠をコード Comment に記載
3. リソース容量を `from_system()` で自動検出
4. `run_experiment()` の関数分割

### 🟢 **将来的（デザイン改善）**
1. 被積分関数の外部化
2. 並列スケジューリングの再実装（JAX fork() 問題解決後）
3. GPU サポート追加
4. 結果の progress JSONL 出力

---

## 総評

**スコア: 7.8/10**

### 総合評価
- **実装成熟度**: ✅ Production-ready
- **可読性**: ✅ 優秀
- **拡張性**: ✅ 良好
- **エラー処理**: ⚠️ 部分的改善余地
- **規約準拠**: ✅ 良好

本実装は、Smolyak スケーリング実験の for 統合実験ランナーとして**十分な成熟度**を達成している。

上記の 🔴 必須項目（3 件）を修正すれば、本番運用に支障なし。推奨項目は反復的改善でも OK。

---

## 参考資料

- 規約: `documents/coding-conventions-experiments.md` §5,6,7
- API: `python/jax_util/experiment_runner/resource_scheduler.py`
- 前回実装: `Archive/experiments/smolyak_experiment_smoke/smoke_test_results.json`
