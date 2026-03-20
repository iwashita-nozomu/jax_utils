# Python 実験ディレクトリ構造：規約作成（2026-03-20）

**作成日**: 2026-03-20  
**背景**: Smolyak 積分器の大規模実験（d=1-50, level=1-50, 4 dtype）を計画・実行するため、実験ロジックをモジュール化して再利用可能にする

---

## 1. 問題設定

### 現状

- `experiments/functional/smolyak_scaling/` 
  - 実験実行スクリプトが 1 つのファイル内に集中
  - テスト関数、ケース生成、実行、集計が分離されていない
  - 複数の異なる実験スイートからの再利用が困難

- `python/benchmark/functional/`
  - 軽量ベンチマーク（Light/Heavy/Extreme）は成功
  - ただし単発実行で、大規模実験向けではない

### 需要

- 大規模スケーリング実験が必要：d=1-50, level=1-50, 4 dtype
- 総ケース数：50 × 50 × 4 = 10,000 ケース
- 各ケースを複数回繰返（num_repeats=3）
  - 総実行数：30,000 単位タスク
- 長時間実行（推定 1-3 週間）が想定されるため、infrastructure が必要

---

## 2. 設計方針

### 目標

- **再利用可能**: ケース生成・実行・分析ロジックを分離
- **テスト可能**: 軽量 smoke test で動作確認
- **スケーラブル**: CPU/GPU 両対応、並列実行
- **監視可能**: JSONL 逐次保存で途中失敗に対応
- **統合可能**: main へ持ち帰るコード・test・ドキュメント

### 採用構成

`python/experiment/` ディレクトリ新設:

```python
# モジュール化
case_generator.py          # ケース生成（dimension × level × dtype の直積）
runner.py                  # 実行エンジン（subprocess 管理・JSONL 保存）
analysis.py                # 結果分析（frontier, 統計）
protocols.py               # Protocol 定義（型安全性）

# Smolyak 専用
smolyak/
  cases.py                 # Smolyak パラメータ範囲定義
  runner_config.py         # 実験構成（d=1-50, level=1-50 等）
  results_aggregator.py    # dtype 別・dimension 別集計
```

### 大規模実験用 worktree

- worktree 名: `experiment-smolyak-complete-20260320` (予定)
- 長時間実行用（1-3 週間）
- results branch で JSONL/final JSON を保存
- main へは `notes/` のメモと集計 JSON のみ持ち帰り

---

## 3. 実装構想

### フェーズ 1: 基礎モジュール（当ノート対象 = 規約作成のみ）

**このフェーズでやること（規約）**:
- `documents/conventions/python/30_experiment_directory_structure.md` 作成 ✅
- `python/experiment/` ディレクトリ規約を定義
- テスト戦略を記述
- 大規模 Smolyak 実験の設計パラメータを確定

**このフェーズでやらないこと**:
- 実装コード
- 実験実行
- 結果集計

### フェーズ 2: 実装（別日程）

- `python/experiment/` 実装開始
- ユニットテスト・smoke test
- 小規模実験で動作確認

### フェーズ 3: 大規模実験実行（別日程）

- worktree 作成
- d=1-50, level=1-50, 4 dtype で実行
- 結果集計・レポート生成

---

## 4. パラメータ設計の根拠

### 次元範囲: 1-50

**段階分析**:

| 段階 | 範囲 | 理由 |
|------|------|------|
| 小規模 | d=1-10 | 既知領域・検証用 |
| 遷移 | d=11-20 | ベンチマーク（Extreme）で指数爆発開始が観測 |
| 指数域 | d=21-35 | 積分器限界への接近 |
| 超大規模 | d=36-50 | アルゴリズム根本限界確認 |

**期待時間コスト**（float64, level=3, 1 回):
- d=10: ~1秒
- d=20: ~3秒（指数爆発から ~50-100倍）
- d=30: ~1-2分（推定）
- d=40: ~1-2時間（推定、実行不可能の可能性）
- d=50: 不明（タイムアウト予想）

### レベル範囲: 1-50

**段階分析**:

| レベル | 評価点数 | 用途 |
|--------|---------|------|
| l=1-5 | ~1-620 | 基本性能確認 |
| l=6-15 | ~10^3-10^5 | 精度-計算コスト分析 |
| l=16-30 | ~10^5-10^8 | 大規模推定 |
| l=31-50 | >10^8 | 理論限界確認 |

**期待時間コスト**（float64, d=3, 1 回):
- level=10: ~20ms
- level=20: ~200ms
- level=30: ~数秒
- level=40: ~数十秒
- level=50: ~数分

### dtype: 4 種類を選択

```
float16   (半精度・最高速)
bfloat16  (Google Brain Float・中速)
float32   (単精度・標準)
float64   (倍精度・基準)
```

**根拠**:
- precision-speed trade-off を全段階で可視化
- 組込デバイス（float16）での実用可能性確認
- 機械学習標準（bfloat16）の相性確認

---

## 5. 大規模実験の全体スケジュール（見積）

### ケース数

- dimension: 50 個
- level: 50 個
- dtype: 4 個
- 合計: 10,000 ケース

### 実行コスト（推定）

各ケースを 3 回繰返：30,000 単位タスク

**CPU alone** (non-parallel):
- 平均 1 ケース &approx; 1 秒 × 3 = 3 秒
- 総時間: 30,000 × 3 秒÷3600 ≈ **25 時間**
- ただし、d&geq;30 は timeout の可能性

**GPU 4 並列** (4 GPU):
- 同時に 4 ケース実行
- 総時間: 25 時間 ÷ 4 ≈ **6-7 時間**
- ただし、ケース順序（dimension outer）でメモリ効率が変わる

**推奨戦略**:
- Phase 1: d=1-20, level=1-20 で試行（フル + 4 GPU ≈ 1 時間）
- Phase 2: d=21-35 で追加分析（オプション）
- Phase 3: d=36-50 はスキップまたは選別実行

---

## 6. 設計決定

### ケース生成順序

**選択: dimension outer loop**

```
for d in 1..50:
  for level in 1..50:
    for dtype in [f16, bf16, f32, f64]:
      run(d, level, dtype)
```

**理由**:
- 同じ dimension では初期化コスト はほぼ同じ（キャッシュ効率）
- level 昇順なら GPU メモリ使用量が単調増加
- dtype 切り替えは計算内容に影響なし

### 失敗分類

各ケース失敗は以下で分類：

```python
class FailureKind(Enum):
    SUCCESS = "success"          # 正常完了
    TIMEOUT = "timeout"          # 指定時間内に完了しない
    OOM = "out_of_memory"        # メモリ不足
    DIVERGENCE = "divergence"    # 積分値が NaN/Inf
    NUMERICAL = "numerical_error" # 精度誤差＞threshold
```

### メタデータ保存

各実験実行時にメタデータを記録：

```json
{
  "timestamp": "2026-03-20T10:30:00Z",
  "git_branch": "experiment-smolyak-complete-20260320",
  "git_commit": "abc1234",
  "platform": "gpu",
  "gpu_indices": [0, 1, 2, 3],
  "case_ordering": "dimension",
  "config": {
    "dimensions": [1, 2, ..., 50],
    "levels": [1, 2, ..., 50],
    "dtypes": ["float16", "bfloat16", "float32", "float64"],
    "num_repeats": 3,
    "num_accuracy_problems": 5,
    "timeout_seconds": 300
  }
}
```

---

## 7. 規約作成と main との関係

### 規約作成の位置付け

このノートおよび `documents/conventions/python/30_experiment_directory_structure.md` は、

- **設計段階**: ケースを実行する前に構造を決定
- **実装ガイド**: 実装者が参照すべき仕様
- **統合基準**: main マージ時の受入条件

### 持ち帰り時のチェックリスト

実装完了後、main へマージする前に確認:

- [ ] `documents/conventions/python/30_experiment_directory_structure.md` が最新版か？
- [ ] `python/experiment/` の実装が仕様を満たしているか？
- [ ] テストカバレッジ ≥ 80%か？
- [ ] smoke test （d=1-3, level=1-3）が通るか？
- [ ] 実際のベンチマーク結果との比較で一貫性があるか？
- [ ] `notes/experiments/` メモで実験計画が記述されているか？
- [ ] 結果 branch での保存構造が計画に従い、参照リンクが main で有効か？

---

## 8. 次のステップ

### 短期（今週）

- [x] 規約ドキュメント作成（`30_experiment_directory_structure.md`）
- [x] このノート作成
- [ ] チームレビュー・反応収集

### 中期（来週）

- [ ] `python/experiment/` 実装開始
- [ ] ユニットテスト・smoke test
- [ ] worktree を使った試験実行（d=1-3, level=1-3）

### 長期（2-3 週間）

- [ ] 大規模実験実行（d=1-50 フル）
- [ ] 結果分析・frontier 生成
- [ ] main マージ
- [ ] 論文・レポート作成

---

## 参考

- [実験環境の運用規約](../../../documents/coding-conventions-experiments.md)
- [Experiment Runner Usage](./experiment_runner_usage.md)
- [既存 Smolyak 実験 README](../../../experiments/functional/smolyak_scaling/README.md)

