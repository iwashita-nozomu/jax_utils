# Skill: `run-experiment` — 実験実行スキル

**目的**: JAX 実験（最適化、ソルバー性能評価等）を標準ワークフローで実行・管理

**対応ドキュメント**:
- `documents/experiment-workflow.md` の 1-4 段階
- `documents/coding-conventions-experiments.md`
- `documents/experiment-report-style.md`

---

## 概要

このスキルは、以下の 5 段階で実験を管理：

1. **準備** — 問題定義・比較対象・Metrics 決定
2. **実装** — 実験コード生成
3. **静的チェック** — 実装が仕様どおりか検証
4. **実行** — `experiment_runner` で実験実行
5. **結果生成** — JSON + レポート自動生成

---

## 使用方法

### CLI で実行

```bash
# 新規実験スタート（対話型）
claude --skill run-experiment --interactive

# 既存設定から実行
copilot --skill run-experiment --config experiments/my-solver/setup.yaml

# 複数実験バッチ実行
claude --skill run-experiment --batch experiments/*/setup.yaml
```

### ステップ 1: 準備

```bash
# テンプレートを開く
cat .github/skills/03-run-experiment/setup-template.md

# または AI にテンプレート生成させる
claude "実験: Smolyak スパースグリッド vs 従来法。

Question: Smolyak の精度・速度性能  
Comparison Target: 従来 Newton 法
Metrics: 精度（相対誤差）、実行時間、メモリ使用量
Stop Condition: verified (完全性能比較)
"
```

### ステップ 2: 実装

```bash
# 実装ガイド参照
cat .github/skills/03-run-experiment/implementation-guide.md

# または AI にコード生成させる
claude --skill run-experiment \
  --stage implementation \
  --problem "experiments/smolyak/setup.yaml"
```

### ステップ 3: 静的チェック

```bash
# 実装が仕様どおりか検証
python .github/skills/03-run-experiment/validate-results.py \
  --experiment experiments/smolyak/

# 失敗例：formula-code mismatch, parameter missing, etc.
```

### ステップ 4: 実行

```bash
# 実験実行
python .github/skills/03-run-experiment/run-experiment.py \
  --config experiments/smolyak/setup.yaml \
  --timeout 3600 \
  --seed 42

# または
python -m python.experiment_runner \
  experiments/smolyak/my_exp.py \
  --num-runs 10 \
  --output experiments/smolyak/results/
```

### ステップ 5: 結果生成

```bash
# 結果の JSON 生成
python .github/skills/03-run-experiment/generate-results.py \
  experiments/smolyak/results/ \
  --output experiments/smolyak/results.json

# レポート自動生成（実装例参照）
ls experiments/smolyak/report/ # auto-generated markdown + plots
```

---

## テンプレート・ガイド

### `setup-template.md` — 実験設定テンプレート

```markdown
# Experiment: [Name]

## 1. Question
実験で何を確かめたいか。例：
- Smolyak スパースグリッドの精度性能
- 最適化ソルバーの収束速度　
- メモリ効率性

## 2. Comparison Target
比較対象。例：
- Newton 法 (baseline)
- 既存 scipy.optimize
- 別実装 (v1.0)

## 3. Metrics
計測項目。例：
```json
{
  "accuracy": "relative error",
  "speed": "wall-clock time (sec)",
  "memory": "peak memory (MB)",
  "convergence_rate": "iterations to tolerance"
}
```

## 4. Stop Condition
smoke / verified / publish？

## 5. Fairness Notes
- Same case set: ✅
- Same timeout: ✅
- Same hardware: ✅
- Same seed policy: ✅
- Same allocator: ✅
```

### `implementation-guide.md` — 実装ガイド

実装時の注意点：
- 数式と実装の対応を明確に
- Parameter はすべてコンフィグから読み込み
- Random seed を固定可能に
- 例外処理を明示的に

---

## スクリプト詳細

### `run-experiment.py` — 実験実行

```bash
python .github/skills/03-run-experiment/run-experiment.py \
  --config SETUP_YAML \
  [--timeout SEC] \
  [--seed SEED] \
  [--verbose] \
  [--report-dir DIR]
```

動作：
1. setup.yaml 読み込み
2. 実験コード ロード
3. Seed 固定
4. 複数回実行（num-runs 回）
5. metric 計測・JSON 保存
6. レポート生成

出力：
- `results.json` — 計測値
- `report.md` — 図表付きレポート
- `logs/` — 実行ログ

### `validate-results.py` — 仕様検証

```bash
python .github/skills/03-run-experiment/validate-results.py \
  --experiment SETUP_YAML \
  [--verbose]
```

検証項目：
- formula-code 対応
- parameter 整合性
- 初期条件・境界条件 明示
- 例外処理の適切性

---

## チェックリスト

実験実行前に以下を確認：

| 項目 | チェック |
|------|---------|
| Question が定義されている | ✅ |
| Comparison Target が決定されている | ✅ |
| Metrics が JSON で定義されている | ✅ |
| 実装が `setup.yaml` 参照している | ✅ |
| 静的チェック (`validate-results.py`) が成功 | ✅ |
| `experiment_runner` で smoke テスト成功 | ✅ |

---

## 実装例

### 例: Smolyak スパースグリッド評価

```bash
# セットアップ
cat experiments/smolyak_experiment/setup.yaml

# 実行
python .github/skills/03-run-experiment/run-experiment.py \
  --config experiments/smolyak_experiment/setup.yaml \
  --timeout 3600

# 結果確認
ls -la experiments/smolyak_experiment/results/
# results.json, report.md, plots/ がなど
```

---

## 実装状況

| 要素 | 状態 | 備考 |
|------|------|------|
| setup-template.md | ✅ 実装 | 標準テンプレート |
| implementation-guide.md | ✅ 実装 | 注意点・ベストプラクティス |
| run-experiment.py | ✅ 実装 | experiment_runner 統合 |
| validate-results.py | ✅ 実装 | 仕様検証スクリプト |
| 実装例（Smolyak） | 🔄 開発中 | Examples ディレクトリ |

---

## 参考資料

- **実験ワークフロー**: `documents/experiment-workflow.md`
- **実験規約**: `documents/coding-conventions-experiments.md`
- **レポート書き方**: `documents/experiment-report-style.md`
- **統合計画**: `.github/SKILLS_INTEGRATION_PLAN.md`
