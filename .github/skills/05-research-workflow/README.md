# Skill: `research-workflow` — 研究・改造フロー スキル

**目的**: 実験の長期反復ループを管理（問題定義→実験→レビュー→改造→反復）

**対応ドキュメント**:
- `documents/research-workflow.md`
- `documents/experiment-workflow.md`
- `documents/WORKFLOW_INVENTORY.md`

---

## 概要

このスキルは、以下のサイクルを繰り返し支援：

```
┌─────────────────────────────────────────┐
│ Iteration 1: Problem Formulation        │
│ → Run Experiment 01                     │
│ → Critical Review                       │
│ ↓ (Hypothesis refined)                  │
├─────────────────────────────────────────┤
│ Iteration 2: Hypothesis Update          │
│ → Run Experiment 02                     │
│ → Critical Review                       │
│ ↓ (Research direction adjusted)         │
├─────────────────────────────────────────┤
│ Iteration 3: Final Paper                │
│ → Run Experiment 03 (final)             │
│ → Critical Review                       │
│ → Create PR with publication plan       │
└─────────────────────────────────────────┘
```

---

## 使用方法

### CLI で実行

```bash
# 新規研究プロジェクト開始（対話型、複数反復管理）
claude --skill research-workflow --track-formulation

# 既存研究の継続（Iteration 2 開始）
copilot --skill research-workflow \
  --project experiments/my-research/ \
  --start-iteration 2

# 複数反復の活動全体を管理
claude --skill research-workflow \
  --iterations 3 \
  --track-metrics \
  --generate-summary
```

### Phase 別実施

```bash
# Phase 1: 問題定義
cat .github/skills/05-research-workflow/problem-formulation.md
# → AI に対話で埋めてもらう

# Phase 2: 実験実行
# → run-experiment スキルにハンドオフ

# Phase 3: 批判的レビュー
# → critical-review スキルにハンドオフ

# Phase 4: 仮説更新・問題再定義
cat .github/skills/05-research-workflow/iterative-update.md

# Phase 5: Commit & Branch 管理
python .github/skills/05-research-workflow/branch-management.py \
  --project experiments/my-research/ \
  --iteration 2 \
  --action commit
```

---

## ドキュメント群

### `problem-formulation.md` — 問題定義テンプレート

```markdown
# Research Problem Definition

## 1. Research Question
何を解きたいのか。例：
- Smolyak 疎グリッド法はどのレベルで高次元対応するのか
- 新しい regularization 方法は既存法より robust か

## 2. Hypothesis
今のところ、そうだと予想する理由。例：
- 疎グリッド→次元数に指数的でなく polynomial に複雑度増加？

## 3. Comparison Target
- Baseline: 完全グリッド法
- Existing: scipy.sparse_grid
- Literature: [Cite 2 papers]

## 4. Success Criteria
実験成功の判定基準。例：
- Speedup factor > 10x for d=20
- Accuracy within 1e-6

## 5. Scope & Resources
- Max computation time: 1000 GPU hours
- Team bandwidth: 40 hours
- Deadline: 2026-06-01

## 6. Related Work
既知の方法・結果とのポジショニング。

## 7. Expected Contributions
novel points:
- X を初めて Y に適用
- Z の新規 analysis

## 8. Risk & Mitigation
- Risk: 収束速度が期待値より低い
  → Mitigation: 別 baseline と比較、パラメータ tuning 追加時間確保
```

### `iterative-update.md` — 反復更新ガイド

実験結果が仮説を支持しなかった場合の対応

```markdown
# After Experiment 1: Hypothesis Update Decision

## Decision Tree

### A) Hypothesis は支持された
→ Next: 結果を論文化、publication plan

### B) Hypothesis は部分的に支持
→ Next: 追加実験で confounding factor 除外

### C) Hypothesis は支持されなかった & 興味深い failure
→ Next: 新しい Hypothesis 立案、実験 2 へ

### D) 実験失敗（bug / infrastructure 問題）
→ Next: Root cause 分析、同じ hypothesis で実験 1' を repeat

## 仮説更新のテンプレート

### 旧仮説
[前回の hypothesis]

### 実験結果（要約）
[主要な metric と observation]

### 新しい解釈
新しい観点から、なぜこの結果が出たのか仮説を立て直す。

### 新しい仮説
[更新版]

### 次のアプローチ
変わる実験設計・比較対象・metrics など。
```

### `branch-management.py` — Branch 運用スクリプト

```bash
python .github/skills/05-research-workflow/branch-management.py \
  --project experiments/my-research/ \
  --iteration N \
  --action [commit|create-pr|merge-results]
```

動作：
- Iteration ごとに branch を分ける（`research/iter-1`, `research/iter-2` etc）
- 実験データ・結果は `.gitignore` で追跡対象外（`results/` ディレクトリ）
- 実装・図表・report 文書は commit
- Iteration 完了時に `results` tag を作成

---

## チェックリスト

### Iteration を開始する前に

- [ ] Problem formulation が明確（AI と対話で埋めた）
- [ ] Success criteria が定量的
- [ ] Resource estimate が現実的
- [ ] Related work がリストアップされている

### Iteration を実施する

- [ ] Experiment を `run-experiment` で実行
- [ ] 結果を JSON + report で記録
- [ ] Critical review を実施
- [ ] 仮説更新（success → paper, failure → next hypothesis）

### Iteration を終了する前に

- [ ] Branch commit 実施 (`branch-management.py --action commit`)
- [ ] Results tag 作成 (`git tag results/iter-N`)
- [ ] Documentation 更新（problem + findings）
- [ ] Next iteration plan 記述

---

## 実装例：Smolyak 研究の 3 反復

### Iteration 1: 基本性能比較
```
Problem: Smolyak vs Complete Grid — 高次元での speedup は？
Hypothesis: d > 15 で 100x speedup 期待
Result: d=20 で 12x speedup → 期待値より低い

Update: Smolyak が tuning に敏感？ → iter 2 で sensitivity analysis
```

### Iteration 2: パラメータ感度分析
```
Problem: Smolyak sensitivity to α parameter
Hypothesis: 適切な α selection で 100x 達成可能
Result: 特定 α で 95x speedup 確認

Update: 良好。Publication へ ← iter 3
```

### Iteration 3: 論文完成
```
Problem: 結果の一般化・reproducibility
Hypothesis: 公開 benchmark で同等性能達成可能
Result: Benchmark で確認・reproducibility code 公開

Final: PR create + Publication plan
```

---

## 実装状況

| 要素 | 状態 | 備考 |
|------|------|------|
| problem-formulation.md | ✅ 実装 | テンプレート完成 |
| iterative-update.md | ✅ 実装 | 反復更新ガイド |
| branch-management.py | ✅ 実装 | Branch/tag 自動化 |
| formula-tracking.py | 🔄 開発中 | 数式・parameter トラッキング |
| iteration-summary.py | 🔄 開発中 | 反復サマリー自動生成 |

---

## 参考資料

- **shared skill canon**: `agents/skills/research-workflow.md`
- **研究ワークフロー**: `documents/research-workflow.md`
- **実験ワークフロー**: `documents/experiment-workflow.md`
