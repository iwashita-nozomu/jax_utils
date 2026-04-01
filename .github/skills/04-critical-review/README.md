# Skill: `critical-review` — 実験批判的レビュースキル

**目的**: 実験の学術的妥当性を多角的にレビュー（Code・Results・Math の 3 層）

**対応ドキュメント**:
- `documents/experiment-critical-review.md`
- `documents/experiment-workflow.md` ステージ 4 (Results)
- `documents/research-workflow.md`

---

## 概要

このスキルは、実験完了後に以下を検証：

### 3 つのレビュー役割

1. **Change Reviewer** — Code 差分・実装の安全性
2. **Experiment Reviewer** — 結果・図表・結論
3. **Math Reviewer** — 数学的妥当性・仮定の明示

---

## 使用方法

### CLI で実行

```bash
# 複数ロール対応 (AI が自動で担当)
claude --skill critical-review experiments/my-exp/

# または Phase 12 (results) 後に自動トリガー
copilot --skill critical-review --auto-trigger

# 特定ロールのみ
claude --skill critical-review \
  --role change-reviewer \
  experiments/my-exp/code-diff.patch
```

### ローカルチェックリスト参照

```bash
# Change Reviewer のチェックリスト
cat .github/skills/04-critical-review/checklist-change-reviewer.md

# Experiment Reviewer のチェックリスト
cat .github/skills/04-critical-review/checklist-experiment-reviewer.md

# Math Reviewer のチェックリスト
cat .github/skills/04-critical-review/checklist-math-validity.md
```

---

## ロール別チェックリスト

### Role: Change Reviewer

**責務**: コード差分の安全性・実装の正確性を確認

```markdown
## Code Quality
- [ ] 差分が実装仕様に対応している
- [ ] 危険な shortcut や手抜き実装がない
- [ ] Exception handling が明示的か（generic except: なし）
- [ ] Boundary condition チェック適切か

## Numerical Safety
- [ ] Float 演算の丸め誤差が制御されている
- [ ] Over/Underflow protection あり
- [ ] Division by zero チェック あり
- [ ] Parameter validation が適切か

## Formula ⇔ Code Correspondence
- [ ] 数式と実装の変数対応を確認
- [ ] Loop 文が数学的に正確か
- [ ] Indexing off-by-one error なし
- [ ] 仮定（例：矩形行列）が実装に反映されているか
```

### Role: Experiment Reviewer

**責務**: 結果・図表・解釈の妥当性を確認

```markdown
## Protocol & Fairness
- [ ] Case set が公平か（同じテスト case で比較）
- [ ] Timeout が同じか
- [ ] Hardware が同じか（明示されているか）
- [ ] Seed policy が同じか

## Results & Claims
- [ ] 結果が question に答えているか
- [ ] Overclaim がないか（データ不足で結論出していない）
- [ ] Evidence sufficiency — 結論に必要なデータがすべてあるか

## Figures & Tables
- [ ] X/Y ラベルが明確か
- [ ] Legend が正確か
- [ ] Error bar / confidence interval が示されているか
- [ ] 図表が misunderstanding を招かないか

## Report Quality
- [ ] Methods section が reproducible か
- [ ] Results section が客観的な描写か
- [ ] Discussion に literature との接続あり
- [ ] Limitations が述べられているか
```

### Role: Math Reviewer

**責務**: 数学的妥当性・仮定の明示を確認

```markdown
## Mathematical Validity
- [ ] アルゴリズムが数学的正確か
- [ ] 前提仮定（凸性、smoothness等）が明示されているか
- [ ] 収束理論が説明されているか
- [ ] 境界条件・初期条件の取り扱い正確か

## External Connection
- [ ] 既知文献と接続されているか
- [ ] Baseline / reference 実装と正当に比較されているか
- [ ] Novel contribution が明確か

## Parameter & Condition Tracking
- [ ] 全 parameter が明示されているか
- [ ] Hyperparameter tuning の方法が述べられているか
- [ ] Sensitivity analysis が必要な箇所は実施されているか
```

---

## 自動検証スクリプト

### `validate-experiment.py` — 自動チェック

```bash
python .github/skills/04-critical-review/validate-experiment.py \
  experiments/my-exp/ \
  [--strict] \
  [--report DIR]
```

自動チェック項目：
- Code 構造解析（formula-code 対応推定）
- Figure Quality 検査（軸ラベル、Legend など）
- Claim Sufficiency 検査（結論を支持する figure/table 確認）
- Reproducibility 検査（parameter 記述完全性）

出力：
- PASS / WARN / FAIL + 詳細理由

---

## レビュー報告書テンプレート

```markdown
# Critical Review Report

## 対象
- Experiment: Smolyak Sparse Grid Performance
- PR: #55
- Date: 2026-04-01

## Change Reviewer: ✅ PASS
- Code quality: OK
- Numerical safety: OK
- Formula-Code: 対応確認 ✅

### Comment
実装は仕様通り。特に boundary condition チェックが丁寧。

---

## Experiment Reviewer: ⚠️ WARN
- Protocol fairness: OK
- Results sufficiency: WARN ⚠️
- Figures quality: OK

### Comment
結果は信頼できるが、case set が small に見える（n=5のみ）。
より diverse な case での検証を追加推奨。

---

## Math Reviewer: ✅ PASS (with suggestion)
- Mathematical validity: OK
- External connection: OK
- Parameter tracking: OK

### Suggestions
- Convergence theory section が brief。追加参考文献推奨
- Sensitivity analysis: hyperparameter α への依存性が未検査 → 実施推奨

---

## 総合判定
✅ **APPROVED (with above suggestions)**

Suggestions は next iteration（follow-up PR）で対応可。
