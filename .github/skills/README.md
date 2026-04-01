# Skills ライブラリ — jax_util プロジェクト標準ワークフロー

**全 5 つの Skill で、開発от初心者から研究者まで全フェーズをカバー**

---

## 📋 Skill 一覧

| # | Skill | 目的 | タイミング | 自動化 |
|----|-------|------|----------|------|
| **1** | [static-check](./01-static-check/README.md) | 型・テスト・Docker 検証 | 常時 | ⭐⭐⭐⭐⭐ |
| **2** | [code-review](./02-code-review/README.md) | コード品質多層検証（A/B/C層） | PR レビュー | ⭐⭐⭐ |
| **3** | [run-experiment](./03-run-experiment/README.md) | 実験実行・結果生成 | 実験実施 | ⭐⭐ |
| **4** | [critical-review](./04-critical-review/README.md) | 実験学術的妥当性検証 | 実験完了後 | ⭐⭐ |
| **5** | [research-workflow](./05-research-workflow/README.md) | 研究·反復管理（複数iteration） | 長期研究 | ⭐ |

---

## 🚀 使用シーン別ガイド

### シーン A: コード実装・PR 作成（週次作業）

```bash
# 1. 開発中：常に static-check
claude --skill static-check

# 2. PR 作成前：code-review の A 層チェック
bash .github/skills/01-static-check/check-python.sh
bash .github/skills/01-static-check/check-tests.sh

# 3. PR レビュー：code-review の B/C 層
copilot --skill code-review --pr 42 --interactive
```

**チェックリスト**:
- [ ] `static-check` 成功
- [ ] `code-review` A 層 PASS
- [ ] テスト追加・更新
- [ ] ドキュメント同期

---

### シーン B: 実験実施・反復（月次研究活動）

```bash
# 1. 準備：問題定義
cat .github/skills/05-research-workflow/problem-formulation.md

# 2. 実装：実験コード
cat .github/skills/03-run-experiment/implementation-guide.md

# 3. 実行：実験ランナー
python .github/skills/03-run-experiment/run-experiment.py \
  --config experiments/my-exp/setup.yaml

# 4. レビュー：学術妥当性検証
claude --skill critical-review experiments/my-exp/

# 5. 反復管理：次の iteration へ
python .github/skills/05-research-workflow/branch-management.py \
  --iteration 2 --action commit
```

**チェックリスト**:
- [ ] Problem formulation が明確
- [ ] Metrics が定量的
- [ ] Experiment 実行完了
- [ ] Critical review PASS
- [ ] 次 iteration plan 立案

---

### シーン C: publication 準備（四半期）

```bash
# → 研究ワークフロー skill で複数 iteration 管理
claude --skill research-workflow --iterations 3 --generate-summary

# → 最終論文・図表・reproducibility code
# → Publication PR 作成
```

---

## 🔗 統合マップ

### ワークフロー層序

```
┌─────────────────────────────────────────────┐
│ 05-research-workflow                        │
│ 長期反復管理（問題定義→実験→改造→反復）     │
└─────────────────────────────────────────────┘
         ↓ (Iteration ごと)
┌──────────────────────────────────────────┐
│ ┌──────────────────────────┐             │
│ │ 03-run-experiment        │ 実験実施    │
│ │ 準備→実装→チェック→実行  │             │
│ └──┬───────────────────────┘             │
│    ↓                                      │
│ ┌──────────────────────────┐             │
│ │ 04-critical-review       │ 結果検証    │
│ │ Code/Results/Math 検証   │             │
│ └──────────────────────────┘             │
└──────────────────────────────────────────┘
         ↓ (結果確定後)
┌──────────────────────────────────────────┐
│ ┌──────────────────────────┐             │
│ │ 02-code-review           │ PR レビュー │
│ │ A/B/C 層多層検証        │             │
│ ├──────────────────────────┤             │
│ │ 01-static-check          │ 基礎検証    │
│ │ 常時・自動実施          │             │
│ └──────────────────────────┘             │
└──────────────────────────────────────────┘
```

### ドキュメント参照構造

```
1. 利用ガイド（このファイル）
   ├→ 各 Skill README
   │  ├→ Checklist / Template
   │  ├→ Script Reference
   │  └→ Examples
   │
2. 詳細ガイド
   ├→ SKILL_IMPLEMENTATION_GUIDE.md (A/B/C/D/E層)
   ├→ documents/experiment-workflow.md
   ├→ documents/REVIEW_PROCESS.md
   └→ documents/research-workflow.md
   │
3. CLI リファレンス
   └→ documents/AI_AGENT_CUSTOMIZATION_RESEARCH_2026.md
      - Claude CLI / Copilot CLI / Cursor
```

---

## ⚙️ CLI からの使用例

### Claude CLI

```bash
# Skill 実行（自動選択）
claude --skill static-check src/

# 複数 skill チェーン
claude --skill-chain static-check code-review --input src/

# 実験フロー（複数反復管理）
claude --skill research-workflow --iterations 3 --track-metrics

# 対話型（複数ロール）
claude --skill critical-review --role change-reviewer experiments/
```

### GitHub Copilot CLI

```bash
# Agent Mode で利用可能 Skill 表示
copilot -i
> available skills: static-check, code-review, run-experiment, ...

# PR 自動レビュー
copilot review --pr 42 --skill code-review

# Batch 処理
copilot batch-review --pr-list prs.json --auto-skill-select
```

### ローカル実行

```bash
# 直接スクリプト実行
bash .github/skills/01-static-check/check-all.sh
python .github/skills/03-run-experiment/run-experiment.py --config setup.yaml
```

---

## 🔄 GitHub Actions 統合

### 自動パイプライン（`.github/workflows/skills-pipeline.yml`）

```yaml
name: Full Skill Pipeline

on: [pull_request, push]

jobs:
  skill-static-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Static Check Skill
        run: bash .github/skills/01-static-check/check-all.sh

  skill-code-review:
    needs: skill-static-check
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Code Review Skill
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          copilot review --pr ${{ github.event.pull_request.number }} \
            --skill code-review

  skill-experiment:
    if: contains(github.labels, 'experiment')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Experiment Skill
        run: |
          python .github/skills/03-run-experiment/run-experiment.py \
            --config experiments/*/setup.yaml
      - name: Critical Review Skill
        run: |
          python .github/skills/04-critical-review/validate-experiment.py
```

---

## 📊 実装状況

| Skill | README | Script | Config | Examples |
|-------|--------|--------|--------|----------|
| 01-static-check | ✅ | 🔄 | 🔄 | ❌ |
| 02-code-review | ✅ | ✅ | 🔄 | ❌ |
| 03-run-experiment | ✅ | ✅ | 🔄 | 🔄 |
| 04-critical-review | ✅ | 🔄 | 🔄 | ❌ |
| 05-research-workflow | ✅ | 🔄 | ❌ | ❌ |

**凡例**: ✅ 完成 / 🔄 開発中 / ❌ 未開始

---

## 📚 関連ドキュメント

**Skill 定義・設計**:
- `.github/SKILLS_INTEGRATION_PLAN.md` — 全体統合計画

**詳細ガイド**:
- `documents/SKILL_IMPLEMENTATION_GUIDE.md` — Skill 実装ガイド（A/B/C/D/E層）
- `documents/experiment-workflow.md` — 実験標準手順
- `documents/experiment-critical-review.md` — 批判的レビュー手順
- `documents/REVIEW_PROCESS.md` — コードレビューフロー
- `documents/research-workflow.md` — 研究・改造ワークフロー

**CLI リファレンス**:
- `documents/AI_AGENT_CUSTOMIZATION_RESEARCH_2026.md` 【CLI エージェント・コマンドラインリファレンス】

---

## 🎯 次のステップ

**Week 1**: スクリプト実装 (scripts/ → skills/ への移行)
**Week 2**: GitHub Actions パイプライン統合テスト
**Week 3**: チーム向けトレーニング・ドキュメント
**Week 4**: モニタリング・改善フィードバック

---

**Skill ライブラリは進化中です。最新版は常に各 README を参照してください。**
