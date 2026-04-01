# Skills 統合計画 — jax_util プロジェクトの標準ワークフローをスキル化

**目的**: 既存の実験・コード レビュー・テスト ワークフローを GitHub Copilot / Claude Skills として標準化し、エージェントと連携可能にする

**更新日**: 2026-04-01  
**対象**: `.github/skills/` ディレクトリ構成

---

## 1. Skills の体系（全 5 個）

### トップレベル構成

```
.github/skills/
├── 01-static-check/          ← 基礎検証（型チェック・テスト）
├── 02-code-review/           ← コードレビュー
├── 03-run-experiment/        ← 実験実行
├── 04-critical-review/       ← 実験の批判的レビュー
├── 05-research-workflow/     ← 研究・改造フロー
└── README.md                 ← Skills 一覧・活用ガイド
```

---

## 2. 各スキルの詳細仕様

### Skill 1: `static-check` — 基礎検証スキル

**対応ドキュメント**: `SKILL_IMPLEMENTATION_GUIDE.md` の A 層（基礎検証）

**主責務**:
- 型注釈・Docstring・テストの検証
- pyright, ruff, pytest の実行
- Docker & requirements.txt の整合性

**実装物**:
```
.github/skills/01-static-check/
├── README.md                      # スキル説明・使用例
├── check-python.sh               # Python 静的チェック（pyright, ruff）
├── check-cpp.sh                  # C++ 静的チェック（clang-tidy）
├── check-tests.sh                # pytest 実行
├── check-docker.py               # Docker 依存関係検証
└── config.yaml                   # スキル定義（optional）
```

**スキル定義テンプレート** (`config.yaml`):
```yaml
skill_name: static-check
description: "基礎検証 — 型チェック・テスト・Docker"
version: "1.0"
language: "bash/python"
auto_trigger: true  # PR 作成時に自動実行
commands:
  - name: "check-python"
    run: "bash .github/skills/01-static-check/check-python.sh"
    timeout: 300
  - name: "check-tests"
    run: "bash .github/skills/01-static-check/check-tests.sh"
    timeout: 600
  - name: "check-docker"
    run: "python .github/skills/01-static-check/check-docker.py"
    timeout: 120
```

**使用例** (Claude/Copilot):
```bash
# CLI で実行
claude --skill static-check "my_module.py"

# または
copilot --skill static-check
```

---

### Skill 2: `code-review` — コードレビュースキル

**対応ドキュメント**: `REVIEW_PROCESS.md` + `SKILL_IMPLEMENTATION_GUIDE.md` の B, C 層

**主責務**:
- Python/C++ コードの品質レビュー（A 層 → B 層 → C 層）
- コード⇔ドキュメント⇔テストの三点セット検証
- PR フロー・チェックリストの実施

**実装物**:
```
.github/skills/02-code-review/
├── README.md                      # レビュー手順
├── checklist-a-layer.md           # A 層チェックリスト（型・Doc・テスト）
├── checklist-b-layer.md           # B 層チェックリスト（アーキテクチャ）
├── checklist-c-layer.md           # C 層チェックリスト（規約整合）
├── report-template.md             # レビュー報告書テンプレート
├── check_doc_test_triplet.py     # 三点セット検証スクリプト
└── config.yaml
```

**スキル定義テンプレート** (`config.yaml`):
```yaml
skill_name: code-review
description: "コードレビュー — A層→B層→C層の多層検証"
version: "1.0"
language: "markdown + python"
interactive: true  # ユーザーの判断を求める
phases:
  - name: "Phase A: 基礎検証"
    checklist: ".github/skills/02-code-review/checklist-a-layer.md"
    auto_check: true
  - name: "Phase B: 深度検証"
    checklist: ".github/skills/02-code-review/checklist-b-layer.md"
    auto_check: false  # 手動確認
  - name: "Phase C: 統合検証"
    checklist: ".github/skills/02-code-review/checklist-c-layer.md"
    auto_check: true
  - name: "Phase D: 報告書生成"
    template: ".github/skills/02-code-review/report-template.md"
    auto_generate: true
```

**使用例** (Claude/Copilot):
```bash
# PR レビュー実施
copilot review --skill code-review https://github.com/user/repo/pull/42

# または対話型
copilot --skill code-review --interactive
```

---

### Skill 3: `run-experiment` — 実験実行スキル

**対応ドキュメント**: `experiment-workflow.md` の 1-4 段階

**主責務**:
- 実験の準備（Question, Metrics, Stop Condition 定義）
- 実験実装・コード生成
- 実験実行・結果生成
- 静的チェック（実装が仕様どおりか）

**実装物**:
```
.github/skills/03-run-experiment/
├── README.md                      # 実験ワークフロー概要
├── setup-template.md              # 実験設定テンプレート（Question等）
├── implementation-guide.md        # 実験コード実装ガイド
├── run-experiment.py              # 実験実行スクリプト
├── validate-results.py            # 結果検証スクリプト
├── config.yaml
└── examples/                       # 実装例（JAX実験等）
    ├── smolyak-experiment/
    └── solver-benchmark/
```

**スキル定義テンプレート** (`config.yaml`):
```yaml
skill_name: run-experiment
description: "実験実行 — 準備→実装→実行→結果生成"
version: "1.0"
language: "python"
stages:
  - stage: 1-setup
    name: "準備: 問題定義・比較対象・Metrics"
    template: "setup-template.md"
    duration: "30 min"
  - stage: 2-implementation
    name: "実装: 実験コード生成"
    guide: "implementation-guide.md"
    duration: "1-2 hours"
  - stage: 3-static-check
    name: "静的チェック: 仕様との対応検証"
    command: "python .github/skills/03-run-experiment/validate-results.py"
    duration: "15 min"
  - stage: 4-execution
    name: "実行: 実験ランナー起動"
    command: "python .github/skills/03-run-experiment/run-experiment.py"
    duration: "depends"
  - stage: 5-results
    name: "結果生成: JSON + report"
    auto_generate: true
```

**使用例** (Claude/Copilot):
```bash
# 新規実験スタート
claude --skill run-experiment --context ./experiments/

# または設定ファイルから
copilot --skill run-experiment --config experiments/my_exp/setup.yaml
```

---

### Skill 4: `critical-review` — 実験批判的レビュースキル

**対応ドキュメント**: `experiment-critical-review.md`

**主責務**:
- 実験コード⇔数式の対応検証
- 結果の学術的妥当性検証
- 図表・結論・overclaim チェック
- Change reviewer と Experiment reviewer 役割支援

**実装物**:
```
.github/skills/04-critical-review/
├── README.md                      # 批判的レビュー手順
├── checklist-change-reviewer.md   # Code diff 確認チェックリスト
├── checklist-experiment-reviewer.md  # 結果・図表確認チェックリスト
├── checklist-math-validity.md     # 数学的妥当性確認
├── report-template.md             # 批判的レビューレポートテンプレート
├── validate-experiment.py         # 自動検証スクリプト
└── config.yaml
```

**スキル定義テンプレート** (`config.yaml`):
```yaml
skill_name: critical-review
description: "実験批判的レビュー — Code・Results・Math 妥当性検証"
version: "1.0"
language: "markdown + python"
roles:
  - name: "change-reviewer"
    checklist: "checklist-change-reviewer.md"
    focus: "コード実装が仕様どおりか、数値的に危くないか"
  - name: "experiment-reviewer"
    checklist: "checklist-experiment-reviewer.md"
    focus: "結果が問いに答えているか、図表が適切か"
  - name: "math-reviewer"
    checklist: "checklist-math-validity.md"
    focus: "数学的妥当性、仮定・境界条件の明示"
auto_checks:
  - "code vs equation"
  - "parameter validation"
  - "figure correctness"
  - "claim sufficiency"
```

**使用例** (Claude/Copilot):
```bash
# 実験レビュー開始（複数ロール対応）
claude --skill critical-review --role change-reviewer experiments/my_exp/

# または複数エージェント協調
copilot subagent workflow --config .github/subagents/experiment-review.yaml
```

---

### Skill 5: `research-workflow` — 研究・改造フロー スキル

**対応ドキュメント**: `research-workflow.md`

**主責務**:
- 問題定式化・逐次改造の管理
- 数式・比較対象・branch 反映の標準化
- 実験反復ループの支援

**実装物**:
```
.github/skills/05-research-workflow/
├── README.md                      # 研究ワークフロー概要
├── problem-formulation.md         # 問題定義テンプレート
├── iterative-update.md            # 逐次改造手順
├── formula-tracking.py            # 数式・parameter トラッキング
├── branch-management.py           # Branch 運用スクリプト
└── config.yaml
```

**スキル定義テンプレート** (`config.yaml`):
```yaml
skill_name: research-workflow
description: "研究・改造フロー — 問題定義→逐次改造→branch反映"
version: "1.0"
language: "markdown + python"
workflow_loop:
  iteration_1:
    step: "Define question & comparison target"
    reference: "problem-formulation.md"
  iteration_2:
    step: "Run experiment & collect metrics"
    reference: ".github/skills/03-run-experiment/"
  iteration_3:
    step: "Critical review & evaluate results"
    reference: ".github/skills/04-critical-review/"
  iteration_4:
    step: "Update problem formulation / hypothesis"
    reference: "iterative-update.md"
  iteration_5:
    step: "Commit & create PR"
    reference: ".github/skills/02-code-review/"
integration:
  branch_strategy: "feature branch per iteration"
  merge_criteria: "All phases completed + reviewer approval"
```

**使用例** (Claude/Copilot):
```bash
# 研究ワークフロー開始
claude --skill research-workflow --track-formulation

# 複数反復の管理
copilot workflow run .github/skills/05-research-workflow/config.yaml \
  --iterations 3 \
  --track-metrics
```

---

## 3. 統合マップ — Skills ⇔ 既存ワークフロー

| 既存ワークフロー | 対応 Skill | 実施タイミング | 自動化 |
|-----------------|-----------|--------------|------|
| SKILL_IMPLEMENTATION_GUIDE A層 | `01-static-check` | 作成中・PR前 | ⭐⭐⭐⭐⭐ |
| SKILL_IMPLEMENTATION_GUIDE B層 | `02-code-review` | PR レビュー | ⭐⭐⭐ |
| SKILL_IMPLEMENTATION_GUIDE C層 | `02-code-review` | PR レビュー | ⭐⭐⭐ |
| REVIEW_PROCESS.md | `02-code-review` | PR フロー | ⭐⭐ |
| experiment-workflow.md (1-4段階) | `03-run-experiment` | 実験実施 | ⭐⭐ |
| experiment-critical-review.md | `04-critical-review` | 実験完了後 | ⭐⭐ |
| research-workflow.md | `05-research-workflow` | 長期研究 | ⭐ |

---

## 4. Skill を CLI/エージェントから使用する方法

### 4.1 Claude CLI

```bash
# Skill 単独実行
claude --skill static-check src/my_module.py

# Skill チェーン（複数 skill の順序実行）
claude --skill-chain static-check code-review \
  --input src/

# Skill + カスタム context
claude --skill run-experiment \
  --repo-context .github/copilot-instructions.md \
  --experiment-config experiments/my_exp/setup.yaml
```

### 4.2 GitHub Copilot CLI

```bash
# Agent Mode で Skill 一覧表示
copilot -i
> available skills: static-check, code-review, run-experiment, ...

# PR レビュー（automatic skill selection）
copilot review --pr 42 --auto-skill-select

# Skill 指定実行
copilot --skill code-review --pr 42 --phase A

# Batch 実行（複数 PR）
copilot batch-review --pr-list prs.json --skill code-review
```

### 4.3 GitHub Actions 統合

```yaml
# .github/workflows/skill-pipeline.yml
name: Full Skill Pipeline

on: [pull_request]

jobs:
  static-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Static Check Skill
        run: |
          bash .github/skills/01-static-check/check-python.sh
          bash .github/skills/01-static-check/check-tests.sh

  code-review:
    needs: static-check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run Code Review Skill
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          copilot --skill code-review --pr ${{ github.event.pull_request.number }}

  experiment-pipeline:
    if: contains(github.event.pull_request.labels, 'experiment')
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

## 5. 実装ロードマップ

### Week 1: 基礎スキル実装

- [ ] Skill 1 (`static-check`) ファイル構成作成
- [ ] 既存 `scripts/run_comprehensive_review.sh` を Skill に統合
- [ ] Skill 1 テスト・ドキュメント完成

### Week 2: コードレビュー スキル

- [ ] Skill 2 (`code-review`) ファイル構成作成
- [ ] A/B/C 層チェックリスト ドキュメント整理
- [ ] レポートテンプレート作成

### Week 3: 実験スキル

- [ ] Skill 3 (`run-experiment`) with `experiment_runner` 統合
- [ ] Skill 4 (`critical-review`) テンプレート作成
- [ ] 実装例（Smolyak 実験）作成

### Week 4: 統合・テスト

- [ ] Skill 5 (`research-workflow`) 実装
- [ ] CLI / GitHub Actions パイプライン テスト
- [ ] ドキュメント統合 + チーム向けガイド作成

---

## 6. 参考資料

- **Skill 定義の詳細**: `SKILL_IMPLEMENTATION_GUIDE.md` セクション 1-5
- **既存ワークフロー**: `experiment-workflow.md`, `REVIEW_PROCESS.md`, `research-workflow.md`
- **CLI リファレンス**: `AI_AGENT_CUSTOMIZATION_RESEARCH_2026.md` - CLI エージェント・コマンドラインリファレンス

---

**このプランは動的であり、実装の進展に応じて更新されます。**  
**最新版は常に `.github/SKILLS_INTEGRATION_PLAN.md` を参照してください。**
