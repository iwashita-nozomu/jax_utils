# Skill 6: `comprehensive-review` — 包括レビュー Skill

この README は implementation surface です。repo-wide review の最上位入口は `agents/skills/project-review.md` に置きます。

## 概要

ドキュメント・Skill・ツール全体の整合性・完全性を統合的にレビュー。
5段階フェーズで、プロジェクトの健全性を多面的に検証します。

**対象レビュー**:
1. ドキュメント — broken link・用語統一・重複検出
2. Skills — 依存関係・一貫性・漏れ検出
3. ツール/スクリプト — 実装状況・テスト・ドキュメント
4. 統合 — CLI・GitHub Actions・ローカル実行
5. プロジェクト規約 — 全体整合性

---

## 使用方法

### CLI で実行

```bash
# 全フェーズ実施（デフォルト）
claude --skill comprehensive-review

# または
copilot --skill comprehensive-review

# 特定フェーズのみ
claude --skill comprehensive-review --phase 1  # ドキュメント
claude --skill comprehensive-review --phase 2  # Skills
claude --skill comprehensive-review --phase 3  # ツール
claude --skill comprehensive-review --phase 4  # 統合
claude --skill comprehensive-review --phase 5  # 報告書

# Verbose モード（詳細出力）
copilot --skill comprehensive-review --verbose

# 修正リストを自動生成（JSON）
claude --skill comprehensive-review --output json > issues.json

# 修正ロードマップを生成
copilot --skill comprehensive-review --generate-roadmap > roadmap.md
```

### ローカルで直接実行

```bash
# Python スクリプト直接実行
python .github/skills/06-comprehensive-review/run-review.py

# オプション指定
python .github/skills/06-comprehensive-review/run-review.py \
  --phases 1,2,3 \
  --output markdown \
  --save-report

# Verbose
python .github/skills/06-comprehensive-review/run-review.py -v
```

---

## 検査フェーズ概要

### Phase 1: ドキュメント・レビュー

```bash
python .github/skills/06-comprehensive-review/checkers/doc_checker.py
```

**検査項目**:
- ✅ Hyperlink 有効性 (broken link)
- ✅ 用語統一 (Skill/skill, feature/Feature 等)
- ✅ 文書の参照整合性 (循環参照なし)
- ✅ 廃止・古い情報の検出
- ✅ 図表・テーブル完全性
- ✅ 各セクション要約の存在

**出力例**:
```
[HIGH] broken link: documents/experiment-workflow.md:45
  → ../references/experiment_workflow/not_found.pdf

[MEDIUM] terminology: 12 places mix "Skill" and "skill"
  → Line 107: "Skill ライブラリ"
  → Line 123: "skill 統合"

[LOW] outdated: reference to 2024 API
```

### Phase 2: Skills 一貫性チェック

```bash
python .github/skills/06-comprehensive-review/checkers/skills_checker.py
```

**検査項目**:
- ✅ 5つのSkill全ての README 存在確認
- ✅ 各 Skill の「対応ドキュメント」フィールド確認
- ✅ CLI 使用例フォーマット統一
- ✅ 自動化レベル(⭐数) の妥当性
- ✅ 依存 Skill の明記
- ✅ チェックリスト・テンプレートの実装状況

**出力例**:
```
Skill 1 (static-check): ✅ PASS
Skill 2 (code-review): ✅ PASS
  ⚠️ WARNING: CLI example has 2 variations (needs standardization)

Skill 3 (run-experiment): ⚠️ WARNING
  ❌ Missing: "Depends on: Skill 1, 2"

Skill 4 (critical-review): ✅ PASS
Skill 5 (research-workflow): ✅ PASS
  ⚠️ WARNING: Overlap with Skill 3 in iteration management
```

### Phase 3: ツール・スクリプト確認

```bash
python .github/skills/06-comprehensive-review/checkers/tools_checker.py
```

**検査項目**:
- ✅ scripts/ + .github/skills/ 内のスクリプト一覧
- ✅ 各スクリプトの実装状況(✅/🔄/❌)
- ✅ Docstring・エラー処理の有無
- ✅ README への記載状況
- ✅ 依存パッケージ明示
- ✅ テストケース

**出力例**:
```
scripts/check_convention_consistency.py: ✅ Implemented
  ✅ Docstring present
  ✅ Error handling: exit code logic
  ✅ README: documented
  ⚠️ Tests: minimal (1/5 edge cases)

scripts/run_comprehensive_review.sh: 🔄 Partial
  ❌ Does not cover all Skill phases
  ⚠️ Timeout handling missing

.github/skills/01-static-check/check-python.sh: ✅ Implemented
```

### Phase 4: 統合検証

```bash
python .github/skills/06-comprehensive-review/checkers/integration_checker.py
```

**検査項目**:
- ✅ CLI (Claude/Copilot/Cursor) での --skill flag 有効性
- ✅ GitHub Actions パイプライン対応
- ✅ タイムアウト設定の妥当性
- ✅ Skill チェーン (sequential/parallel) 可能性
- ✅ ローカル実行ガイドの完全性
- ✅ Docker 環境対応

**出力例**:
```
CLI Integration:
  ✅ Claude CLI: --skill flag works
  ✅ Copilot CLI: Agent Mode ready
  ⚠️ Cursor: --apply-rules needs .cursorrules file

GitHub Actions:
  ✅ Workflow file exists: .github/workflows/skills-pipeline.yml
  ⚠️ Coverage: Skill 1,2 only (need 3,4,5)

Docker:
  ✅ Requirements defined
  ❌ Docker Compose file missing
```

### Phase 5: 報告書生成

```bash
python .github/skills/06-comprehensive-review/checkers/report_generator.py
```

**生成物**:
```
reports/comprehensive-review-20260401-100000.md
├─ Executive Summary
├─ Phase 1-5 Details
├─ Priority-based Todo List
├─ Implementation Roadmap
└─ Recommendations
```

---

## チェックリスト

Comprehensive Review 実施フロー：

- [ ] Phase 1 実行 — ドキュメント問題抽出
- [ ] Phase 2 実行 — Skills 依存関係検証
- [ ] Phase 3 実行 — ツール実装確認
- [ ] Phase 4 実行 — 統合テスト
- [ ] Phase 5 実行 — 報告書自動生成
- [ ] レポート確認 (`reports/comprehensive-review-*.md`)
- [ ] 高優先度 issue から修正開始
- [ ] GitHub Issue に自動登録（オプション）

---

## 既知の検出可能な Issue

このSkillでの検出予定：

| Phase | Issue | 優先度 |
|-------|-------|--------|
| 1 | broken link (参考資料 PDF) | HIGH |
| 1 | 用語混在 (Skill/skill) | MEDIUM |
| 2 | Skill 3-5 の重複機能 | MEDIUM |
| 2 | 依存関係の明記不足 | MEDIUM |
| 3 | スクリプト実装状況バラバラ | MEDIUM |
| 3 | テストカバレッジ不均等 | LOW |
| 4 | GitHub Actions coverage不足 | MEDIUM |
| 4 | Docker Compose 未定義 | LOW |
| 5 | 自動修正ロードマップ生成 | N/A |

---

## 実装状況

| コンポーネント | 状態 |
|--------------|------|
| README.md | ✅ 完成 |
| run-review.py | 🔄 開発中 |
| doc_checker.py | 🔄 開発中 |
| skills_checker.py | 🔄 開発中 |
| tools_checker.py | 🔄 開発中 |
| integration_checker.py | 🔄 開発中 |
| report_generator.py | 🔄 開発中 |
| config/*.yaml | 🔄 開発中 |
| CLI 統合テスト | ⏳ 未開始 |

---

## 参考資料

- **shared skill canon**: `agents/skills/comprehensive-review.md`
- **GitHub skill hub**: `.github/skills/README.md`
- **実装ガイド**: `documents/SKILL_IMPLEMENTATION_GUIDE.md`
