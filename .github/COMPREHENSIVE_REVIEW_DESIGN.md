# Skill 6: `comprehensive-review` — 包括レビュー Skill（設計文書）

**目的**: 5つのSkill + ドキュメント + ツール全体を統合的にレビュー

**レビュー対象**:
1. ドキュメント間の矛盾・重複・不足
2. Skill間の一貫性・依存関係
3. スクリプト・ツールの状態・完全性
4. CLI/GitHub Actions対応状況
5. プロジェクト規約との整合性

---

## 1. 包括レビュー Skill の階層構造

```
Skill 6: comprehensive-review
├── Phase 1: Documentation Review
│   ├── 文書間の参照一貫性
│   ├── 用語統一・定義明確性
│   └── 対象読者への適切性
│
├── Phase 2: Skills Coherence Check
│   ├── 5つのSkill間の依存関係
│   ├── 重複機能の検出
│   ├── 欠落機能の検出
│   └── 順序・タイミングの妥当性
│
├── Phase 3: Tools & Scripts Inventory
│   ├── スクリプト実装状況
│   ├── テスト・エラー処理
│   └── ドキュメント参照
│
├── Phase 4: Integration Validation
│   ├── CLI (Claude/Copilot) 整合性
│   ├── GitHub Actions パイプライン
│   └── ローカル実行方法
│
└── Phase 5: Recommendations & Report
    ├── 修正提案 (High/Medium/Low priority)
    ├── 実装ロードマップ
    └── 自動報告書生成
```

---

## 2. 検査チェックリスト

### Phase 1: ドキュメント・レビュー

**対象ファイル**: documents/README.md, SKILL_IMPLEMENTATION_GUIDE.md, AI_AGENT_*.md等

チェック項目：
- [ ] 全ハイパーリンク有効性 (broken link なし)
- [ ] 用語の統一 (Skill, skill, SKILL が混在していないか)
- [ ] 図表・表の完全性 (セクション数の一貫性)
- [ ] 文書の階層構造 (深さ3以上なら再構成)
- [ ] 各セクションの冒頭に要約あり
- [ ] 古い情報 (2025年以前 or 廃止機能) なし

### Phase 2: Skills 一貫性チェック

**対象**: .github/skills/ の 5つのSkill

チェック項目：
- [ ] 各Skillの README で「対応ドキュメント」が明示されている
- [ ] CLI使用例（Claude/Copilot/Cursor）が統一フォーマット
- [ ] 自動化レベルが適切に記載されている（⭐マーク）
- [ ] チェックリスト・テンプレートが実装例付き
- [ ] 依存Skillが明記されている

### Phase 3: ツール・スクリプト確認

**対象**: scripts/ + .github/skills/ 内のスクリプト

チェック項目：
- [ ] 実装状況: ✅/🔄/❌ が明示
- [ ] 各スクリプトに Docstring あり
- [ ] エラーハンドリング適切 (exit code)
- [ ] 使用例が README に記載
- [ ] 依存パッケージが記載
- [ ] ローカル + CI/CD で実行可能

### Phase 4: 統合検証

**対象**: CLI連携 + GitHub Actions パイプライン

チェック項目：
- [ ] `--skill` flag が全 Skill で動作
- [ ] `.github/workflows/` にパイプライン定義あり
- [ ] 各Skillのタイムアウト設定適切
- [ ] Skill チェーン (A → B → C) が想定される
- [ ] ローカル実行方法が明確
- [ ] Docker環境での実行ガイドあり

### Phase 5: 報告書自動生成

出力形式：
```markdown
# Comprehensive Review Report

## 実施日時
2026-04-01 10:00:00

## 概要
- 文書数: X
- Skill数: 5
- スクリプト数: Y
- 重大な問題: Z個

## Phase 1: ドキュメント
### 問題点
- [HIGH] broken link: X件
- [MEDIUM] 用語混在: Y箇所
- [LOW] 古い情報: Z件

### 改善提案
1. ...
2. ...

## Phase 2: Skills
### 一貫性スコア: XX/100
### 問題点
...

## Phase 3: ツール
### 実装状況
- ✅ check_*.py: N個実装
- 🔄 run_comprehensive_review.sh: テスト中
- ❌ generate-report.py: 未実装

### 推奨修正順序 (ロードマップ)
Week 1:
- [ ] fix broken links
- [ ] unify terminology
...

## 総合判定
Grade: A/B/C/D/F

## 次のステップ
1. ...
2. ...
```

---

## 3. 実装方針

### スクリプト構成

```
.github/skills/06-comprehensive-review/
├── README.md                    # Skill説明
├── run-review.py               # メイン実行スクリプト
├── checkers/
│   ├── doc_checker.py          # Phase 1
│   ├── skills_checker.py       # Phase 2
│   ├── tools_checker.py        # Phase 3
│   ├── integration_checker.py  # Phase 4
│   └── report_generator.py     # Phase 5
├── config/
│   ├── doc-config.yaml         # ドキュメント設定
│   ├── skills-config.yaml      # Skill設定
│   └── tools-config.yaml       # ツール設定
└── config.yaml                 # Skill定義
```

### 手順フロー

```
1. User: claude --skill comprehensive-review
   ↓
2. run-review.py: 設定ロード + 実行parameter確認
   ↓
3. Phase 1-5 を順序実行（または指定phase のみ）
   ├→ issue 検出
   └→ JSON/Markdown結果出力
   ↓
4. report_generator.py: Markdown報告書生成
   ↓
5. Output: report-YYYYMMDD-HHmmss.md （自動保存）
```

---

## 4. 優先度別実装ロードマップ

### Week 1: 基礎実装
- [ ] .github/skills/06-comprehensive-review/ ディレクトリ作成
- [ ] README.md + 基本説明
- [ ] run-review.py (空のフレームワーク)
- [ ] doc_checker.py (broken link検出)

### Week 2: 中核機能
- [ ] skills_checker.py (依存関係検証)
- [ ] tools_checker.py (スクリプト状況把握)
- [ ] report_generator.py (基本形式)

### Week 3: 統合・テスト
- [ ] integration_checker.py (CLI/Actions)
- [ ] CLI からの実行テスト
- [ ] GitHub Actions パイプライン統合

### Week 4: 公開・改善
- [ ] ドキュメント完成
- [ ] 使用例・テストケース充実
- [ ] チーム向けトレーニング資料

---

## 5. 現状での推定 Issue リスト（プレビュー）

**予想される主要な問題点**:

1. **ドキュメント**:
   - broken links: 参考資料の PDF へのリンク確認
   - 用語: "Skill" vs "skill" vs "SKILL"
   - 古い情報: 2025年以前のコンテンツ確認

2. **Skills**:
   - Skill 3 (run-experiment) と Skill 5 (research-workflow) の重複
   - スクリプト実装状況の不一致（.md vs 実装）

3. **ツール**:
   - scripts/run_comprehensive_review.sh は存在するが、6つのSkillに完全対応していない可能性
   - check_*.py の実装状況バラバラ

4. **統合**:
   - GitHub Actions パイプラインが Skill 1-2 のみ対応かもしれない
   - Skill 6 自体の定義がないため、circular dependency の可能性

5. **推奨優先処理**:
   1. High: broken links 修正
   2. High: 用語統一
   3. Medium: Skills 依存関係の明確化
   4. Medium: スクリプト実装完成度向上
   5. Low: 古い情報の削除

---

## 次のステップ

1. このプラン確認 ✓
2. 実際の包括レビュー実施 (run-review.py で自動実行)
3. Issue 報告書生成
4. 修正タスク分配

---

**このドキュメントは、Skill 6 実装の設計仕様です。**  
**実装後、自動レビュー結果により更新されます。**
