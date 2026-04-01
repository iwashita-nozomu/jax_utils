# Skill: `code-review` — コードレビュースキル

**目的**: Python/C++ コードの品質を A 層 → B 層 → C 層の多層検証で保証

**対応ドキュメント**:
- `documents/SKILL_IMPLEMENTATION_GUIDE.md` の A/B/C/D 層
- `documents/REVIEW_PROCESS.md`
- `documents/coding-conventions-reviews.md`

---

## 概要

このスキルは、PR レビューの標準化されたワークフローを提供します。

### 検証レイヤー

| レイヤー | チェック対象 | 実施主体 | 自動化 |
|---------|------------|--------|------|
| **A 層** | 型注釈・Docstring・テスト | CI + 開発者 | ⭐⭐⭐⭐⭐ |
| **B 層** | テストアーキテクチャ・スタイル | レビュアー | ⭐⭐⭐ |
| **C 層** | プロジェクト規約整合・三点セット | レビュアー | ⭐⭐⭐ |
| **D 層** | レビュー実施フロー | Maintainer | ⭐⭐ |

---

## 使用方法

### CLI で実行

```bash
# 対話型レビュー（全フェーズ）
claude --skill code-review --interactive

# PR 自動レビュー（A層のみ）
copilot --skill code-review --pr 42 --phase A

# 全フェーズ実施
copilot review --skill code-review https://github.com/user/repo/pull/42
```

### ローカルテンプレート参照

```bash
# A 層チェックリスト確認
cat .github/skills/02-code-review/checklist-a-layer.md

# B 層チェックリスト確認
cat .github/skills/02-code-review/checklist-b-layer.md

# レポートテンプレート確認
cat .github/skills/02-code-review/report-template.md
```

---

## スクリプト詳細

### `check_doc_test_triplet.py` — 三点セット検証

Python ファイルの「実装 ⇔ Docstring ⇔ テスト」の対応を検証。

```bash
python .github/skills/02-code-review/check_doc_test_triplet.py [FILES...]
```

検証項目：
- 全公開関数に Docstring あり
- Docstring に Args / Returns / Raises
- 全公開関数にテストあり
- Docstring と実装の型一致

出力：
- 許容範囲: `PASS`
- マイナー: `WARN` (改善推奨)
- 要修正: `FAIL` (修正必須)

---

## チェックリスト概要

### A 層: 基礎検証

```markdown
## Python コード
- [ ] 全関数に型注釈あり (`-> Type:`)
- [ ] 全関数に Docstring あり
- [ ] Docstring に Args / Returns / Raises
- [ ] エラーハンドリングは具体的 (generic except: なし)
- [ ] doctest が実行可能

## C++ コード
- [ ] ヘッダガード `#pragma once`
- [ ] メモリ安全 (生ポインタなし)
- [ ] 例外安全性確認
- [ ] インデント統一（スペース数）

## ドキュメント
- [ ] Markdown フォーマット統一 (mdformat OK)
- [ ] リンク参照 (内外リンク確認)
- [ ] 画像・図表 が accessible
```

### B 層: 深度検証

```markdown
## Python テスト・アーキテクチャ
- [ ] テストカバレッジ 70% 以上
- [ ] エッジケース テストあり
- [ ] 統合テスト存在
- [ ] Mock/Patch 適切に使用

## C++ スタイル・ベストプラクティス
- [ ] const 正当性
- [ ] 移動セマンティクス活用
- [ ] RAII 原則守られている
- [ ] ネーミング統一
```

### C 層: 統合検証

```markdown
## プロジェクト規約整合性
- [ ] `coding-conventions-*.md` に従う
- [ ] `.ruff.toml` / `pyproject.toml` に従う
- [ ] `documents/design/` とのアーキテクチャ一致

## 三点セット検証
- [ ] 実装 ⇔ Docstring 型一致
- [ ] 実装 ⇔ テスト カバレッジ適切
- [ ] Doc ⇔ テスト と両立

## 規約矛盾検出
- [ ] 既存関数と引数順序・命名統一
- [ ] エラーメッセージフォーマット統一
```

---

## 報告書テンプレート

PR レビュー完了後、以下テンプレートで報告書を作成：

```markdown
# Code Review Report

## 対象
- PR: #42
- ファイル: src/solver.py
- 行数変更: +45, -10

## A 層: 基礎検証
### 結果: ✅ PASS
- 型注釈: OK (全関数)
- Docstring: OK (全関数)
- テスト: OK (新テスト 3 個追加)

## B 層: 深度検証
### 結果: ⚠️ WARN
- テスト覆率: 75% (目標 70%) ✅
- エッジケース: 境界値テスト追加推奨
- 統合テスト: 新規フロー未テスト → 追加推奨

## C 層: 統合検証
### 結果: ✅ PASS
- 規約整合: OK
- 三点セット: OK
- 既存実装との矛盾: なし

## 承認判定
- **総合**: ✅ **APPROVED** （B層マイナー指摘は optional fix）
- **条件**: 上記マイナー指摘への対応は merge 後の follow-up PR で OK

## 結論
実装は良好。学術的価値も確認。推奨修正を付けて承認。
```

---

## 実装状況

| 要素 | 状態 | 備考 |
|------|------|------|
| checklist-a-layer.md | ✅ 実装 | Python/C++/Doc 統合 |
| checklist-b-layer.md | ✅ 実装 | テスト・アーキテクチャ |
| checklist-c-layer.md | ✅ 実装 | 規約・三点セット |
| check_doc_test_triplet.py | ✅ 実装 | 自動検証スクリプト |
| report-template.md | 🔄 開発中 | レポート定型化 |
| config.yaml | 🔄 開発中 | Skill 定義 |

---

## 参考資料

- **実装ガイド A-D 層**: `documents/SKILL_IMPLEMENTATION_GUIDE.md`
- **レビュー手順**: `documents/REVIEW_PROCESS.md`
- **統合計画**: `.github/SKILLS_INTEGRATION_PLAN.md`
