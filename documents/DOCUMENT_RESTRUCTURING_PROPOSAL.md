# 📋 ドキュメント構造化提案書

**作成日:** 2026-03-19  
**対象:** root README + documents ハブ化  
**目的:** 初心者がスムーズに理解できる階層構造

---

## 現状分析

### 総ドキュメント数: 119ファイル

| 型 | 数 | 行数 | 用途 |
|---|---|---|---|
| **README** | 21 | 1,482 | ナビゲーション（現在冗長） |
| **NOTES** | 23 | 3,520 | 実験記録・ワークツリー |
| **OTHER** | 42 | 3,233 | ツール・設計・規約各種 |
| **REVIEW** | 9 | 2,358 | AI レビュー・進捗報告 |
| **CONVENTION** | 9 | 589 | 実装規約 |
| **AGENT/WORKFLOW** | 4 | 903 | 作業フロー・チェックリスト |
| **DIARY** | 8 | 1,670 | 開発ログ |

### 問題点

1. **迷路化** — README.md が 21 個、初心者が迷う
2. **冗長性** — documents/, notes/, reviews/ が独立していて連携不十分
3. **エントリーポイント不在** — root/README.md が薄い
4. **実験・作業ログ** — notes/ が雑多（整理不足）

---

## 推奨構造（3層構造化）

### 層 1: ルートレベル（入口）

**`README.md`（新規作成・拡張）**

```
# jax_util

## 🚀 クイックスタート
→ QUICK_START.md へ

## 📖 ドキュメント
→ documents/README.md へ

## 🛠️ ツールとスクリプト
→ scripts/README.md へ

## 📓 実験・開発ログ
→ notes/README.md へ

## 👥 チーム・ワークフロー
→ .github/AGENTS.md へ
```

---

### 層 2: documents/README.md（ハブ）

**中核ハブドキュメント**

```
# 📚 プロジェクトドキュメント

## 1. 新規開発者向け
├─ [コーディング規約](./conventions/README.md)
├─ [ワークツリー規約](./WORKTREE_SCOPE_TEMPLATE.md)
└─ [テスト規約](./coding-conventions-testing.md)

## 2. 実装ガイド
├─ [Python 実装](./coding-conventions-python.md)
├─ [ツール使用方法](./tools/README.md)
└─ [ワークフロー](./FILE_CHECKLIST_OPERATIONS.md)

## 3. 設計・仕様
├─ [プロジェクト設計](./design/README.md)
└─ [構成](./conventions/README.md)

## 4. 運用
├─ [レビュー手順](./REVIEW_PROCESS.md)
└─ [チーム調整](./AGENTS_COORDINATION.md)

## 5. トラブルシューティング
└─ [よくある問題](./TROUBLESHOOTING.md) ← NEW
```

---

### 層 3: サブディレクトリ（詳細）

#### documents/tools/ (ツール情報)
```
├─ README.md (ツール一覧・使用方法)
├─ check_markdown_lint.py (マークダウンチェッカー)
├─ fix_markdown_headers.py (ヘッダーレベル修正)
├─ fix_markdown_code_blocks.py (言語指定修正)
└─ TOOLS_DIRECTORY.md (詳細リファレンス)
```

#### documents/conventions/ (規約体系)
```
├─ README.md (規約の全体像)
├─ python/
│  └─ {01-20}_*.md (各章詳細)
└─ common/
   └─ {01-05}_*.md
```

#### notes/ (実験・記録)
```
├─ README.md (ナビゲーション)
├─ experiments/ (実験結果)
├─ worktrees/ (ワークツリー記録)
├─ knowledge/ (知識ベース)
├─ themes/ (お題別記録)
└─ branches/ (ブランチ記録)
```

---

## 具体的な改善案

### 1. 冗長性除去

| 現状 | 改善案 |
|---|---|
| 21 個の README.md | 各層で 1 つの README.md + トップレベル 1 個に集約 |
| REVIEW/ 9 ファイル（AI レビュー）| archives/ に移動（参考資料扱い） |
| DIARY/ 8 ファイル | notes/worktrees/ に統合 |

### 2. ナビゲーション改善

**追加ドキュメント（新規）:**
- `README.md` (root) — 入口グローバルナビゲーション
- `documents/README.md` — ハブ（現在の README.md から拡張）
- `documents/TROUBLESHOOTING.md` — よくある問題
- `NAVIGATION.md` (root) — ドキュメント体系図

### 3. サムネイル / クイックリンク

```markdown
## 🔍 よく使うリンク

| 用途 | リンク | 所要時間 |
|---|---|---|
| 環境構築 | [セットアップ](./documents/setup.md) | 10 分 |
| 最初のコミット | [ワークツリー作成](./documents/worktree-lifecycle.md) | 5 分 |
| コード規約 | [Python 規約](./documents/conventions/python/README.md) | 読む |
| テスト実行 | [テスト手順](./documents/coding-conventions-testing.md) | 5 分 |
| ツール使用 | [ツール一覧](./documents/tools/README.md) | リークエスト |
```

---

## 実装順序

1. **root/README.md 拡張** ← **即座**
   - 3 層構造を明示
   - 各層への相対パスリンク

2. **documents/README.md ハブ化**
   - 現在の documents/README.md を基に拡張
   - 4-5 セクション体系化

3. **documents/TROUBLESHOOTING.md 新規** ← **優先**
   - よくあるエラー
   - 解決策リンク

4. **notes/README.md 強化**
   - 実験ナビゲーション
   - 日記 → worktrees/ 統合提案

5. **不要ファイル整理**
   - reviews/ → archives/REVIEWS/ 移行提案
   - AGENT/ 系も .github/ に統合検討

---

## 初心者向けナビゲーション経路（想定シナリオ）

### シナリオ 1: 環境構築したい

```
1. README.md 開く
   ↓
2. 「🚀 クイックスタート」クリック (QUICK_START.md)
   ↓
3. 「ワークツリー作成」リンク
   ↓
4. WORKTREE_SCOPE_TEMPLATE.md
   ↓
5. 完成！
```

### シナリオ 2: コード書きたい

```
1. README.md 開く
   ↓
2. 「📖 ドキュメント」クリック (documents/README.md)
   ↓
3. 「実装ガイド」→「Python 実装」クリック
   ↓
4. documents/coding-conventions-python.md
   ↓
5. 規約理解 + 実装開始
```

### シナリオ 3: テスト実行したい

```
1. README.md 開く
   ↓
2. 「🛠️ ツール」クリック (scripts/README.md)
   ↓
3. 「テスト実行」セクション
   ↓
4. `make ci` 実行
   ↓
5. 完成！
```

---

## 決定項目

### 保持もしくは削除

- [ ] reviews/ → archives/REVIEWS/ に移行?
- [ ] diary/ → notes/worktrees/ に統合?
- [ ] reports/COMPLETE_PROJECT_MAPPING_20260319.md → documents/PROJECT_MAP.md か?

### 新規作成

- [ ] `root/README.md` 拡張
- [ ] `documents/README.md` ハブ化
- [ ] `documents/TROUBLESHOOTING.md`
- [ ] `root/NAVIGATION.md` (全体図)

---

## 推定効果

| 指標 | 現状 | 改善後 |
|---|---|---|
| ナビゲーション段数 | 3～4 | 2～3 |
| README.md 数 | 21 → ? | 8～10 |
| 初心者の迷い | 高 | 低 |
| ドキュメント一貫性 | 低 | 高 |
