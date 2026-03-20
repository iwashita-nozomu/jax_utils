# 📚 ドキュメント ハブ

> jax_util プロジェクトの全階層・全分野向けドキュメント入口

______________________________________________________________________

## 🎯 目的別ナビゲーション

### 👤 **初心者向け（最初の 1 時間で読むべき）**

| 項目                 | 場所                                             | 読む時間 | 目的                   |
| -------------------- | ------------------------------------------------ | -------- | ---------------------- |
| **環境セットアップ** | [セットアップガイド](./worktree-lifecycle.md)    | 20分     | ワークツリー・環境作成 |
| **最初の実装**       | [コーディング規約](./conventions/README.md)      | 15分     | Python 実装ルール      |
| **作業フロー**       | [チェックリスト](./FILE_CHECKLIST_OPERATIONS.md) | 15分     | 8 種類の作業手順       |
| **テスト方法**       | [テスト規約](./coding-conventions-testing.md)    | 10分     | ユニット・統合テスト   |

**👉 推奨順序：** セットアップ → コーディング規約 → チェックリスト → テスト

______________________________________________________________________

### 👨‍💻 **実装者向け（プロジェクトの全体像）**

#### 規約・ガイドライン

- **[コーディング規約 (全言語)](./conventions/README.md)** — Python・C++ 共通 + 言語別ガイド
- **[Python 実装ガイド](./coding-conventions-python.md)** — 型注釈・関数・モジュール構成
- **[テスト規約](./coding-conventions-testing.md)** — ユニット・統合・エッジケーステスト
- **[Markdown 記法](./coding-conventions.md)** — ドキュメント統一・図解方法

#### 設計・アーキテクチャ

- **[プロジェクト設計](./design/README.md)** — モジュール依存・API 構造
- **[型システム・基本型](./type-aliases.md)** — 型エイリアス・Protocol 定義
- **[基本コンポーネント](./design/base_components.md)** — LinOp・linearize・adjoint

#### ツール・スクリプト

- **[ツール一覧](./tools/README.md)** — Markdown・テスト・解析ツール
- **[ツール詳細リファレンス](./tools/TOOLS_DIRECTORY.md)** — コマンド・オプション・フロー

______________________________________________________________________

### 🔧 **レビュー・運用向け**

- **[レビュー手順](./REVIEW_PROCESS.md)** — 実装人間系・検査項目・チェックリスト
- **[チーム調整](./AGENTS_COORDINATION.md)** — ロール・責任・コミュニケーション
- **[ワークツリー管理](./WORKTREE_SCOPE_TEMPLATE.md)** — スコープ・ブランチ管理

______________________________________________________________________

### 🎓 **設計詳細（深掘り参照）**

#### API・ソルバー設計

- **[Solvers 設計](./design/jax_util/README.md)** — シンプル KKT・PDIPM・LOBPCG
- **[補助設計文書](./design/)** — 実験・拡張提案

#### 実験・試行段階

- **[実験環境](./coding-conventions-experiments.md)** — HLO 採取・ベンチマーク
- **[レビュー文書](./coding-conventions-reviews.md)** — AI レビュー・解析結果

______________________________________________________________________

### ❓ **困ったとき**

- **[よくある質問](./TROUBLESHOOTING.md)** — エラー・解決策・ベストプラクティス
- **[ツール一覧](./tools/README.md)** → 対応ツールで解決
- **[チェックリスト](./FILE_CHECKLIST_OPERATIONS.md)** → 作業フロー手順確認

______________________________________________________________________

## 📑 **ドキュメント 階層構成**

```text
documents/ (ここ) ← 📍 ハブ

 📋 規約・方針（最初に読む）
  ├─ conventions/README.md ← コーディング規約の主入口
  ├─ conventions/python/ (Python ガイド詳細)
  ├─ coding-conventions.md (共通ルール)
  ├─ coding-conventions-testing.md (テスト)
  ├─ coding-conventions-python.md (Python 詳細)
  └─ coding-conventions-experiments.md (実験)

 🏛️ 設計・アーキテクチャ
  ├─ design/README.md (設計 入口)
  ├─ design/base_components.md (型・Protocol)
  ├─ design/jax_util/README.md (ソルバー)
  └─ type-aliases.md (型定義)

 🛠️ ツール・スクリプト
  ├─ tools/README.md (ツール概要)
  ├─ tools/TOOLS_DIRECTORY.md (詳細リファレンス)
  └─ tools/check_markdown_lint.py (ツール実行体)

 📊 運用・管理
  ├─ FILE_CHECKLIST_OPERATIONS.md (作業チェックリスト)
  ├─ REVIEW_PROCESS.md (レビュー手順)
  ├─ AGENTS_COORDINATION.md (チーム・ロール)
  ├─ WORKTREE_SCOPE_TEMPLATE.md (スコープ テンプレート)
  └─ worktree-lifecycle.md (ワークツリー ライフサイクル)

 ❓ ヘルプ・参考
   ├─ TROUBLESHOOTING.md ← エラー解決
   ├─ DOCUMENT_RESTRUCTURING_PROPOSAL.md (構造改善提案)
   └─ reviews/ (AI レビュー・解析)
```

______________________________________________________________________

## 🚀 **クイックアクセス**

### よくある作業

| 作業              | 参照先                                                                                                             | 時間 |
| ----------------- | ------------------------------------------------------------------------------------------------------------------ | ---- |
| 新規ブランチ作成  | [チェックリスト1](./FILE_CHECKLIST_OPERATIONS.md#%E3%83%81%E3%82%A7%E3%83%83%E3%82%AF%E3%83%AA%E3%82%B9%E3%83%881) | 20分 |
| Python コード実装 | [Python 規約](./coding-conventions-python.md)                                                                      | 15分 |
| テスト作成        | [テスト規約](./coding-conventions-testing.md)                                                                      | 10分 |
| ドキュメント追加  | [MD 記法](./coding-conventions.md#markdown-%E6%9B%B8%E5%BC%8F%E4%BF%AE%E6%AD%A3%E3%83%AB%E3%83%BC%E3%83%AB)        | 10分 |
| レビュー実施      | [レビュー手順](./REVIEW_PROCESS.md)                                                                                | 30分 |
| Markdown リント   | [ツール](./tools/README.md#markdown-%E3%83%84%E3%83%BC%E3%83%AB)                                                   | 5分  |

______________________________________________________________________

## 📝 **追記・更新ルール**

### 規約を追加する場合

1. **汎用 / 共通ルール** → `conventions/common/` に追記
1. **Python 固有** → `conventions/python/` に追記
1. **テスト・実験・レビュー** → `coding-conventions-*.md` に追記

### 設計仕様を追加する場合

1. **基本型・Protocol** → `design/base_components.md` 更新
1. **ソルバー・API** → `design/jax_util/*.md` に新規追記
1. **実験段階** → `design/` 直下に独立ファイル作成

### 文書管理の指針

- ✅ テーマごと独立ファイル化 → 参照の連鎖を避ける
- ✅ 各ファイルは「それ単体で読める」状態維持
- ✅ リンク集（このファイル）で逆引きできる
- ✅ Markdown は `mdformat`・`markdownlint` 統一

______________________________________________________________________

## 🔗 **外部リンク**

- **最初に読む：** [root/README.md](../README.md) → クイックスタート
- **コマンド参考：** [QUICK_START.md](../QUICK_START.md)
- **プロジェクト情報：** [pyproject.toml](../pyproject.toml)

______________________________________________________________________

## 📋 **現在の構成（トリプル層）**

### 1. 一般的なコーディング方針

- `coding-conventions.md` （共通・Markdown）
- `coding-conventions-python.md` （Python 詳細）
- `coding-conventions-testing.md` （テスト）
- `coding-conventions-experiments.md` （実験）
- `coding-conventions-reviews.md` （レビュー）
- `conventions/` （言語別ガイド）
- **対象:** 命名・型注釈・JAX 運用・テスト・ログなど共通ルール

### 2. ベースコンポーネント設計

- `design/base_components.md`
- `type-aliases.md`
- **対象:** `LinOp`・Protocol・型エイリアス

### 3. 安定 API 詳細設計

- `design/` （ソルバー・最適化・HLO）
- **対象:** サブモジュール単位の API 仕様
