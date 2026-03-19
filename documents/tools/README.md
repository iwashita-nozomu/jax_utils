# documents/tools

このディレクトリは、プロジェクトで使用するツール・スクリプトに関するドキュメントを置きます。

## ドキュメント一覧

### 1. [TOOLS_DIRECTORY.md](./TOOLS_DIRECTORY.md)（必読）
**ツール・スクリプト詳細目録**

- 全ツール・スクリプト（20個）の用途・使用法を説明
- カテゴリ別分類（Git管理、ドキュメント処理、設計管理など）
- ツール依存関係グラフ
- 使用フロー別ガイド（5種）

**対象:** 開発者全員  
**読む時期:** プロジェクト参画時・ツール使用時  
**所要時間:** 20分

---

### 2. [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md)（必読）
**作業別チェックリスト・実行手順書**

- **チェックリスト1～8：** 典型的な8つの作業フロー
  1. 新規開発ブランチ開始
  2. Python パッケージ作成
  3. コード実装＆テスト
  4. ドキュメント更新
  5. 設計ドキュメント整理
  6. ワークツリー完了＆クリーンアップ
  7. ワークツリー規約遵守チェック（管理者向け）
  8. ドキュメント品質チェック（管理者向け）

**対象:** 開発者全員  
**読む時期:** 新しい種類の作業を始める前  
**所要時間:** 10～30分（フロー別）

---

### 3. [WORKTREE_TOOL_UNIFICATION.md](./WORKTREE_TOOL_UNIFICATION.md)（参考）
**ツール統一提案：setup_worktree.sh vs create_worktree.sh**

- 現状問題分析
- 統一方針の提案
- 実装手順・テストチェックリスト
- 決定フロー

**対象:** プロジェクト管理者  
**読む時期:** ツール統一検討時  
**所要時間:** 10分

---

## 推奨読み順

### 新規参画者向け
1. [TOOLS_DIRECTORY.md](./TOOLS_DIRECTORY.md) — 全体像把握（20分）
2. [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md) — チェックリスト1（5分）
3. 実際に作業開始

### 特定の作業をする時
1. [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md) — 該当チェックリスト（5～30分）
2. [TOOLS_DIRECTORY.md](./TOOLS_DIRECTORY.md) — 必要なツール詳細を参照（3～5分）

### 管理者向け
1. [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md) — チェックリスト7・8（10～30分）
2. [WORKTREE_TOOL_UNIFICATION.md](./WORKTREE_TOOL_UNIFICATION.md) — 統一方針検討（10分）

---

## ツール・スクリプト一覧（簡易版）

| 用途 | ツール | パス |
| --- | --- | --- |
| **Git初期化** | `git_config.sh` / `git_init.sh` / `git_repo_init.sh` | `scripts/` |
| **規約表示** | `view_conventions.sh` / `read_conventions.sh` | `scripts/` |
| **ワークツリー** | `create_worktree.sh` / `setup_worktree.sh` | `scripts/tools/` / `scripts/` |
| **テスト実行** | `run_pytest_with_logs.sh` | `scripts/` |
| **ドキュメント処理** | `format_markdown.py` / `audit_and_fix_links.py` / `fix_markdown_docs.py` | `scripts/tools/` |
| **設計管理** | `organize_designs.py` / `find_redundant_designs.py` / `find_similar_designs.py` / `tfidf_similar_docs.py` | `scripts/tools/` |
| **HLO分析** | `summarize_hlo_jsonl.py` | `scripts/hlo/` |
| **ワークツリーチェック** | `check_worktree_scopes.sh` | `scripts/tools/` |

詳細は [TOOLS_DIRECTORY.md](./TOOLS_DIRECTORY.md) 参照。

---

## 関連ドキュメント

- [documents/README.md](../README.md) — ドキュメント体系全体
- [documents/coding-conventions-project.md](../coding-conventions-project.md) — プロジェクト運用規約
- [documents/worktree-lifecycle.md](../worktree-lifecycle.md) — ワークツリーライフサイクル規約
- [../../README.md](../../README.md) — プロジェクト全体 README

---

## よくある質問

### Q: 新規ブランチを作成したいのですが、どのスクリプトを使えばいいですか？

**A:** [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md) の **チェックリスト1** を参照してください。

推奨コマンド：
```bash
bash scripts/tools/create_worktree.sh my-feature-name
```

### Q: テストを実行してログを保存したいのですが？

**A:** [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md) の **チェックリスト3** ステップ2 を参照。

推奨コマンド：
```bash
bash scripts/run_pytest_with_logs.sh
```

### Q: Markdown ドキュメントを整形・修正したいのですが？

**A:** [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md) の **チェックリスト4** / **チェックリスト8** を参照。

推奨コマンド：
```bash
python scripts/tools/format_markdown.py
python scripts/tools/audit_and_fix_links.py
```

### Q: 設計ファイルが重複しているようですが、整理できますか？

**A:** [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md) の **チェックリスト5** を参照。

推奨コマンド：
```bash
python scripts/tools/find_redundant_designs.py
python scripts/tools/tfidf_similar_docs.py
```

### Q: setup_worktree.sh と create_worktree.sh の違いは？

**A:** [WORKTREE_TOOL_UNIFICATION.md](./WORKTREE_TOOL_UNIFICATION.md) で詳しく説明しています。

**現在の推奨:** `scripts/tools/create_worktree.sh` を使用してください。

---

## ツール実行に必要な環境

### 必須
- Bash 4.0+
- Python 3.6+（一部ツール）
- Git 2.20+

### 推奨ツール
- `pyright` — Python型チェック
- `ruff` — Python リント・フォーマット
- `black` — Python コードフォーマッタ
- `pytest` — Python テストフレームワーク
- `markdownlint` — Markdown チェック（オプション）

### インストール
```bash
# Docker 環境を使用している場合
docker/requirements.txt を確認し、pip install で追加

# ローカル環境の場合
pip install -e ".[dev]"  # pyproject.toml に基づいて
```

---

## サポート・問題報告

### 実行に失敗した場合
1. [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md) の **トラブルシューティング** セクション参照
2. スクリプトのヘッダーコメント確認
3. `--help` オプション試行（対応ツールのみ）

### ジョイント・改善提案
- ツール追加・改修提案は `documents/WORKFLOW_INVENTORY.md` で管理
- 優先度判定後、実装を検討

---

## 変更履歴

| 日付 | 変更事項 |
| --- | --- |
| 2026-03-19 | 初期版作成。ツール目録・チェックリスト・統一提案を整備 |

