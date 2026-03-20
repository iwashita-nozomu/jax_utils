# 🛠️ ツール実行ガイド（documents/tools）

> プロジェクトのツール・スクリプトを **効率的に実行するための入口**

## 📍 このドキュメントの役割

- ✅ **ツール選択の判断** — 何をしたいか → どのツール？
- ✅ **使用方法の確認** — 各ツール・チェックリストの読み方
- ✅ **詳細の参照先** — 掘り下げたいときは TOOLS_DIRECTORY.md へ

⚠️ **注意：** スクリプト実装の詳細は [`scripts/README.md`](../../scripts/README.md) を参照してください。

______________________________________________________________________

## ドキュメント一覧

### 1. [TOOLS_DIRECTORY.md](./TOOLS_DIRECTORY.md)（必読）

**ツール・スクリプト詳細目録**

- 全ツール・スクリプト（20個）の用途・使用法を説明
- カテゴリ別分類（Git管理、ドキュメント処理、設計管理など）
- ツール依存関係グラフ
- 使用フロー別ガイド（5種）

**対象:** 開発者全員\
**読む時期:** プロジェクト参画時・ツール使用時\
**所要時間:** 20分

______________________________________________________________________

### 2. [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md)（必読）

**作業別チェックリスト・実行手順書**

- **チェックリスト1～8：** 典型的な8つの作業フロー
  1. 新規開発ブランチ開始
  1. Python パッケージ作成
  1. コード実装＆テスト
  1. ドキュメント更新
  1. 設計ドキュメント整理
  1. ワークツリー完了＆クリーンアップ
  1. ワークツリー規約遵守チェック（管理者向け）
  1. ドキュメント品質チェック（管理者向け）

**対象:** 開発者全員\
**読む時期:** 新しい種類の作業を始める前\
**所要時間:** 10～30分（フロー別）

______________________________________________________________________

### 3. [WORKTREE_TOOL_UNIFICATION.md](./WORKTREE_TOOL_UNIFICATION.md)（参考）

**ツール統一提案：setup_worktree.sh vs create_worktree.sh**

- 現状問題分析
- 統一方針の提案
- 実装手順・テストチェックリスト
- 決定フロー

**対象:** プロジェクト管理者\
**読む時期:** ツール統一検討時\
**所要時間:** 10分

______________________________________________________________________

## 推奨読み順

### 新規参画者向け

1. [TOOLS_DIRECTORY.md](./TOOLS_DIRECTORY.md) — 全体像把握（20分）
1. [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md) — チェックリスト1（5分）
1. 実際に作業開始

### 特定の作業をする時

1. [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md) — 該当チェックリスト（5～30分）
1. [TOOLS_DIRECTORY.md](./TOOLS_DIRECTORY.md) — 必要なツール詳細を参照（3～5分）

### 管理者向け

1. [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md) — チェックリスト7・8（10～30分）
1. [WORKTREE_TOOL_UNIFICATION.md](./WORKTREE_TOOL_UNIFICATION.md) — 統一方針検討（10分）

______________________________________________________________________

## 🔄 CI スクリプト実行ガイド（scripts/ci）

> GitHub Actions で実行する CI チェックをローカルで実行するためのガイド

### 目的

- 開発中に CI が失敗することを防ぐ
- リモート CI 実行前にバグを検出
- CI と開発環境の一貫性確保

### 主なスクリプト

#### run_all_checks.sh（★推奨: 統合テスト）

**用途:** pytest + pyright + ruff を一括実行

**実行方法:**

```bash
# 標準実行（全チェック）
bash scripts/ci/run_all_checks.sh

# 高速モード（ruff をスキップ）
bash scripts/ci/run_all_checks.sh --quick

# 詳細出力
bash scripts/ci/run_all_checks.sh --verbose
```yaml

**実行内容:**

1. `pytest python/tests/` — ユニット・統合テスト
1. `pyright python/` — Python 型チェック
1. `ruff check python/` — リント・スタイルチェック

**戻り値:**

- `0`: すべて成功
- `1`: テスト失敗 または致命的エラー

**所要時間:** 30秒～2分（マシン依存）

## 推奨ワークフロー

```bash
# 1. コード編集
vim python/jax_util/module/file.py

# 2. ローカルで CI チェック（失敗時は修正）
bash scripts/ci/run_all_checks.sh

# 3. 成功後、コミット
git add -A
git commit -m "feat: new feature"

# 4. リモート push
git push origin branch-name
```text

## GitHub Actions との関係

`.github/workflows/ci.yml` はこのスクリプト（`run_all_checks.sh`）を呼び出します：

```yaml
- name: Run all checks
  run: bash scripts/ci/run_all_checks.sh
```yaml

**つまり:** ローカルで成功 = リモート CI でも成功（高確率）

### トラブルシューティング

#### pytest 失敗時

```bash
# 失敗したテスト単体を再実行
pytest python/tests/test_module.py::TestClass::test_method -v

# 失敗情報を詳しく表示
pytest python/tests/ -v --tb=long
```yaml

詳細: [FILE_CHECKLIST_OPERATIONS.md#チェックリスト3](../FILE_CHECKLIST_OPERATIONS.md)

## pyright 警告が多い場合

```bash
# 警告ファイルをフィルタ表示
pyright python/ 2>&1 | grep "error"

# 特定モジュールのみチェック
pyright python/jax_util/solvers/
```text

## ruff エラー修正

```bash
# 自動修正
ruff check --fix python/

# 修正内容確認
git diff python/
```text

## 依存パッケージが不足の場合

```bash
# 環境確認
python -c "import pytest, pyright, ruff; print('OK')"

# インストール
pip install -r docker/requirements.txt

# または Docker 使用
docker build -t jax-util -f docker/Dockerfile .
docker run --rm -v $(pwd):/workspace -w /workspace jax-util bash scripts/ci/run_all_checks.sh
```text

## カスタマイズ

### PYTHONPATH（自動設定）

スクリプトは自動的に以下を設定します：

```bash
export PYTHONPATH="/workspace/python:${PYTHONPATH:-}"
```yaml

参照: [coding-conventions-project.md](../coding-conventions-project.md)

______________________________________________________________________

## ツール・スクリプト一覧（簡易版）

| 用途                     | ツール                                                                                                    | パス                          |
| ------------------------ | --------------------------------------------------------------------------------------------------------- | ----------------------------- |
| **Git初期化**            | `git_config.sh` / `git_init.sh` / `git_repo_init.sh`                                                      | `scripts/`                    |
| **規約表示**             | `view_conventions.sh` / `read_conventions.sh`                                                             | `scripts/`                    |
| **ワークツリー**         | `create_worktree.sh` / `setup_worktree.sh`                                                                | `scripts/tools/` / `scripts/` |
| **テスト実行**           | `run_pytest_with_logs.sh`                                                                                 | `scripts/`                    |
| **ドキュメント処理**     | `format_markdown.py` / `audit_and_fix_links.py` / `fix_markdown_docs.py`                                  | `scripts/tools/`              |
| **設計管理**             | `organize_designs.py` / `find_redundant_designs.py` / `find_similar_designs.py` / `tfidf_similar_docs.py` | `scripts/tools/`              |
| **HLO分析**              | `summarize_hlo_jsonl.py`                                                                                  | `scripts/hlo/`                |
| **ワークツリーチェック** | `check_worktree_scopes.sh`                                                                                | `scripts/tools/`              |

詳細は [TOOLS_DIRECTORY.md](./TOOLS_DIRECTORY.md) 参照。

______________________________________________________________________

## 関連ドキュメント

- [documents/README.md](../README.md) — ドキュメント体系全体
- [documents/coding-conventions-project.md](../coding-conventions-project.md) — プロジェクト運用規約
- [documents/worktree-lifecycle.md](../worktree-lifecycle.md) — ワークツリーライフサイクル規約
- [../../README.md](../../README.md) — プロジェクト全体 README

______________________________________________________________________

## よくある質問

### Q: 新規ブランチを作成したいのですが、どのスクリプトを使えばいいですか？

**A:** [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md) の **チェックリスト1** を参照してください。

推奨コマンド：

````bash
bash scripts/tools/create_worktree.sh my-feature-name
```yaml

### Q: テストを実行してログを保存したいのですが？

**A:** [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md) の **チェックリスト3** ステップ2 を参照。

推奨コマンド：

```bash
bash scripts/run_pytest_with_logs.sh
```yaml

### Q: Markdown ドキュメントを整形・修正したいのですが？

**A:** [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md) の **チェックリスト4** / **チェックリスト8** を参照。

推奨コマンド：

```bash
python scripts/tools/format_markdown.py
python scripts/tools/audit_and_fix_links.py
```yaml

### Q: 設計ファイルが重複しているようですが、整理できますか？

**A:** [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md) の **チェックリスト5** を参照。

推奨コマンド：

```bash
python scripts/tools/find_redundant_designs.py
python scripts/tools/tfidf_similar_docs.py
```yaml

### Q: setup_worktree.sh と create_worktree.sh の違いは？

**A:** [WORKTREE_TOOL_UNIFICATION.md](./WORKTREE_TOOL_UNIFICATION.md) で詳しく説明しています。

**現在の推奨:** `scripts/tools/create_worktree.sh` を使用してください。

______________________________________________________________________

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
```text

______________________________________________________________________

## サポート・問題報告

### 実行に失敗した場合

1. [FILE_CHECKLIST_OPERATIONS.md](../FILE_CHECKLIST_OPERATIONS.md) の **トラブルシューティング** セクション参照
1. スクリプトのヘッダーコメント確認
1. `--help` オプション試行（対応ツールのみ）

### ジョイント・改善提案

- ツール追加・改修提案は `documents/WORKFLOW_INVENTORY.md` で管理
- 優先度判定後、実装を検討

______________________________________________________________________

## 変更履歴

| 日付       | 変更事項                                               |
| ---------- | ------------------------------------------------------ |
| 2026-03-19 | 初期版作成。ツール目録・チェックリスト・統一提案を整備 |
````
