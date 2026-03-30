# scripts — 開発補助スクリプト・ツール集

このディレクトリは、全24個のツール・スクリプト、CI パイプライン、初期化スクリプトを統合管理します。

**最後に更新:** 2026-03-19

______________________________________________________________________

## 📋 全ツール一覧（24個）

| #                          | カテゴリ       | スクリプト                | パス                                                                   | 用途                                                   |
| -------------------------- | -------------- | ------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------ |
| **初期化**                 |                |                           |                                                                        |                                                        |
| 1                          | 初期化         | git_config.sh             | [./git_config.sh](./git_config.sh)                                     | Git ユーザー設定（user.name, email）                   |
| 2                          | 初期化         | git_init.sh               | [./git_init.sh](./git_init.sh)                                         | リポジトリ初期化（git_config.sh 呼び出し）             |
| 3                          | 初期化         | git_repo_init.sh          | [./git_repo_init.sh](./git_repo_init.sh)                               | 新規 Python パッケージディレクトリ作成                 |
| 4                          | 初期化         | create_toml.sh            | [./create_toml.sh](./create_toml.sh)                                   | pyproject.toml テンプレート作成                        |
| **ワークツリー・ブランチ** |                |                           |                                                                        |                                                        |
| 5                          | ワークツリー   | setup_worktree.sh         | [./setup_worktree.sh](./setup_worktree.sh)                             | ワークツリー・ブランチ作成（★推奨・正本）              |
| 6                          | ワークツリー   | create_worktree.sh        | [./tools/create_worktree.sh](./tools/create_worktree.sh)               | `setup_worktree.sh` 互換ラッパー                       |
| 7                          | 情報表示       | guide.sh                  | [./guide.sh](./guide.sh)                                               | 作業ガイド＆ワークツリー状況表示                       |
| 8                          | 情報表示       | view_conventions.sh       | [./view_conventions.sh](./view_conventions.sh)                         | 規約ファイルを検索・表示                               |
| 9                          | 情報表示       | read_conventions.sh       | [./read_conventions.sh](./read_conventions.sh)                         | 規約ファイル一覧を表形式で表示                         |
| **CI・テスト**             |                |                           |                                                                        |                                                        |
| 10                         | CI             | run_all_checks.sh         | [./ci/run_all_checks.sh](./ci/run_all_checks.sh)                       | pytest + pyright + pydocstyle + ruff 一括実行（★推奨） |
| 11                         | テスト         | run_pytest_with_logs.sh   | [./run_pytest_with_logs.sh](./run_pytest_with_logs.sh)                 | pytest 実行＆ログを時系列保存                          |
| **ドキュメント処理**       |                |                           |                                                                        |                                                        |
| 12                         | ドキュメント   | format_markdown.py        | [./tools/format_markdown.py](./tools/format_markdown.py)               | Markdown ファイル正規化（改行・空白等）                |
| 13                         | ドキュメント   | fix_markdown_docs.py      | [./tools/fix_markdown_docs.py](./tools/fix_markdown_docs.py)           | Markdown ドキュメント品質修正                          |
| 14                         | ドキュメント   | audit_and_fix_links.py    | [./tools/audit_and_fix_links.py](./tools/audit_and_fix_links.py)       | リンク監査・修正（相対パス統一）                       |
| **設計ファイル管理**       |                |                           |                                                                        |                                                        |
| 15                         | 設計           | organize_designs.py       | [./tools/organize_designs.py](./tools/organize_designs.py)             | 設計ファイルをサブモジュール別に整理                   |
| 16                         | 設計           | create_design_template.py | [./tools/create_design_template.py](./tools/create_design_template.py) | サブモジュール用設計テンプレート生成                   |
| 17                         | 設計           | find_redundant_designs.py | [./tools/find_redundant_designs.py](./tools/find_redundant_designs.py) | 完全一致の重複設計ファイル検出                         |
| 18                         | 設計           | find_similar_designs.py   | [./tools/find_similar_designs.py](./tools/find_similar_designs.py)     | 内容類似の設計ファイル検出（TF-IDF)                    |
| 19                         | 設計           | find_similar_documents.py | [./tools/find_similar_documents.py](./tools/find_similar_documents.py) | ドキュメント全般の類似度検出                           |
| 20                         | 設計           | tfidf_similar_docs.py     | [./tools/tfidf_similar_docs.py](./tools/tfidf_similar_docs.py)         | 高度な類似度分析（TF-IDF）                             |
| **ワークツリー管理**       |                |                           |                                                                        |                                                        |
| 21                         | 管理           | check_worktree_scopes.sh  | [./tools/check_worktree_scopes.sh](./tools/check_worktree_scopes.sh)   | 全ワークツリーが WORKTREE_SCOPE.md を持つか検査        |
| **その他ユーティリティ**   |                |                           |                                                                        |                                                        |
| 22                         | ユーティリティ | jsonl_to_md.sh            | [./jsonl_to_md.sh](./jsonl_to_md.sh)                                   | JSONL ファイルを Markdown に変換                       |
| 23                         | ユーティリティ | extract_deps_from_svg.sh  | [./extract_deps_from_svg.sh](./extract_deps_from_svg.sh)               | Graphviz SVG から依存関係抽出                          |
| **HLO 分析**               |                |                           |                                                                        |                                                        |
| 24                         | HLO            | summarize_hlo_jsonl.py    | [./hlo/summarize_hlo_jsonl.py](./hlo/summarize_hlo_jsonl.py)           | HLO JSONL ログの集計・分析                             |

**合計:** 24個のツール・スクリプト

______________________________________________________________________

## 🚀 クイックスタート

### 新規ブランチ開始（最初の1回）

```bash
# 1. 規約確認（推奨）
bash scripts/view_conventions.sh naming

# 2. ワークツリー作成
bash scripts/setup_worktree.sh work/my-feature-YYYYMMDD

# 3. ワークツリーに移動
cd .worktrees/work-my-feature-YYYYMMDD
vim WORKTREE_SCOPE.md
git add WORKTREE_SCOPE.md
git commit -m "chore(worktree): init scope"
git push -u origin work/my-feature-YYYYMMDD
```

## テスト＆静的解析実行

> **詳細は [`documents/tools/README.md` — CI スクリプト実行ガイド](../documents/tools/README.md#%F0%9F%94%84-ci-%E3%82%B9%E3%82%AF%E3%83%AA%E3%83%97%E3%83%88%E5%AE%9F%E8%A1%8C%E3%82%AC%E3%82%A4%E3%83%89scriptsci) を参照**

```bash
# ★推奨: 統合テスト（pytest + pyright + ruff）
bash scripts/ci/run_all_checks.sh

# またはシンプル版（pytest のみ＋ログ保存）
bash scripts/run_pytest_with_logs.sh
```

## ドキュメント整形

```bash
# Markdown 正規化
python scripts/tools/format_markdown.py

# リンク監査・修正
python scripts/tools/audit_and_fix_links.py
```

## 設計ファイル整理

```bash
# 重複ファイル検出
python scripts/tools/find_redundant_designs.py

# 類似度分析
python scripts/tools/tfidf_similar_docs.py
```

______________________________________________________________________

## 📖 詳細ドキュメント

### 全体学習・オンボーディング

| 対象           | ドキュメント                                                                           | 概要                         |
| -------------- | -------------------------------------------------------------------------------------- | ---------------------------- |
| **新規参画者** | [../documents/tools/README.md](../documents/tools/README.md)                           | ツール整理ハブ・推奨読み順   |
| **全体理解**   | [../documents/tools/TOOLS_DIRECTORY.md](../documents/tools/TOOLS_DIRECTORY.md)         | 全ツール詳細目録 (470行以上) |
| **作業別手順** | [../documents/FILE_CHECKLIST_OPERATIONS.md](../documents/FILE_CHECKLIST_OPERATIONS.md) | 8つのチェックリスト＋手順    |

### カテゴリ別詳細ドキュメント

| カテゴリ             | ドキュメント                                                                                       | 概要                                             |
| -------------------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **CI・テスト**       | [./ci/README.md](./ci/README.md)                                                                   | ローカル CI 実行ガイド                           |
| **ツール詳細**       | [./tools/README.md](./tools/README.md)                                                             | ツール配下のスクリプト説明                       |
| **ワークツリー統一** | [../documents/tools/WORKTREE_TOOL_UNIFICATION.md](../documents/tools/WORKTREE_TOOL_UNIFICATION.md) | worktree 作成入口の整理メモ |

### 規約・運用方針

| ドキュメント                                                                             | 概要                                 |
| ---------------------------------------------------------------------------------------- | ------------------------------------ |
| [../documents/coding-conventions-project.md](../documents/coding-conventions-project.md) | プロジェクト全体の運用規約           |
| [../documents/worktree-lifecycle.md](../documents/worktree-lifecycle.md)                 | ワークツリー管理規約                 |
| [../documents/WORKFLOW_INVENTORY.md](../documents/WORKFLOW_INVENTORY.md)                 | 自動化ワークフロー現状・未自動化項目 |

______________________________________________________________________

## 🛠 カテゴリ別操作ガイド

### 初期化関連（リポジトリ初期化時のみ）

**対象:** [git_config.sh](./git_config.sh), [git_init.sh](./git_init.sh), [git_repo_init.sh](./git_repo_init.sh), [create_toml.sh](./create_toml.sh)

**用途:**

```bash
# Git 初期設定（初回のみ）
bash scripts/git_init.sh

# 新規パッケージ作成時
bash scripts/git_repo_init.sh
# → python/<package-name>/ 作成

# pyproject.toml ひな形
bash scripts/create_toml.sh <package-name>
```

**詳細:** [../documents/FILE_CHECKLIST_OPERATIONS.md#チェックリスト2](../documents/FILE_CHECKLIST_OPERATIONS.md)

______________________________________________________________________

## ワークツリー関連（開発フロー中心）

**推奨:**

```bash
# ★ このコマンドを使用してください
bash scripts/setup_worktree.sh work/my-feature-YYYYMMDD

# 旧入口も同じ処理へ委譲されます
bash scripts/tools/create_worktree.sh work/my-feature-YYYYMMDD
```

**補助:**

```bash
# 作業ガイド＆ワークツリー状況確認
bash scripts/guide.sh

# 全ワークツリー検査
bash scripts/tools/check_worktree_scopes.sh
```

**詳細:**

- [../documents/FILE_CHECKLIST_OPERATIONS.md#チェックリスト1](../documents/FILE_CHECKLIST_OPERATIONS.md) — 新規ブランチ開始
- [../documents/FILE_CHECKLIST_OPERATIONS.md#チェックリスト6](../documents/FILE_CHECKLIST_OPERATIONS.md) — ワークツリー完了＆クリーンアップ

______________________________________________________________________

## テスト・CI 関連（品質保証）

**推奨フロー:**

```bash
# 1. ★統合テスト（pytest + pyright + ruff）
bash scripts/ci/run_all_checks.sh

# 2. 高速モード（ruff skip）
bash scripts/ci/run_all_checks.sh --quick

# 3. シンプル版（pytest のみ）
bash scripts/run_pytest_with_logs.sh
```yaml

**詳細:**

- [./ci/README.md](./ci/README.md) — CI スクリプトガイド
- [../documents/FILE_CHECKLIST_OPERATIONS.md#チェックリスト3](../documents/FILE_CHECKLIST_OPERATIONS.md) — コード実装＆テスト

______________________________________________________________________

## ドキュメント処理（品質改善）

**推奨フロー:**

```bash
# 1. Markdown 正規化（改行・空白・EOF 統一）
python scripts/tools/format_markdown.py

# 2. リンク監査＆修正（相対パス統一）
python scripts/tools/audit_and_fix_links.py

# 3. ドキュメント品質修正
python scripts/tools/fix_markdown_docs.py

# 4. 変更をコミット
git add documents/
git commit -m "docs: apply formatting and link fixes"
```yaml

**詳細:** [../documents/FILE_CHECKLIST_OPERATIONS.md#チェックリスト4](../documents/FILE_CHECKLIST_OPERATIONS.md)

______________________________________________________________________

## 設計ファイル管理（整理・統合）

**推奨フロー:**

```bash
# 1. 重複検出（完全一致）
python scripts/tools/find_redundant_designs.py

# 2. 類似度検出（内容）
python scripts/tools/find_similar_designs.py
python scripts/tools/tfidf_similar_docs.py

# 3. 整理実行（dry-run で確認）
python scripts/tools/organize_designs.py --dry-run

# 4. 実際に実行
python scripts/tools/organize_designs.py

# 5. テンプレート作成（新規モジュール時）
python scripts/tools/create_design_template.py python/jax_util/<module>/
```yaml

**詳細:** [../documents/FILE_CHECKLIST_OPERATIONS.md#チェックリスト5](../documents/FILE_CHECKLIST_OPERATIONS.md)

______________________________________________________________________

## 🔗 GitHub Actions との連携

### CI ワークフロー

`.github/workflows/ci.yml` は `run_all_checks.sh` と同等の主要チェックを直接実行します：

```yaml
- name: Run pytest with coverage
  run: python -m pytest ...
````

`bash scripts/ci/run_all_checks.sh` を通せば、GitHub Actions の主要失敗条件とも揃います。

詳細: [../.github/workflows/ci.yml](../.github/workflows/ci.yml)

### エージェント調整ワークフロー

`.github/workflows/agent-coordination.yml` — 自動化エージェント間の協調

詳細: [../.github/AGENTS.md](../.github/AGENTS.md)

______________________________________________________________________

## ⚙️ Makefile ターゲット

`make` でよく使用するターゲット：

````bash
# Git 初期化
make git_init

# テスト実行（ci/run_all_checks.sh 内部から呼び出し可能）
make test          # 未実装（scripts/ci/run_all_checks.sh で代替）
```yaml

詳細: [../Makefile](../Makefile)

______________________________________________________________________

## 📊 ツール使用フロー図

```text
開発開始
  ├→ git_init.sh          （初回のみ）
  │
  ├→ setup_worktree.sh
  │  └→ create_worktree.sh は互換ラッパー
  │  └→ WORKTREE_SCOPE.md 編集・コミット
  │
  ├→ コード編集
  │
  ├→ ci/run_all_checks.sh （★推奨）
  │  ├→ pytest python/tests/
  │  ├→ pyright python/
  │  └→ ruff check python/
  │
  │  または：run_pytest_with_logs.sh
  │
  ├→ ドキュメント更新
  │  ├→ format_markdown.py
  │  ├→ audit_and_fix_links.py
  │  └→ fix_markdown_docs.py
  │
  └→ git commit & push
     └→ GitHub Actions CI 実行

設計整理（必要時）
  ├→ find_redundant_designs.py
  ├→ tfidf_similar_docs.py
  └→ organize_designs.py

ワークツリー完了
  ├→ carry-over ファイル保存
  └→ git worktree remove
```text

______________________________________________________________________

## 🆘 トラブルシューティング

### ci/run_all_checks.sh が失敗

→ [./ci/README.md#トラブルシューティング](./ci/README.md)

### ワークツリー作成が失敗

→ [../documents/FILE_CHECKLIST_OPERATIONS.md#トラブルシューティング](../documents/FILE_CHECKLIST_OPERATIONS.md)

### pytest 失敗

```bash
# 失敗したテスト単体を再実行
pytest python/tests/test_module.py::TestClass::test_method -v -s
```text

## pyright エラー多い場合

```bash
# 型チェックの詳細表示
pyright python/ --verbose
```text

## ruff 自動修正

```bash
# エラー自動修正
ruff check --fix python/

# 修正内容確認
git diff python/
```text

## パッケージ不足

```bash
# 環境チェック
python -c "import pytest; import pyright; import ruff; print('OK')"

# インストール
pip install -r docker/requirements.txt
```text

______________________________________________________________________

## 🔐 配置規則・設計方針

### ディレクトリ構成

```text
scripts/
├── README.md                    ← このファイル（統合窓口）
├── ci/                          ← CI スクリプト集約（推奨実行）
│   ├── README.md
│   └── run_all_checks.sh        ← ★ほぼすべての開発で実行
├── tools/                       ← 高度なツール類
│   ├── worktree/                ← ワークツリー関連（将来）
│   ├── docs/                    ← ドキュメント処理
│   ├── design/                  ← 設計ファイル管理
│   └── analysis/                ← 分析・レポート（将来）
├── hlo/                         ← HLO 分析
├── setup/                       ← 初期化（将来 Phase 2）
├── dev/                         ← 開発補助（将来 Phase 2）
└── guide/                       ← ガイド・表示（将来 Phase 2）
```yaml

**現状:** Phase 1（tools/ 集約）
**今後:** Phase 2 で細分化（setup/, dev/, ci/, guide/ へ段階移行）

### 相対パス規則

すべてのドキュメント内参照は **相対パス** で記述：

```markdown
# scripts/README.md から ../documents/... への参照
[TOOLS_DIRECTORY.md](../documents/tools/TOOLS_DIRECTORY.md)

# ./ci/README.md から ../../documents/... への参照
[FILE_CHECKLIST_OPERATIONS.md](../../documents/FILE_CHECKLIST_OPERATIONS.md)
```yaml

______________________________________________________________________

## 🔄 スクリプト更新・整備予定（Phase 2）

現在 Phase 1（tools/ に集約）。今後以下への移行を検討：

- [ ] `scripts/setup/` — 初期化スクリプト集約
- [ ] `scripts/dev/` — 開発補助スクリプト
- [ ] `scripts/ci/` — CI スクリプト（既に작成）
- [ ] `scripts/guide/` — 情報表示スクリプト

______________________________________________________________________

## 🤝 貢献・改善

### 新規ツール追加

1. 実装: このディレクトリに追加
1. 軽文書: `## カテゴリ別操作ガイド` に 1 行追加
1. 正式化: [../documents/WORKFLOW_INVENTORY.md](../documents/WORKFLOW_INVENTORY.md) に登録

### ドキュメント修正・改善

- GitHub Issues で報告
- [../documents/tools/README.md](../documents/tools/README.md) から導線の改善

______________________________________________________________________

## 📞 サポート

### ツール使用方法

- このファイル（scripts/README.md）の該当セクション参照
- 各ツール先頭のコメント (`# 用途:` など）参照
- `--help` オプション試行（対応ツール）

### 規約・運用ルール

- [../documents/README.md](../documents/README.md) — ドキュメント体系
- [../documents/coding-conventions-project.md](../documents/coding-conventions-project.md) — プロジェクト運用規約

### 動作トラブル

- [../documents/tools/README.md#トラブルシューティング](../documents/tools/README.md#トラブルシューティング) — CI・ツール関連
- [../documents/FILE_CHECKLIST_OPERATIONS.md#トラブルシューティング](../documents/FILE_CHECKLIST_OPERATIONS.md) — 作業関連

______________________________________________________________________

**最終更新:** 2026-03-19\
**管理:** GitHub Copilot & Development Team
````
