# ツール・スクリプト目録

この文書は、プロジェクトのツール・スクリプト体系を整理し、
用途・使用法・依存関係を明確にする公式目録です。

**作成日:** 2026-03-19\
**現在の状態:** 整理中（散在状態から構造化中）

## 概要

現在、以下のツール・スクリプトが `scripts/` 配下に散在しています：

| ディレクトリ     | ファイル数 | 用途                                                        |
| ---------------- | ---------- | ----------------------------------------------------------- |
| `scripts/` 直下  | 8個        | Git初期化、コンベンション表示、ワークツリー管理、テスト実行 |
| `scripts/tools/` | 11個       | ワークツリー管理、ドキュメント処理、設計ファイル整理        |
| `scripts/hlo/`   | 1個        | HLO分析・可視化                                             |
| **合計**         | **20個**   | —                                                           |

**問題点:**

- 重複・役割混在（`setup_worktree.sh` / `create_worktree.sh`）
- ドキュメント化が不十分
- 前提・依存関係が不明確
- 実行順序・チェックリストが不足

## スクリプト分類

### 1. Git・ブランチ管理（scripts/直下）

#### 1.1 `git_config.sh`

- **用途:** Git ユーザー名・メールアドレス・デフォルトブランチ設定
- **実行:** `bash scripts/git_config.sh`
- **前提:** なし
- **依存:** Git
- **出力:** コンソール（設定のみ）
- **用途:** 初期セットアップ時の1回実行

#### 1.2 `git_init.sh`

- **用途:** Git初期化（`git_config.sh` を呼び出し）
- **実行:** `bash scripts/git_init.sh`
- **実行:** `make git_init`（Makefileラッパー）
- **前提:** Git リポジトリが未初期化
- **依存:** `git_config.sh`
- **出力:** コンソール

#### 1.3 `git_repo_init.sh`

- **用途:** 新しい Python パッケージディレクトリを作成
- **実行:** `bash scripts/git_repo_init.sh`
- **前提:** リポジトリが初期化済み
- **依存:** なし
- **入力:** インタラクティブプロンプト（パッケージ名）
- **出力:** `python/<package-name>/` ディレクトリ、`__init__.py` ファイル

### 2. ドキュメント・コンベンション表示（scripts/直下）

#### 2.1 `view_conventions.sh`

- **用途:** プロジェクト規約ファイルを整理して表示
- **実行:** `bash scripts/view_conventions.sh [検索キーワード]`
- **前提:** `documents/conventions/` が存在
- **依存:** なし
- **出力:** Markdown形式でコンソール表示
- **例:** `bash scripts/view_conventions.sh naming`

#### 2.2 `read_conventions.sh`

- **用途:** 規約ファイル一覧をフォーマットして表示
- **実行:** `bash scripts/read_conventions.sh`
- **前提:** なし
- **依存:** なし
- **出力:** テーブル形式でコンソール表示

### 3. ワークツリー・ブランチ管理

#### 3.1 `scripts/setup_worktree.sh` ⚠️ 要統一

- **用途:** ワークツリー・ブランチをセットアップして規約に従う
- **実行:** `bash scripts/setup_worktree.sh <branch-name> [説明]`
- **前提:** リポジトリが `main` ブランチで clean
- **依存:** `git_config.sh`（スコープテンプレート読み込み）
- **入力:** ブランチ名、オプション説明文
- **出力:** ワークツリー作成、`WORKTREE_SCOPE.md` 自動生成
- **例:**
  ```bash
  bash scripts/setup_worktree.sh protocol-improvements "Protocol改善"
  ```

#### 3.2 `scripts/tools/create_worktree.sh` ⚠️ 要統一

- **用途:** `setup_worktree.sh` と同様（ブランチ・ワークツリー作成）
- **実行:** `bash scripts/tools/create_worktree.sh <branch-name> [WT_PATH]`
- **前提:** リポジトリが clean
- **依存:** なし（独立）
- **入力:** ブランチ名、オプションワークツリーパス
- **出力:** ワークツリー作成、スコープテンプレート自動コピー
- **差異:** 相対パスかワークツリー場所の指定可（`setup_worktree.sh` より柔軟）

**推奨:**

- 実装が重複している（両方ともワークツリー作成・スコープ自動配置）
- **統一方針:** `scripts/tools/create_worktree.sh` を標準にし、`scripts/setup_worktree.sh` は廃止予定ラッパーとするか、
  またはドキュメントで「`scripts/tools/create_worktree.sh` を使用」と統一

#### 3.3 `scripts/guide.sh`

- **用途:** 作業フロー全体ガイド（説明用スクリプト）
- **実行:** `bash scripts/guide.sh`
- **出力:** テキストガイド＋現在のブランチ・ワークツリー一覧表示
- **用途:** 新規ユーザーへのオンボーディング

#### 3.4 `scripts/tools/check_worktree_scopes.sh`

- **用途:** 全ワークツリーが `WORKTREE_SCOPE.md` を持っているか検査
- **実行:** `bash scripts/tools/check_worktree_scopes.sh`
- **出力:** `reports/worktree_scope_report.txt` に検査結果
- **用途:** 規約遵守チェック、CI で定期実行推奨

### 4. テスト実行ログ管理（scripts/直下）

#### 4.1 `run_pytest_with_logs.sh`

- **用途:** `pytest` 実行のログを `python/tests/logs/` に時系列保存
- **実行:** `bash scripts/run_pytest_with_logs.sh`
- **前提:** `pytest` インストール済み
- **依存:** Python, pytest
- **入力:** テストが `python/tests/` 配下
- **出力:** ログディレクトリ `python/tests/logs/[YYYYMMDD]-[HHMMSS]/`
  - `pytest.raw.txt`: 生テキストログ
  - `pytest.jsonl`: JSON形式ログ（フィルタ済み）
  - `exit_code.txt`: 終了コード
- **用途:** テスト結果のトレーサビリティ

### 5. ドキュメント処理・整理（scripts/tools/）

#### 5.1 `format_markdown.py`

- **用途:** Markdown ファイルの正規化
  - 改行の LF 設定、末尾空白削除、連続改行圧縮、EOF改行統一
- **実行:**
  ```bash
  python scripts/tools/format_markdown.py [ファイル1] [ファイル2] ...
  python scripts/tools/format_markdown.py  # デフォルト: README.md, documents/, notes/, reviews/
  ```
- **依存:** Python 3.6+
- **出力:** ファイル直接修正
- **用途:** ドキュメント整形・統一

#### 5.2 `fix_markdown_docs.py`

- **用途:** Markdown ドキュメント内のリンク・記述の修正
- **実行:** `python scripts/tools/fix_markdown_docs.py`
- **依存:** Python, `pathlib`
- **出力:** `reports/markdown_fixes_report.txt`, 修正したファイル
- **用途:** ドキュメント品質改善

#### 5.3 `audit_and_fix_links.py`

- **用途:** Markdown 内の相対リンク・ファイル参照を検査・修正
- **実行:** `python scripts/tools/audit_and_fix_links.py`
- **依存:** Python
- **出力:** `reports/link_audit.txt`, 修正内容
- **用途:** リンク切れ・絶対パス混入の防止

### 6. 設計ファイル管理（scripts/tools/）

#### 6.1 `organize_designs.py`

- **用途:** 設計ファイル `documents/design/` をサブモジュール別ディレクトリにコピー
- **実行:** `python scripts/tools/organize_designs.py [--dry-run]`
- **前提:** `documents/design/` / `python/jax_util/*/design/` が存在
- **依存:** Python
- **出力:**
  - 実行: 設計ファイルをコピー
  - `--dry-run`: レポート出力のみ
- **用途:** 設計ドキュメントの一元配置

#### 6.2 `create_design_template.py`

- **用途:** サブモジュール用設計テンプレートを生成
- **実行:** `python scripts/tools/create_design_template.py <module-path>`
- **依存:** Python
- **出力:** `<module-path>/design/template.md`
- **用途:** 新規サブモジュール設計ファイルのひな形

#### 6.3 `find_redundant_designs.py`

- **用途:** 完全一致する重複設計ファイルを検出・削除
- **実行:** `python scripts/tools/find_redundant_designs.py [--delete]`
- **依存:** Python
- **出力:**
  - 通常: 重複候補をレポート
  - `--delete`: 実際に削除
- **用途:** 設計ファイル整理・重複排除

#### 6.4 `find_similar_designs.py`

- **用途:** 内容が似ている設計ファイルを検出（完全一致以外）
- **実行:** `python scripts/tools/find_similar_designs.py [--threshold 0.85]`
- **依存:** Python（TF-IDF）
- **出力:** 類似度スコア付きレポート
- **用途:** マージ候補の検出

#### 6.5 `find_similar_documents.py`

- **用途:** ドキュメント全般の類似度検出（設計以外も対象）
- **実行:** `python scripts/tools/find_similar_documents.py`
- **依存:** Python
- **出力:** 類似ドキュメントペアのレポート

#### 6.6 `tfidf_similar_docs.py`

- **用途:** TF-IDF による高度な類似度検出
- **実行:** `python scripts/tools/tfidf_similar_docs.py [--output report.txt]`
- **依存:** Python（scikit-learn）
- **出力:** 詳細な類似度スコア付きレポート
- **用途:** 詳細分析・重複検出

### 7. その他ユーティリティ（scripts/直下）

#### 7.1 `create_toml.sh`

- **用途:** Python パッケージ用 `pyproject.toml` テンプレート作成
- **実行:** `bash scripts/create_toml.sh <package-name>`
- **入力:** パッケージ名
- **出力:** `pyproject.toml` ファイル
- **用途:** 新規パッケージセットアップの補助

#### 7.2 `jsonl_to_md.sh`

- **用途:** JSONL (JSON Lines) ファイルを Markdown に変換
- **実行:** `bash scripts/jsonl_to_md.sh <input.jsonl> <output.md>`
- **入力:** JSONL ファイル
- **出力:** Markdown ファイル
- **用途:** JSON ログの可視化

#### 7.3 `extract_deps_from_svg.sh`

- **用途:** Graphviz の SVG（deps.dot）から依存関係を抽出
- **実行:**
  ```bash
  bash scripts/extract_deps_from_svg.sh deps.dot
  bash scripts/extract_deps_from_svg.sh deps.dot --internal
  bash scripts/extract_deps_from_svg.sh deps.dot --internal --prefix jax_util
  ```
- **入力:** SVG/DOT ファイル
- **出力:** コンソール（抽出結果）
- **用途:** 依存関係可視化・分析

### 8. HLO 分析（scripts/hlo/）

#### 8.1 `summarize_hlo_jsonl.py`

- **用途:** HLO JSONL ログを集計・分析
- **実行:** `python scripts/hlo/summarize_hlo_jsonl.py <input.jsonl> [--top 50]`
- **前提:** `jax_util.hlo.dump_hlo_jsonl` で作成した JSONL
- **依存:** Python, jax_util
- **出力:** 統計情報コンソール出力
  - レコード数、タグ出現回数、方言、HLO op 頻度
- **用途:** HLO 処理の最適化分析

______________________________________________________________________

## ツール依存関係グラフ

```
main ブランチ
  ↓
git_init.sh 
  ├→ git_config.sh
  └→ [プロジェクト初期化]
      ↓
      ├─ setup_worktree.sh OR create_worktree.sh
      │   ├→ WORKTREE_SCOPE_TEMPLATE.md テンプレート読込
      │   └→ ワークツリー作成
      │
      ├─ view_conventions.sh / read_conventions.sh
      │   └→ 規約確認
      │
      ├─ run_pytest_with_logs.sh
      │   └→ テスト実行・ログ保存
      │
      └─ [ドキュメント処理]
          ├→ format_markdown.py
          ├→ fix_markdown_docs.py
          ├→ audit_and_fix_links.py
          └→ 設計ファイル管理
              ├→ organize_designs.py
              ├→ create_design_template.py
              ├→ find_redundant_designs.py
              ├→ find_similar_designs.py
              └→ tfidf_similar_docs.py
```

______________________________________________________________________

## 使用フロー別ガイド

### フロー1: 新規開発ブランチ作成＆開始

```bash
# 1. main ブランチに移動（既に main にいる場合はスキップ）
cd /workspace

# 2. 規約確認（推奨）
bash scripts/view_conventions.sh

# 3. ワークツリー・ブランチ作成
bash scripts/tools/create_worktree.sh my-feature "新機能実装"

# 4. ワークツリーに移動
cd .worktrees/my-feature

# 5. WORKTREE_SCOPE.md を編集（重要）
vim WORKTREE_SCOPE.md

# 6. 作業開始
```

### フロー2: テスト実行＆ログ保存

```bash
# テストをログ付きで実行
bash scripts/run_pytest_with_logs.sh

# ログ確認
ls python/tests/logs/
cat python/tests/logs/[latest]/pytest.raw.txt
```

### フロー3: ドキュメント整形・リンク修正

```bash
# 整形
python scripts/tools/format_markdown.py

# リンク監査
python scripts/tools/audit_and_fix_links.py

# ドキュメント修正
python scripts/tools/fix_markdown_docs.py
```

### フロー4: 設計ファイル整理

```bash
# dry-run で効果確認
python scripts/tools/organize_designs.py --dry-run

# 実際に実行
python scripts/tools/organize_designs.py

# 類似ファイル検出
python scripts/tools/find_similar_designs.py
python scripts/tools/tfidf_similar_docs.py

# 重複削除
python scripts/tools/find_redundant_designs.py --delete
```

### フロー5: ワークツリー規約遵守チェック

```bash
# スコープファイル検査
bash scripts/tools/check_worktree_scopes.sh

# レポート確認
cat reports/worktree_scope_report.txt
```

______________________________________________________________________

## 整理方針・推奨変更

### 優先度1: 重複解決

| ファイル                                              | 問題                                      | 推奨                                                            |
| ----------------------------------------------------- | ----------------------------------------- | --------------------------------------------------------------- |
| `scripts/setup_worktree.sh`                           | `scripts/tools/create_worktree.sh` と重複 | 統一：`create_worktree.sh` を標準化、もしくはシンボリックリンク |
| `scripts/view_conventions.sh` + `read_conventions.sh` | 役割が曖昧                                | 用途を明確化、統一推奨                                          |

### 優先度2: ドキュメント充実

- 各ツール用の個別 `README.md` を `scripts/tools/README.md` に集約
- フロー別チェックリスト作成
- CI 統合文書作成

### 優先度3: 拡張計画

- `scripts/ci/` ディレクトリ新設（静的解析・自動修正スクリプト）
- GitHub Actions ワークフロー統合
- 定期実行タスク（cron 対応）

______________________________________________________________________

## 次ステップ

1. **統一決定:** `setup_worktree.sh` vs `create_worktree.sh` をどちらに統一するか決定
1. **個別ドキュメント:** 各ツール用 README を作成
1. **チェックリスト充実:** 作業フロー別チェックリスト拡充（別ファイル）
1. **CI 統合:** GitHub Actions での定期実行設定
1. **廃止計画:** 不要ツール・スクリプトの削除予定表作成
