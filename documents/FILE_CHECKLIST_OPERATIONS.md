# 作業別チェックリスト・実行手順書

この文書は、プロジェクト内での典型的な作業フローに対応したチェックリスト・実行手順をまとめます。

**対象読者:** 開発者・メンテナー  
**更新日:** 2026-03-19

---

## チェックリスト1: 新規開発ブランチ開始

新しい機能開発・バグ修正を始める場合の統一手順です。

### 前提条件
- [ ] リポジトリが `main` ブランチである
- [ ] 作業ツリーが clean 状態（`git status` で何も表示されない）
- [ ] `origin/main` に接続可能

### セットアップ手順

#### ステップ1: 規約確認（5分以内）
```bash
# 1-1. プロジェクト全体ガイド表示
bash scripts/guide.sh

# 1-2. 具体的な規約検索（例：命名規則）
bash scripts/view_conventions.sh naming

# 1-3. 重要なドキュメント確認
cat documents/README.md
cat documents/coding-conventions-project.md
```

**確認項目:**
- [ ] ブランチ運用方針 (WORKTREE_SCOPE.md の目的理解)
- [ ] 実装対象ディレクトリの責務確認
- [ ] テスト実行方法確認
- [ ] コミット規則確認（feat: / fix: / docs: など）

#### ステップ2: ワークツリー作成（3分）
```bash
# 2-1. 推奨: create_worktree.sh を使用
bash scripts/tools/create_worktree.sh my-feature-name

# OR 従来方式: setup_worktree.sh を使用
# bash scripts/setup_worktree.sh my-feature-name "機能説明"

# 2-2. ワークツリー移動
cd .worktrees/my-feature-name
```

**確認項目:**
- [ ] ワークツリーディレクトリが作成された
- [ ] `WORKTREE_SCOPE.md` が存在
- [ ] `git branch -v` で新ブランチが表示される

#### ステップ3: スコープ設定（5分）
```bash
# 3-1. WORKTREE_SCOPE.md を編集
vim WORKTREE_SCOPE.md

# 3-2. 主要項目を確認・編集
#   - Branch: correct?
#   - Purpose: 今回の作業目的
#   - Editable Directories: 編集対象ディレクトリ
#   - Required Checks: 実行必須テスト
#   - Special Rules: 特別な指定事項
```

**確認項目:**
- [ ] `Branch` が正しい（work/my-feature-name-YYYYMMDD）
- [ ] `Purpose` に日本語で作業内容記述
- [ ] `Editable Directories` が具体的
- [ ] `Required Checks` にテスト・チェック項目を記載

#### ステップ4: 最初のコミット（3分）
```bash
# 4-1. スコープ設定をコミット
git add WORKTREE_SCOPE.md
git commit -m "chore(worktree): initialize scope"
git push -u origin work/my-feature-name-YYYYMMDD

# 4-2. 確認
git log -n2 --oneline
git branch -vv
```

**確認項目:**
- [ ] 初回コミット作成
- [ ] origin に push 成功
- [ ] ブランチが origin を tracking している

### 完了チェック
- [ ] 新ワークツリーが `.worktrees/` に存在
- [ ] `WORKTREE_SCOPE.md` が記入済み
- [ ] 初回コミットが origin に push 完了
- [ ] `git worktree list` に表示される

**想定所要時間:** 15分

---

## チェックリスト2: Python パッケージ作成・追加

`python/` 配下に新規サブパッケージを追加する場合。

### 前提条件
- [ ] メインワークツリーで `main` ブランチ
- [ ] `python/` ディレクトリが存在
- [ ] 別ワークツリーで作業する場合はそれも clean

### セットアップ手順

#### ステップ1: 新規パッケージディレクトリ作成
```bash
# 1-1. スクリプト実行 OR 手動作成
bash scripts/git_repo_init.sh
# → python/<package-name>/ が作成される

# OR 手動で作成
mkdir -p python/jax_util/<my-module>
touch python/jax_util/<my-module>/__init__.py
```

**確認項目:**
- [ ] `python/jax_util/<my-module>/` ディレクトリ存在
- [ ] `__init__.py` ファイル存在

#### ステップ2: pyproject.toml 確認
```bash
# 2-1. トップレベルの pyproject.toml 確認
cat pyproject.toml

# 2-2. 新規パッケージが [tool.pyright] include に追加されているか確認
grep -A 5 "tool.pyright" pyproject.toml
```

**確認項目:**
- [ ] インクルードパスに「python/jax_util/<my-module>」が含まれている
- [ ] 分析対象に含まれている

#### ステップ3: pyright・pytest 実行
```bash
# 3-1. 型チェック
pyright python/jax_util/<my-module>/

# 3-2. テスト実行（後述のチェックリスト3 参照）
pytest python/tests/<my-module>/
```

**確認項目:**
- [ ] pyright でエラーなし
- [ ] pytest で既存テストパス

### 完了チェック
- [ ] ディレクトリ・`__init__.py` 作成完了
- [ ] pyproject.toml に登録
- [ ] 型チェック・テスト実行可能

**想定所要時間:** 5分

---

## チェックリスト3: コード実装＆テスト

実装作業完了後のテスト・品質確保手順。

### 前提条件
- [ ] コード編集完了
- [ ] ワークツリー内で作業中（もしくはメインワークツリー）
- [ ] `python/tests/` にテストファイルが存在

### 実行手順

#### ステップ1: 静的解析（5分）

```bash
# 1-1. pyright 型チェック
pyright python/

# 1-2. ruff リント・スタイルチェック
ruff check python/

# 1-3. 自動フォーマット（オプション）
black python/
```

**確認項目:**
- [ ] pyright エラー数 = 0
- [ ] ruff エラー数 = 0
- [ ] black フォーマット可能

#### ステップ2: テスト実行（10分～）

```bash
# 2-1. 全テスト実行（ログ保存）
bash scripts/run_pytest_with_logs.sh

# 2-2. ログ確認
ls -lt python/tests/logs/ | head -5
tail -50 python/tests/logs/[latest]/pytest.raw.txt

# 2-3. 特定テストのみ実行（デバッグ時）
pytest python/tests/<module>/test_<name>.py -v
pytest python/tests/<module>/test_<name>.py::TestClass::test_method -v -s
```

**確認項目:**
- [ ] テスト成功数・失敗数確認
- [ ] 期待値との合致確認
- [ ] ログファイル保存確認

#### ステップ3: 修正対応（必要時）

```bash
# 3-1. テスト失敗時は原因を特定
# a. pyright エラー → 型アノテーション修正
# b. 論理エラー → 実装修正

# 3-2. ruff エラー修正
ruff check --fix python/

# 3-3. 再実行
bash scripts/run_pytest_with_logs.sh
```

**確認項目:**
- [ ] 前回失敗していたテストがパス
- [ ] 新規エラー発生していない

### 完了チェック
- [ ] pyright: 0 エラー
- [ ] ruff: 0 エラー
- [ ] pytest: 全テストパス
- [ ] ログ出力確認

**想定所要時間:** 15～30分

---

## チェックリスト4: ドキュメント更新

実装に伴うドキュメント更新手順。

### 前提条件
- [ ] 実装変更が完了
- [ ] テストが全パス
- [ ] `documents/` 配下に対応ファイルが存在

### 実行手順

#### ステップ1: 対象ドキュメント特定（3分）

実装変更の種類に応じた対象ドキュメント：

| 変更内容 | 対象ドキュメント |
| --- | --- |
| 新規 API 追加 | `documents/design/apis/<module>.md` |
| 既存 Protocol 変更 | `documents/design/base_components.md` |
| 型定義追加 | `documents/design/base_components.md` |
| ユーティリティ追加 | 対応モジュール説明ドキュメント |
| テスト枠組み変更 | `documents/coding-conventions-testing.md` |
| 運用ルール変更 | `documents/coding-conventions-project.md` |

```bash
# 1-1. 該当ドキュメント確認
ls documents/design/
ls documents/
```

**確認項目:**
- [ ] 対応ファイル特定

#### ステップ2: ドキュメント更新（10～30分）

```bash
# 2-1. テキストエディタで編集
vim documents/design/apis/<module>.md

# 2-2. コード参照の追加（相対パス）
# 例: [関数 `solve_kkt`](../../python/jax_util/solvers/kkt.py#L42)

# 2-3. 数式・図解の追加（必要に応じて）
# 例: $$ \min_x \frac{1}{2} x^T A x + b^T x $$
```

**確認項目:**
- [ ] 日本語で明確に説明記述
- [ ] コード例・数式を含む
- [ ] リンク参照は相対パス

#### ステップ3: ドキュメント整形（3分）

```bash
# 3-1. Markdown 正規化
python scripts/tools/format_markdown.py documents/

# 3-2. リンク監査
python scripts/tools/audit_and_fix_links.py

# 3-3. ドキュメント品質チェック
python scripts/tools/fix_markdown_docs.py
```

**確認項目:**
- [ ] 改行・余白が正規化
- [ ] リンク切れなし（報告書確認）
- [ ] Markdown 記法エラーなし

#### ステップ4: コミット（2分）

```bash
# 4-1. ドキュメント変更をコミット
git add documents/
git commit -m "docs: update API documentation for <module>"

# 4-2. push
git push origin work/<branch>
```

**確認項目:**
- [ ] コミット作成
- [ ] origin に push

### 完了チェック
- [ ] 対象ドキュメント全て更新
- [ ] 整形・リンク監査清掃
- [ ] コミット・push 完了

**想定所要時間:** 20～50分

---

## チェックリスト5: 設計ドキュメント整理

設計ファイルが `documents/design/` に多数ある場合の整理・統合。

### 前提条件
- [ ] `documents/design/` にファイル複数存在
- [ ] サブモジュール別整理が必要
- [ ] メインワークツリーで作業

### 実行手順

#### ステップ1: 類似度検出（5分）

```bash
# 1-1. 完全一致する重複ファイル検出
python scripts/tools/find_redundant_designs.py

# 1-2. 内容類似のファイル検出
python scripts/tools/find_similar_designs.py

# 1-3. ドキュメント全般の類似度検出（詳細分析）
python scripts/tools/tfidf_similar_docs.py --output reports/similar_analysis.txt
```

**確認項目:**
- [ ] 重複候補レポート生成
- [ ] 類似度スコア確認
- [ ] `reports/` にレポート保存

#### ステップ2: 設計ファイル整理（10～30分）

```bash
# 2-1. dry-run で確認
python scripts/tools/organize_designs.py --dry-run

# 2-2. レポート確認
cat reports/organize_designs_dryrun.txt

# 2-3. OK なら実行
python scripts/tools/organize_designs.py
```

**確認項目:**
- [ ] dry-run結果確認
- [ ] 移動・コピー内容妥当
- [ ] サブモジュール別コピー完了

#### ステップ3: テンプレート作成（必要時、3分）

```bash
# 3-1. サブモジュール用テンプレート作成
python scripts/tools/create_design_template.py python/jax_util/solvers/

# 3-2. 確認
cat python/jax_util/solvers/design/template.md
```

**確認項目:**
- [ ] サブモジュール用デザインテンプレート生成

#### ステップ4: 古いファイル削除（危険操作、慎重に）

```bash
# 4-1. 不要なファイルを確認・削除
# 注意: バックアップ後に実行

# 4-2. 完全一致ファイルを自動削除
python scripts/tools/find_redundant_designs.py --delete
```

**確認項目:**
- [ ] 削除前にバックアップ済み
- [ ] 削除対象確認漏れなし
- [ ] リンク参照がないか二重確認

### 完了チェック
- [ ] 重複ファイル統合
- [ ] サブモジュール別整理完了
- [ ] テンプレート作成完了

**想定所要時間:** 30～60分

---

## チェックリスト6: ワークツリー完了＆クリーンアップ

作業完了後のワークツリー削除・ブランチクローズ手順。

### 前提条件
- [ ] すべてのコード変更が完了・テスト済み
- [ ] すべてのコミットが origin に push 完了
- [ ] PR が merge された OR merge 予定

### 実行手順

#### ステップ1: carry-over 作業（5分）

実験結果・メモをメインブランチに持ち帰る場合：

```bash
# 1-1. メモ・結果を持ち帰る
mkdir -p notes/worktrees/<worktree-name>-<date>
cp -r notes/experiments/* notes/worktrees/<worktree-name>-<date>/

# 1-2. 持ち帰り内容を整理
ls -la notes/worktrees/
```

**確認項目:**
- [ ] 重要なメモ・結果をコピー
- [ ] ワークツリー削除前に完了

#### ステップ2: ワークツリー削除（2分）

```bash
# 2-1. メインワークツリーに戻る
cd /workspace

# 2-2. ワークツリー削除
git worktree remove .worktrees/work-my-feature-name-20260318

# 2-3. ブランチ削除（merge 後のみ）
git branch -d work/my-feature-name-20260318

# 2-4. リモートブランチ削除（merge 後のみ）
git push origin --delete work/my-feature-name-20260318
```

**確認項目:**
- [ ] ワークツリーが削除された
- [ ] `git worktree list` に表示されない
- [ ] ブランチが削除された
- [ ] リモートブランチが削除された

#### ステップ3: 最終確認（1分）

```bash
# 3-1. ワークツリー一覧確認
git worktree list

# 3-2. ブランチ確認
git branch -v

# 3-3. ファイル構造確認
ls -la .worktrees/
```

**確認項目:**
- [ ] 削除対象ワークツリーが表示されない
- [ ] 他のワークツリーは環境維持
- [ ] `.worktrees/` から完全削除

### 完了チェック
- [ ] carry-over ファイル保存
- [ ] ワークツリー削除完了
- [ ] ローカル・リモートブランチ削除完了

**想定所要時間:** 10分

---

## チェックリスト7: ワークツリー規約遵守チェック（管理者向け）

全ワークツリーが規約に従っているか検査する定期タスク。

### 実行手順

#### ステップ1: スコープ検査（3分）

```bash
# 1-1. 全ワークツリー検査
bash scripts/tools/check_worktree_scopes.sh

# 1-2. レポート確認
cat reports/worktree_scope_report.txt
```

**確認項目:**
- [ ] 全ワークツリーが `WORKTREE_SCOPE.md` を保持
- [ ] missing 項目がないか確認

#### ステップ2: 修復対応（必要時、10分～）

WORKTREE_SCOPE.md がない場合：

```bash
# 2-1. テンプレートをコピー
MISSING_WT=".worktrees/work-nn-develop-20260318"
cp documents/WORKTREE_SCOPE_TEMPLATE.md "$MISSING_WT/WORKTREE_SCOPE.md"

# 2-2. 編集
vim "$MISSING_WT/WORKTREE_SCOPE.md"

# 2-3. コミット
cd "$MISSING_WT"
git add WORKTREE_SCOPE.md
git commit -m "chore(worktree): add WORKTREE_SCOPE"
git push -u origin $(git rev-parse --abbrev-ref HEAD)
```

**確認項目:**
- [ ] scopeファイル作成
- [ ] 内容を適切に編集
- [ ] origin に push

### 完了チェック
- [ ] スコープ検査実行
- [ ] 不足ワークツリーが修復

**想定所要時間:** 10～30分（ワークツリー数に応じて）

**推奨実行頻度:** 週1回（金曜日など）

---

## チェックリスト8: ドキュメント品質チェック（管理者サポート向け）

### 実行手順

#### ステップ1: 整形・統一（3分）

```bash
# 1-1. Markdown 整形
python scripts/tools/format_markdown.py

# 1-2. リンク監査・修正
python scripts/tools/audit_and_fix_links.py

# 1-3. ドキュメント修正推奨
python scripts/tools/fix_markdown_docs.py
```

#### ステップ2: 設計ファイル品質（5分）

```bash
# 2-1. 類似度検出
python scripts/tools/tfidf_similar_docs.py

# 2-2. レポート確認
cat reports/similar_analysis.txt

# 2-3. 重複削除提案確認
python scripts/tools/find_redundant_designs.py
```

#### ステップ3: コミット（3分）

```bash
# 修正内容をコミット
git add documents/
git commit -m "docs: apply formatting and link fixes"
git push origin main
```

### 完了チェック
- [ ] 整形・リンク修正完了
- [ ] 設計ファイル類似度分析完了

**想定所要時間:** 10分

**推奨実行頻度:** 週1回（月曜日など）

---

## トラブルシューティング

### 問題: `setup_worktree.sh` と `create_worktree.sh` の違いは？

**現状:** 
- 両方共存（要統一）
- 機能がほぼ重複

**推奨:**
- `scripts/tools/create_worktree.sh` を標準使用
- `scripts/setup_worktree.sh` は廃止予定
- または、`setup_worktree.sh` を `create_worktree.sh` へのシンボリックリンクに変更

### 問題: テスト失敗時のデバッグ方法

```bash
# 失敗したテスト単体を verbose 実行
pytest python/tests/<module>/test_<name>.py::TestName::test_method -v -s

# ソースコード確認
cat python/jax_util/<module>/<file>.py
```

### 問題: リンク切れが多い場合

```bash
# リンク監査実行
python scripts/tools/audit_and_fix_links.py

# 詳細レポート確認
cat reports/link_audit.txt

# 手動修正（相対パス統一）
vim documents/<affected-file>.md
```

### 問題: ワークツリー削除に失敗した

```bash
# アクティブなワークツリーを確認
git worktree list

# ワークツリーのプロセスを確認・停止
ps aux | grep <worktree-name>
lsof +D <worktree-path>

# 強制削除（最終手段）
git worktree remove --force .worktrees/<name>
```

---

## 参考資料・関連ドキュメント

- [documents/README.md](./README.md) — ドキュメント体系全体
- [documents/tools/TOOLS_DIRECTORY.md](./tools/TOOLS_DIRECTORY.md) — ツール詳細目録
- [documents/coding-conventions-project.md](./coding-conventions-project.md) — プロジェクト運用規約
- [documents/worktree-lifecycle.md](./worktree-lifecycle.md) — ワークツリー管理規約
- [.github/workflows/](../.github/workflows/) — CI/CD ワークフロー

