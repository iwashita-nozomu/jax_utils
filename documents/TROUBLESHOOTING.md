# ❓ トラブルシューティング＆ FAQ

> よくある問題と解決策（初心者～上級者向け）

______________________________________________________________________

## 🔍 **問題から探す（五十音順）**

### 環境・セットアップ

#### **Python がインストールされていない / バージョンが古い**

**症状：** `python3: command not found` または `Python 3.10+` が必要

`````bash
# 1. Python バージョン確認
python3 --version

# 2. Docker 環境を使用（推奨）
docker-compose build && docker-compose run python bash
```yaml

**参照：** [セットアップガイド](./worktree-lifecycle.md)

______________________________________________________________________

## **モジュールが見つからない**

**症状：** `ModuleNotFoundError: No module named 'jax'`

```bash
# 解決策
pip install -e ".[dev]"  # 開発環境依存をインストール
# または
pip install -e .         # 最小限の依存
```text

**参照：** [requirements.txt](../docker/requirements.txt)

______________________________________________________________________

## **ワークツリー作成に失敗**

**症状：**

```text
fatal: cannot create worktree with file '...'
```text

```bash
# 原因: 既にワークツリーが存在している可能性
ls -la .worktrees/

# 削除して再作成
rm -rf .worktrees/work-my-feature-YYYYMMDD/
bash scripts/setup_worktree.sh work/my-feature-YYYYMMDD
```text

**参照：** [ワークツリーライフサイクル](worktree-lifecycle.md)

______________________________________________________________________

## コーディング・実装

### **型チェックエラー（pyright 失敗）**

**症状：**

```text
error: Expression of type "..." is not assignable to declared type "..."
```text

**解決策（優先順）：**

1. **型注釈を追加する**

```python
# ❌ 悪い例
def compute(x):
    return x + 1

# ✅ 良い例
def compute(x: jnp.ndarray) -> jnp.ndarray:
    return x + 1
```text

2. **スタイルガイドを確認**

```bash
# Python 実装規約確認
cat documents/coding-conventions-python.md
```text

3. **型注釈の自動追加**

```bash
python -m pylancer_refactor source.addTypeAnnotation
```text

**参照：** [Python コーディング規約](coding-conventions-python.md)

______________________________________________________________________

## **テストが失敗する**

**症状：**

```text
FAILED tests/test_import.py::test_import - AssertionError: ...
```text

**デバッグ手順：**

```bash
# 1. 単体テスト実行（詳細表示）
pytest tests/test_import.py -v

# 2. ログ保存版で実行
bash scripts/run_pytest_with_logs.sh

# 3. カバレッジ確認
pytest --cov=python/ tests/
```text

**参照：** [テスト規約](./coding-conventions-testing.md)

______________________________________________________________________

## **リント・フォーマッタ エラー（ruff / pyright の警告）**

**症状：** linting エラーが多数表示される

```bash
# 1. 自動修正を試す
ruff check --fix python/

# 2. pyre / pyright で詳細確認
pyright python/

# 3. 全チェック実行
make ci
```yaml

**参照：** [チェックリスト（CI）](./FILE_CHECKLIST_OPERATIONS.md#%E3%83%81%E3%82%A7%E3%83%83%E3%82%AF%E3%83%AA%E3%82%B9%E3%83%883)

______________________________________________________________________

## ドキュメント・Markdown

### **Markdown リント エラー（markdownlint 失敗）**

**症状：** `MD001: heading levels`、`MD040: fenced code`

```bash
# 1. 全スキャン
python scripts/tools/check_markdown_lint.py .

# 2. 自動修正を試す
python scripts/tools/fix_markdown_headers.py .
python scripts/tools/fix_markdown_code_blocks.py .

# 3. mdformat で形式統一
mdformat documents/ --wrap 100
```text

**よくあるエラー：**

| エラー    | 原因                          | 解決策                             |
| --------- | ----------------------------- | ---------------------------------- |
| **MD001** | ヘッダー段階の飛び（H1 → H3） | fix_markdown_headers.py を実行     |
| **MD040** | コードブロック言語不指定      | fix_markdown_code_blocks.py を実行 |
| **MD009** | 末尾の空白                    | mdformat で自動修正                |
| **MD010** | ハードタブ混在                | mdformat で自動修正                |

**参照：** [Markdown 規約](./coding-conventions.md)

______________________________________________________________________

## **リンクが壊れている**

**症状：** VS Code で「ファイルが見つかりません」警告

1. **相対パスを確認**

```markdown
❌ 悪い例
[Reference](../../../documents/coding-conventions.md)

✅ 正しい例（workspace ルートから相対）
[Reference](documents/coding-conventions-project.md)
```

2. **ツールで自動修正**

```bash
python scripts/tools/audit_and_fix_links.py documents/
```text

**参照：** [コーディング規約](./coding-conventions-project.md#%E3%83%AA%E3%83%B3%E3%82%AF%E8%A8%98%E6%B3%95)

______________________________________________________________________

### CI・ビルド

#### **`make ci` が失敗する**

**症状：**

```text
Error in step 1: pytest failed with exit code 1
```text

**デバッグ順序：**

```bash
# 1. pytest 個別実行
pytest tests/ -v

# 2. pyright 実行
pyright python/

# 3. ruff チェック
ruff check python/

# 4. mdformat チェック
mdformat --check documents/
```text

**参照：** [CI フロー](./FILE_CHECKLIST_OPERATIONS.md#%E3%83%81%E3%82%A7%E3%83%83%E3%82%AF%E3%83%AA%E3%82%B9%E3%83%883)

______________________________________________________________________

## **Git コマンド失敗（ワークツリー系）**

**症状：**

```text
fatal: working directory is controlled by 'main' worktree
```text

**解決策：**

```bash
# 1. 正しい worktree にいるか確認
pwd
# 期待: /workspace/.worktrees/my-feature-name/

# 2. worktree 一覧確認
git worktree list

# 3. 必要に応じて削除して再作成
cd ../..  # ルートに戻る
git worktree remove .worktrees/work-my-feature-YYYYMMDD/
bash scripts/setup_worktree.sh work/my-feature-YYYYMMDD
```text

**参照：** [ワークツリー規約](./WORKTREE_SCOPE_TEMPLATE.md)

______________________________________________________________________

## レビュー・統合

### **PR / レビューでリジェクトされた**

**よくある理由と対応：**

| 理由                   | チェック項目              | 対処                                                                                                                    |
| ---------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| **テスト不足**         | `pytest` で全テスト成功？ | [テスト規約](./coding-conventions-testing.md) 参照                                                                      |
| **Markdown エラー**    | `mdformat --check`？      | `mdformat --fix` で自動修正                                                                                             |
| **型エラー**           | `pyright` 警告ない？      | 型注釈追加、または除外設定                                                                                              |
| **コード品質**         | `ruff check` 警告ない？   | `ruff check --fix` で自動修正                                                                                           |
| **ドキュメント未更新** | API 変更時に .md 更新？   | [チェックリスト4](./FILE_CHECKLIST_OPERATIONS.md#%E3%83%81%E3%82%A7%E3%83%83%E3%82%AF%E3%83%AA%E3%82%B9%E3%83%884) 参照 |

______________________________________________________________________

#### **統合後エラー（main にマージ後に問題発生）**

**症状：** `main` で他の機能が壊れた

```bash
# 1. デグレード確認
git log -n 5 --oneline main

# 2. 最新の main で再テスト
git checkout main && make ci

# 3. 問題の commit 特定
git bisect start
git bisect bad main
git bisect good HEAD~5  # 5 commit 前は OK と仮定
```yaml

**参照：** [レビュー手順](./REVIEW_PROCESS.md)

______________________________________________________________________

## 🚀 **よくある質問（FAQ）**

### Q1: 新規機能を追加するにはどうすればいい？

**A:**

1. [ワークツリー作成](./WORKTREE_SCOPE_TEMPLATE.md) — WORKTREE_SCOPE.md に機能説明を記載
1. [Python 規約](./coding-conventions-python.md) を確認 → 実装
1. [テスト](./coding-conventions-testing.md) を作成
1. `make ci` でチェック
1. [レビュー手順](./REVIEW_PROCESS.md) に従って PR → マージ

**チェックリスト：** [FILE_CHECKLIST_OPERATIONS.md](./FILE_CHECKLIST_OPERATIONS.md)

______________________________________________________________________

### Q2: ドキュメント（.md）を追加する流れは？

**A:**

1. ファイル作成：`documents/<topic>.md`
1. `mdformat documents/` で形式統一
1. `python scripts/tools/check_markdown_lint.py documents/` でリント確認
1. ハブ [documents/README.md](./README.md) にリンク追加
1. `git add` → コミット

**規約：** [Markdown 規約](./coding-conventions.md)

______________________________________________________________________

### Q3: Python コード内の JAX 使用方法は？

**A:**

```python
import jax
import jax.numpy as jnp
from jax_util.core import LinOp

# JAX の基本使用パターン
@jax.jit  # JIT コンパイル
def forward(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(x ** 2)

# 微分
grad_f = jax.grad(forward)
```yaml

**詳細：** [Python 実装ガイド](./coding-conventions-python.md)

______________________________________________________________________

## Q4: テストの書き方がわかりません

**A:**

```python
import pytest
from jax_util.core import LinOp

class TestLinOp:
    """LinOp のテストクラス"""

    def test_linop_creation(self):
        """テスト1: LinOp 作成"""
        linop = LinOp(...)
        assert linop is not None

    def test_linop_adjoint(self):
        """テスト2: 随伴演算子"""
        linop = LinOp(...)
        adj_linop = linop.adjoint()
        assert adj_linop is not None
```yaml

**詳細：** [テスト規約](./coding-conventions-testing.md)

______________________________________________________________________

### Q5: Markdown ファイルにコード例を挿入する方法は？

**A:**

````markdown
## コード例

### Python

```python
def hello():
    print("Hello, jax_util!")
```text

### Bash

```bash
make ci
```text

### YAML

```yaml
image: python:3.11
steps:
  - run: pytest
```text
`````

**規約：** [Markdown 規約](./coding-conventions.md)

______________________________________________________________________

### Q6: 実装時に「型が合わない」と言われました

**A:**

````python
# 問題のコード
def compute(x):
    return jnp.sum(x)

# 修正：型注釈を追加
from typing import Union
import jax.numpy as jnp

def compute(x: Union[jnp.ndarray, float]) -> jnp.ndarray:
    return jnp.sum(x)
```yaml

**参照：** [Python 型注釈ガイド](./coding-conventions-python.md#%E5%9E%8B%E6%B3%A8%E9%87%88)

______________________________________________________________________

## Q7: リポジトリの構造はどうなっているのか？

**A:**

```text
workspace/
├── README.md              ← 最初にここを読む
├── documents/             ← 規約・ガイド・設計
│   ├── README.md         ← ドキュメント ハブ
│   ├── coding-conventions-python.md
│   ├── conventions/       ← 言語別規約
│   └── tools/             ← ツール説明
├── python/                ← Python ソースコード
│   └── jax_util/
├── tests/                 ← テスト
├── scripts/               ← ユーティリティスクリプト
└── docker/                ← Docker 設定
```yaml

**詳細：** [プロジェクト設計](./design/README.md)

______________________________________________________________________

### Q8: Markdown の自動修正をしたい

**A:**

```bash
# 1. 形式統一（mdformat）
mdformat documents/

# 2. ヘッダー修正（MD001）
python scripts/tools/fix_markdown_headers.py .

# 3. コードブロック言語指定（MD040）
python scripts/tools/fix_markdown_code_blocks.py .

# 4. すべてのリント確認
python scripts/tools/check_markdown_lint.py .
```yaml

**ツール詳細：** [ツール一覧](./tools/README.md)

______________________________________________________________________

## Q9: CI（Continuous Integration）で何をチェックしているの？

**A:**

```bash
make ci  # 以下を順に実行：

1. pytest ..................... ユニットテスト
2. pyright .................... 型チェック
3. ruff check ................ コード品質・リント
4. mdformat --check ......... Markdown 形式
5. check_markdown_lint.py .... Markdown ルール
```yaml

**各項目の詳細：** [チェックリスト3](./FILE_CHECKLIST_OPERATIONS.md#%E3%83%81%E3%82%A7%E3%83%83%E3%82%AF%E3%83%AA%E3%82%B9%E3%83%883)

______________________________________________________________________

### Q10: Pull Request をマージするまでの手順は？

**A:**

1. 新ブランチ作成 → 実装 → テスト（`make ci`）
1. `git add . && git commit`
1. `git push origin <branch>`
1. GitHub で PR 作成
1. 自動 CI 実行 → 二重チェック
1. レビューア承認 → マージ

**詳細：** [レビュー手順](./REVIEW_PROCESS.md)

______________________________________________________________________

## 🛠️ **便利なコマンド集**

| 用途                  | コマンド                                        | 説明                           |
| --------------------- | ----------------------------------------------- | ------------------------------ |
| **すべてのテスト**    | `make ci`                                       | pytest + linter + 型チェック   |
| **Python テストのみ** | `pytest tests/`                                 | ユニットテスト実行             |
| **型チェック**        | `pyright python/`                               | JAX 型整合性確認               |
| **コード品質**        | `ruff check python/`                            | 自動修正版: `ruff check --fix` |
| **Markdown 修正**     | `mdformat documents/`                           | 形式統一                       |
| **Markdown リント**   | `python scripts/tools/check_markdown_lint.py .` | ルール確認                     |

______________________________________________________________________

## 📚 **参考資料へのリンク**

- [クイックスタート](../QUICK_START.md)
- [コーディング規約（統合版）](./conventions/README.md)
- [チェックリスト（8 フロー）](./FILE_CHECKLIST_OPERATIONS.md)
- [レビュー手順](./REVIEW_PROCESS.md)
- [ツール一覧](./tools/README.md)

______________________________________________________________________

## 💬 **まだ疑問が残っている？**

1. **[documents/README.md](./README.md)** の目的別ナビゲーションで、該当ドキュメントを探す
1. **スタッフに相談** — AGENTS.md を参照
1. **正本をたどる** — [documents/README.md](./README.md) の目的別ナビゲーションを確認
````
