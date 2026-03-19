# コーディング規約（共通）

この文書は、general なコーディング方針の全体像を短くまとめる文書です。
共通ルールの詳細は各章に分けますが、この文書自体も概要が読めるように保ちます。

## 共通規約（章ごとの要約）

1. [基本方針](./conventions/common/01_principles.md) — 読みやすさ・保守性・依存最小を最優先にします。
2. [命名](./conventions/common/02_naming.md) — 役割が伝わる名前を使い、省略を最小限にします。
3. [コメント](./conventions/common/03_comments.md) — 意図と前提を明確にし、数式や安定性の注意を優先します。
4. [演算子記法（共通）](./conventions/common/04_operators.md) — 適用は `@`、合成は `*` を基本にします。
5. [ドキュメント運用](./conventions/common/05_docs.md) — 実装変更に合わせて文書も更新します。

## Markdown 書式修正ルール

**必須ルール: md ファイル編集後は必ず `mdformat` ツールで書式を直してください。**

### 実行方法

```bash
# 単一ファイル編集後
mdformat path/to/file.md

# 複数ファイル一括修正
mdformat path/to/docs/ scripts/

# 修正前に確認（ドライラン）
mdformat --check path/to/file.md
```yaml

## 対象ファイル

- `documents/` 配下の全 `.md` ファイル
- `scripts/` 配下の全 `.md` ファイル
- `.github/` 配下の全 `.md` ファイル
- `README.md` / `WORKTREE_SCOPE.md` など

### ツール設定

`mdformat` は以下の拡張を有効にしています：

- `mdformat-gfm`: GitHub Flavored Markdown 対応
- `mdformat-myst`: MyST 対応
- `mdformat-front-matters`: YAML front matter 対応

### CI 統合

Git commit 前に、`make ci` または `scripts/ci/run_all_checks.sh` 実行時に Markdown チェックが含まれます。

## Python 実装向けの追加規約

- [Python 規約](./coding-conventions-python.md)
- [ソルバー規約](./coding-conventions-solvers.md)
- [テスト規約](./coding-conventions-testing.md)
- [ログ規約](./coding-conventions-logging.md)
- [プロジェクト運用規約](./coding-conventions-project.md)
