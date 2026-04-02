# コーディング規約（共通）

この文書は、general なコーディング方針の全体像を短くまとめる文書です。
共通ルールの詳細は各章に分けますが、この文書自体も概要が読めるように保ちます。

## 共通規約（章ごとの要約）

1. [基本方針](./conventions/common/01_principles.md) — 読みやすさ・保守性・依存最小を最優先にします。
1. [命名](./conventions/common/02_naming.md) — 役割が伝わる名前を使い、省略を最小限にします。
1. [コメント](./conventions/common/03_comments.md) — 意図と前提を明確にし、数式や安定性の注意を優先します。
1. [演算子記法（共通）](./conventions/common/04_operators.md) — 適用は `@`、合成は `*` を基本にします。
1. [ドキュメント運用](./conventions/common/05_docs.md) — 実装変更に合わせて文書も更新し、禁止事項は `禁止` と明記します。

## Markdown 書式修正ルール

**必須ルール: md ファイル編集後は必ず `mdformat` ツールで書式を直してください。**

### Canonical Only

- `documents/` と `scripts/` の Markdown には正本だけを残します。
- `.bak`、proposal、report、summary のような履歴用ファイルを常設しません。
- 履歴を残したい場合は Git 履歴を使い、必要な内容だけを既存の正本へ統合します。

### 実行方法

````bash
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

Markdown 整形とリンク監査は `mdformat` や `scripts/tools/audit_and_fix_links.py` で個別に実行します。
`make ci` や `scripts/ci/run_all_checks.sh` は Python 系チェックの正本であり、Markdown チェックは別フローとして扱います。

## Python 実装向けの追加規約

- [Python 規約](./coding-conventions-python.md)
- [ソルバー規約](./coding-conventions-solvers.md)
- [テスト規約](./coding-conventions-testing.md)
- [ログ規約](./coding-conventions-logging.md)
- [プロジェクト運用規約](./coding-conventions-project.md)
````
