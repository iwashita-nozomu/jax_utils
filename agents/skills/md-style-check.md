# Markdown Style Check

## Purpose

Markdown 文書の体裁、lint、format を機械的に確認します。

## Use This For

- `mdformat` の適用確認
- markdown lint rule の確認
- code fence、header、list、trailing space の確認
- doc 変更後の速い style gate

## Must Read Before Reviewing

- `documents/coding-conventions.md`
- `.markdownlint.json`
- `scripts/tools/check_markdown_lint.py`

## Inputs

- 対象 Markdown path
- 必要なら対象ルールや除外範囲

## Outputs

- formatting / lint findings
- `mdformat` と markdown lint の実行結果

## Default Commands

- `mdformat --check <path>`
- `python3 scripts/tools/check_markdown_lint.py <path>`

## Boundary

- 文書の内容不足は `agents/skills/docs-completeness-review.md` を使います。
- 文書間の矛盾や曖昧性は `agents/skills/docs-consistency-review.md` を使います。

## Implementation Surface

- `.github/skills/22-md-style-check/`
- `.github/skills/10-documentation-validation/`
