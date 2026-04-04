# Docs Completeness Review

## Purpose

文書が「ある」だけでなく、読者が作業や判断に必要な情報を欠かさず持っているかをレビューします。

## Use This For

- README、設計文書、workflow 文書の不足確認
- 入口、対象、前提、手順、出力先、禁止事項の欠落確認
- 実装変更後に docs が追随しているかの確認
- 文書単体で読めるかの確認

## Must Read Before Reviewing

- `documents/coding-conventions-project.md`
- 対象の文書群
- 必要なら対応する code / scripts / tests

## Inputs

- 対象文書
- 関連する code / workflow / command
- 期待する読者と用途

## Outputs

- findings-first の completeness review
- missing section、missing command、missing rationale、missing output path の指摘

## Mandatory Checklist

- 文書だけで対象、目的、入口が分かる
- 手順文書なら入力、出力、主要コマンドがある
- 規約文書なら禁止、必須、許可、任意が明示されている
- 実験文書なら run_name、result path、report path がある
- 設計文書なら責務境界と更新先がある
- 変更後に docs / code / tests の三点が揃っている

## Boundary

- Markdown の形式だけを見るなら `agents/skills/md-style-check.md` を使います。
- 文書間の矛盾や曖昧性を潰すなら `agents/skills/docs-consistency-review.md` を使います。

## Implementation Surface

- `.github/skills/21-docs-completeness-review/`
