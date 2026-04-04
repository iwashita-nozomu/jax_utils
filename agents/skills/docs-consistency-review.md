# Docs Consistency Review

## Purpose

文書間の矛盾、曖昧な規範表現、入口のねじれ、古い説明の残存をレビューします。

## Use This For

- 正本と adapter の矛盾確認
- 同じ概念の多重定義確認
- `禁止` / `必須` / `許可` / `任意` の崩れ確認
- workflow、skill、role、routing の説明差分確認

## Must Read Before Reviewing

- `documents/coding-conventions-project.md`
- `AGENTS.md`
- `agents/README.md`
- 必要なら対象領域の正本文書

## Inputs

- 対象文書群
- 正本と adapter の対応関係
- 関連する machine-readable config

## Outputs

- contradiction / ambiguity findings
- どの文書を正本に戻すべきかの指摘
- 削除候補や adapter の薄化候補

## Mandatory Checklist

- 正本と adapter が同じ role / workflow / rule を別内容で述べていない
- 1 つの概念に複数の入口がある場合、正本が明示されている
- 曖昧な規範表現が残っていない
- stale な historical wording が残っていない
- machine-readable config と human-facing summary が矛盾していない

## Boundary

- 文書の欠落は `agents/skills/docs-completeness-review.md` を使います。
- Markdown 形式は `agents/skills/md-style-check.md` を使います。

## Implementation Surface

- `.github/skills/23-docs-consistency-review/`
