# Skill: `python-review` — Python コードレビュー

このディレクトリは GitHub 側の implementation surface です。skill の説明正本は `agents/skills/python-review.md` に置きます。

## Read First

1. `agents/skills/python-review.md`
1. `documents/coding-conventions-python.md`
1. `documents/conventions/python/04_type_annotations.md`
1. `documents/conventions/python/07_type_checker.md`
1. `pyproject.toml`

## Review Focus

- `pyright` の warning / error
- `ruff` warning の放置有無
- 型境界、戻り値、`dtype` の受け渡し
- `cast`、`Any`、`object`、`ignore` の増加
- 実装、docstring、test の三点整合
