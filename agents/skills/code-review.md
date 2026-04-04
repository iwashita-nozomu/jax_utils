# Code Review

## Purpose

変更差分を correctness、設計、保守性の観点でレビューします。

## Use This For

- PR 前後の変更レビュー
- doc / code / test の整合確認
- refactor での境界確認
- 規約逸脱や過剰修正の検出

## Inputs

- diff または対象ファイル群
- 期待する受け入れ条件
- 必要なら関連 docs / tests

## Outputs

- severity 付き findings
- required change
- 根拠となるファイルやコマンド

## Review Stance

- 人ではなく差分と主張に対して厳しく振る舞います。
- 「問題がない理由」を探すより、「壊れる理由」を先に探します。
- 証拠が弱い実装説明、薄いテスト、過剰な README 記述には強く反応します。
- 完了判定より finding の発見を優先します。

## Implementation Surface

- `.github/skills/02-code-review/`
- `.github/skills/12-code-review-refactoring/`

## Boundary

- 速い自動検査だけなら `agents/skills/static-check.md` を使います。
- Python 差分で pyright、ruff、型追跡を強く見る場合は `agents/skills/python-review.md` を使います。
- 実験主張の批判的評価は `agents/skills/critical-review.md` を使います。
