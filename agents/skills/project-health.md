# Project Health

## Purpose

日次・週次・継続運用での健康状態を監視し、automation の壊れ方を早めに見つけます。

## Use This For

- project health の監視
- CI / CD 健全性確認
- routine maintenance の起点作り
- 運用上の drift 検出

## Inputs

- 監視対象
- interval
- 必要なら health 指標

## Outputs

- health report
- 継続監視に必要な action item
- 追跡対象の regression

## Implementation Surface

- `.github/skills/07-health-monitor/`
- `.github/skills/16-ci-cd-integration/`
- `.github/skills/17-project-health/`

## Boundary

- 変更差分のレビューは `agents/skills/code-review.md` を使います。
- 実験の比較設計は `agents/skills/research-workflow.md` を使います。
- repo-wide review の最上位入口としては `agents/skills/project-review.md` を使います。
