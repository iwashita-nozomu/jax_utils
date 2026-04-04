# Research Workflow

## Purpose

単発の run ではなく、比較、反復、仮説更新を含む長期研究フローを整理します。

## Use This For

- baseline と改造案の比較設計
- 反復ごとの branch / note / report 整理
- 次の仮説への引き継ぎ
- 複数 run をまたぐ知見の吸い上げ

## Inputs

- 研究課題
- baseline
- 比較計画
- 既存 notes / reports

## Outputs

- 反復計画
- 比較表
- 次の実験条件
- notes への知見吸収

## Implementation Surface

- `.github/skills/05-research-workflow/`

## Boundary

- 単発の実験実行は `agents/skills/experiment-lifecycle.md` を使います。
- 実装差分のレビューは `agents/skills/code-review.md` を使います。
