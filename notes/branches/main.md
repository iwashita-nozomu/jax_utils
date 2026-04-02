# Main Branch Summary

## Branch

- `main`

## Role

- 再生成可能な code、文書、最小限の実験雛形を集約する既定 branch
- 長時間実験の生成物は直接は置かず、results branch と note への入口を整える branch

## What To Read First

- 全体方針:
  - [coding-conventions-project.md](/workspace/documents/coding-conventions-project.md)
  - [coding-conventions-experiments.md](/workspace/documents/coding-conventions-experiments.md)
- 実験 note の入口:
  - [notes/README.md](/workspace/notes/README.md)
  - [notes/branches/README.md](/workspace/notes/branches/README.md)

## Current Position

- Smolyak 実験コードそのものは `main` に入っている
- 旧版結果の要約は [legacy_smolyak_results_20260316.md](/workspace/notes/experiments/legacy_smolyak_results_20260316.md) にある
- tuned 実験と runner modularization の判断は worktree note と branch note から辿れる

## Interpretation

- `main` は結果ファイル本体の保管場所ではなく、再現のための code と、判断履歴への入口を保つ場所として扱う
- そのため branch note を厚めに残し、results branch へ飛ばないと分からない状態を減らします
