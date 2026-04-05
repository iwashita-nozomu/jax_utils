# Worktree Scope

## Branch

- `work/differential-equations-problems-20260405`

## Purpose

- 微分方程式の問題集を置くための新しいサブモジュールを準備する
- 既存の `neuralnetwork` や `functional` 本体と衝突しない独立 package を切る
- まずは problem catalog と import / test の土台までを作る

## Editable Directories

- `python/jax_util/differential_equations/`
- `python/tests/differential_equations/`
- `notes/worktrees/worktree_differential_equations_problems_2026-04-05.md`
- `WORKTREE_SCOPE.md`

## Runtime Output Directories

- なし

## Non-Goals

- 具体的な ODE / PDE 問題の大量実装
- 既存 `neuralnetwork` module の改造
- 実験 run や report の追加

## Initial Plan

1. `differential_equations` package を新設する
1. 問題定義と problem set を表す最小の型を置く
1. import と validation の最小 test を足す
1. 以後の ODE / PDE 問題追加はこの package 配下で行う
