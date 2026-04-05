# Worktree Log

## 2026-04-05

- `work/differential-equations-problems-20260405` を `main` の `6030e1e` から作成
- `python/jax_util/differential_equations/` を新設
- 微分方程式の問題集を置くための最小 catalog と test の土台を追加
- `documents/design/jax_util/differential_equations.md` を追加し、problem catalog の責務と依存境界を固定
- `protocols.py` を追加し、`L(f)` を返す微分作用素と `equation => residual=0` の tag 約束を定義
- 具体 problem は JAX autodiff primitive で記述する方針を文書と Protocol docstring に反映
- `lotka_volterra.py` と `heat_1d_dirichlet.py` を追加し、1 問題 1 ファイルの import 形を starter modules として実装
- import 形の正本を `documents/conventions/python/11_naming.md` などの規約文書へ移し、README 依存を避けた
- `problem_set.py` をやめ、単一問題 metadata だけを `problem.py` に寄せて `ProblemSet` 抽象を外した

## Notes

- 現在の `/workspace` には未コミット変更が多いため、この worktree は commit 済み `main` から clean に切っている
- 後続の具体問題は ODE / PDE / BVP / IVP を段階的に増やす
