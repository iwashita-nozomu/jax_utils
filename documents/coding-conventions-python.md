# Python コーディング規約（目次）

この文書は、`python/jax_util/` 配下の実装を対象とした**目次**です。
各章は `documents/conventions/python/` に分割してあります。

## 目次（章ごとの要約）
1. [対象と関連文書](./conventions/python/01_scope.md) — 対象は `python/jax_util/`、テスト・ログ・ソルバーは専用文書を参照します。
2. [型エイリアスの方針](./conventions/python/02_type_aliases.md) — `Scalar` / `Vector` / `Matrix` を基準に型を統一します。
3. [作用素と継承関係](./conventions/python/03_operators.md) — `Operator` / `LinearOperator` を基準に設計します。
4. [関数の型注釈](./conventions/python/04_type_annotations.md) — 引数・戻り値は意味のある型で明示します。
5. [バッチ対応](./conventions/python/05_batching.md) — `LinOp` で 1D/2D を内部で分岐します。
6. [コメント](./conventions/python/06_comments.md) — 意図と前提を丁寧に説明します。
7. [型チェッカの活用](./conventions/python/07_type_checker.md) — cast ではなく pyright を優先します。
8. [合成型の定義](./conventions/python/08_composition.md) — 合成は演算子として最小限に定義します。
9. [ファイルごとの役割](./conventions/python/09_file_roles.md) — `base` と `Algorithms` の役割を明確化します。
10. [依存関係の概要](./conventions/python/10_dependencies.md) — `base` を基盤とした依存構造を整理します。
11. テスト規約（共通）: `documents/coding-conventions-testing.md` を参照します。
12. [演算子記法（線形/非線形・投影/前処理）](./conventions/python/12_operator_rules.md) — `@` と `*` の厳格運用、禁止事項、投影/前処理の表記をまとめます。
13. [JAX/Equinox の運用規約](./conventions/python/15_jax_rules.md) — 反復・JIT・デバッグ方針を定めます。
14. ソルバー規約（共通）: `documents/coding-conventions-solvers.md` を参照します。
15. [数値安定性](./conventions/python/17_numerical_stability.md) — ゼロ除算回避と悪条件ログを徹底します。
16. [ニューラルネットワーク規約](./conventions/python/18_neuralnetwork.md) — `neuralnetwork` 配下の設計と役割を整理します。