# Python コーディング規約（目次）

この文書は、`python/jax_util/` 配下の Python 実装向け規約の目次です。
規約本文の見取り図を示しつつ、この文書単体でも対象範囲が分かるように保ちます。

## 現在の対象

- 安定サブモジュール: `base` / `solvers` / `optimizers` / `hlo`
- 実験段階: `neuralnetwork`

## 目次（章ごとの要約）

1. [対象](./conventions/python/01_scope.md) — 対象範囲を整理します。
1. [型エイリアスの方針](./conventions/python/02_type_aliases.md) — `Scalar` / `Vector` / `Matrix` を基準に型を統一します。
1. [作用素と継承関係](./conventions/python/03_operators.md) — `Operator` / `LinearOperator` を基準に設計します。
1. [関数の型注釈](./conventions/python/04_type_annotations.md) — 引数・戻り値は意味のある型で明示します。
1. [バッチ対応](./conventions/python/05_batching.md) — `LinOp` で 1D/2D を内部で分岐します。
1. [コメント](./conventions/python/06_comments.md) — 意図と前提に加え、各関数の責務コメントを徹底します。
1. [型チェッカの活用](./conventions/python/07_type_checker.md) — cast ではなく pyright を優先します。
1. [合成型の定義](./conventions/python/08_composition.md) — 合成は演算子として最小限に定義します。
1. [責務分離](./conventions/python/09_file_roles.md) — どのディレクトリに何を置くかを定めます。
1. [依存関係の制約](./conventions/python/10_dependencies.md) — 安定サブモジュールの依存方向を整理します。
1. [命名規約](./conventions/python/11_naming.md) — ファイル名・関数名のルール（公開/内部の区別）を定めます。
1. テスト規約（共通）: `documents/coding-conventions-testing.md` を参照します。
1. [演算子記法（線形/非線形・投影/前処理）](./conventions/python/12_operator_rules.md) — `@` と `*` の厳格運用、禁止事項、投影/前処理の表記をまとめます。
1. [JAX/Equinox の運用規約](./conventions/python/15_jax_rules.md) — 反復・JIT・デバッグ方針を定めます。
1. ソルバー規約（共通）: `documents/coding-conventions-solvers.md` を参照します。
1. [数値安定性](./conventions/python/17_numerical_stability.md) — ゼロ除算回避と悪条件ログを徹底します。
1. [ニューラルネットワーク規約](./conventions/python/18_neuralnetwork.md) — 実験段階の `neuralnetwork` に関する補足です。

## コメント・ドキュメント修正のルール

### Python ファイル修正後

- `pyright` でタイプチェック（必須）
- `pytest` で単体テスト実行（必須）
- `ruff` でスタイル・import チェック（推奨）

### Markdown ファイル修正後

- **`mdformat` で書式修正（必須）**
  ```bash
  mdformat path/to/file.md
  ```
- 相対パス確認（コーディング規約 [ドキュメント運用](./coding-conventions.md)）
- CI 実行: `make ci` または `scripts/ci/run_all_checks.sh`

詳細は [Markdown 書式修正ルール](./coding-conventions.md) を参照してください。
