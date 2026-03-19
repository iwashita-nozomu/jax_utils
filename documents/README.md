# documents

`documents/` は、このリポジトリの規約と設計の正本です。
`notes/` や `reviews/` は補助資料として残してよいですが、運用ルールそのものは `documents/` に集約します。

## 1. 全体方針

- 高レベルの規約は top-level の `documents/*.md` に置きます。
- 共通コーディング規約は `documents/conventions/common/` に置きます。
- Python 実装向けの詳細規約は `documents/conventions/python/` に置きます。
- `base` の型・Protocol・共通クラスの設計は `documents/design/jax_util/base_components.md` に置きます。
- 安定サブモジュールの API 詳細は `documents/design/jax_util/` に置きます。
- worktree の運用は `documents/worktree-lifecycle.md` に置きます。

## 2. 正本の対応表

| 話題 | 正本 | 補助文書 |
| --- | --- | --- |
| `documents/` 全体の構成 | [README.md](./README.md) | なし |
| 共通コーディング規約の入口 | [README.md](./README.md) | `conventions/common/` |
| Python 規約の入口 | [README.md](./README.md) | `conventions/python/` |
| プロジェクト全体の運用 | [coding-conventions-project.md](./coding-conventions-project.md) | なし |
| 実験環境の運用 | [coding-conventions-experiments.md](./coding-conventions-experiments.md) | なし |
| レビュー文書の運用 | [coding-conventions-reviews.md](./coding-conventions-reviews.md) | なし |
| テスト | [coding-conventions-testing.md](./coding-conventions-testing.md) | なし |
| ログ | [coding-conventions-logging.md](./coding-conventions-logging.md) | なし |
| ソルバー共通ルール | [coding-conventions-solvers.md](./coding-conventions-solvers.md) | なし |
| C++ 実装の最低限ルール | [coding-conventions-cpp.md](./coding-conventions-cpp.md) | なし |
| `base` の型・Protocol・共通クラス | [base_components.md](./design/jax_util/base_components.md) | なし |
| stable API 設計 | [README.md](./README.md) | `design/jax_util/*.md` |
| worktree の作成・削除・吸い出し | [worktree-lifecycle.md](./worktree-lifecycle.md) | [WORKTREE_SCOPE_TEMPLATE.md](./WORKTREE_SCOPE_TEMPLATE.md) |

## 3. 共通規約の章

- [基本方針](./conventions/common/01_principles.md)
  - 読みやすさ、保守性、依存最小の共通原則
- [命名](./conventions/common/02_naming.md)
  - 言語を問わない命名の基本
- [コメント](./conventions/common/03_comments.md)
  - 意図、前提、数式、責務コメントの共通ルール
- [演算子記法（共通）](./conventions/common/04_operators.md)
  - 線形/非線形作用素の基本記法
- [ドキュメント運用](./conventions/common/05_docs.md)
  - `documents/` 全体の書き方と更新方針

## 4. Python 実装規約の章

- [対象](./conventions/python/01_scope.md)
  - Python 規約の適用範囲
- [型エイリアスの方針](./conventions/python/02_type_aliases.md)
  - `Scalar` / `Vector` / `Matrix` の使い分け
- [作用素と継承関係](./conventions/python/03_operators.md)
  - `Operator` / `LinearOperator` / `LinOp` の役割
- [関数の型注釈](./conventions/python/04_type_annotations.md)
  - `dtype`、引数、戻り値、公開 API の型
- [バッチ対応](./conventions/python/05_batching.md)
  - `Vector` / `Matrix` と `LinOp` の扱い
- [コメントの Python 差分](./conventions/python/06_comments.md)
  - 内部補助関数や JAX 制御フローでの補足ルール
- [型チェッカの活用](./conventions/python/07_type_checker.md)
  - pyright と ignore の最小化
- [合成型の定義](./conventions/python/08_composition.md)
  - 合成の増やし方
- [責務分離](./conventions/python/09_file_roles.md)
  - ディレクトリごとの責務
- [依存関係の制約](./conventions/python/10_dependencies.md)
  - stable module 間の依存方向
- [命名規約](./conventions/python/11_naming.md)
  - ファイル名、関数名、公開/内部の区別
- [演算子記法の厳格ルール](./conventions/python/12_operator_rules.md)
  - Python 実装での禁止事項と具体表記
- [JAX/Equinox の運用規約](./conventions/python/15_jax_rules.md)
  - `lax.scan` / `while_loop` / `fori_loop` の使い分け
- [数値安定性](./conventions/python/17_numerical_stability.md)
  - `EPS`、`AVOID_ZERO_DIV`、悪条件ログ
- [ニューラルネットワーク規約](./conventions/python/18_neuralnetwork.md)
  - 実験段階モジュールの扱い

## 5. 運用規約

- [coding-conventions-project.md](./coding-conventions-project.md)
  - プロジェクト全体の位置づけ、文書配置、Docker、branch 方針を扱います。
- [coding-conventions-experiments.md](./coding-conventions-experiments.md)
  - `experiments/` のコード、results branch、最終 JSON の持ち帰りを扱います。
- [coding-conventions-reviews.md](./coding-conventions-reviews.md)
  - `reviews/` の置き場と命名を扱います。
- [worktree-lifecycle.md](./worktree-lifecycle.md)
  - worktree の作成、`WORKTREE_SCOPE.md`、削除、吸い出しを扱います。

## 6. 設計書

- [base_components.md](./design/jax_util/base_components.md)
  - `base` の型エイリアス、作用素 Protocol、`LinOp`、微分作用素、環境設定の正本です。
- [solvers.md](./design/jax_util/solvers.md)
  - `solvers` の stable API 設計
- [optimizers.md](./design/jax_util/optimizers.md)
  - `optimizers` の stable API 設計
- [hlo.md](./design/jax_util/hlo.md)
  - `hlo` の stable API 設計

## 7. 整理方針

- 同じルールを複数ファイルへ重ねて書かず、正本を 1 つ決めます。
- 他ファイルでは要点だけを書き、必要なら正本を参照します。
- 情報を減らすのではなく、重複を減らします。
- `notes/` や `reviews/` の運用に関するルールも、正本は `documents/` に置きます。

## 8. 現在の対象

- 安定サブモジュール: `base` / `solvers` / `optimizers` / `hlo`
- 実験段階: `neuralnetwork`
- 保管領域: `solvers/archive`
- 補助資料: `./notes/`
- レビュー文書: `./reviews/`
- 作業ログ: `./diary/`

## 9. 追記ルール

- 規約を増やすときは、まずこの表で既存の正本を確認します。
- 共通規約なら `documents/conventions/common/` に追記します。
- Python 実装規約なら `documents/conventions/python/` に追記します。
- `base` の型や Protocol を変えるときは `documents/design/jax_util/base_components.md` を更新します。
- stable API を変えるときは `documents/design/jax_util/` の対応文書を更新します。
- worktree 運用を変えるときは `documents/worktree-lifecycle.md` を更新します。
- 実験や notes の運用ルールを変えるときも、まず `documents/` の正本を更新します。
