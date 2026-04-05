# 命名規約（Python）

対象: `python/jax_util/` 配下。

## 要約

- **ファイル名**は `snake_case.py`。
- **関数名**は `snake_case`。
- **公開API** と **内部実装** を名前で区別します。

## ファイル名

- `snake_case.py` を使ってください。
- 略語は一般的なもののみ許可します（例: `pcg`, `minres`, `kkt`, `hlo`）。
- 役割が分かる語を置き、意味のない接尾辞（`_util` の乱用等）は避けます。
- `python/jax_util/differential_equations/` の concrete problem file は `<problem_name>.py` とし、1 file に 1 problem だけを置きます。

### 禁止/注意

- 大文字を含むファイル名（例: `MyAlgo.py`）は避けます。
- OS差分が出やすい表記（空白・ハイフン混在）は避けます。

## 関数名

- `snake_case` を使ってください。
- **動詞で始める**ことを推奨します（例: `solve_*`, `init_*`, `make_*`, `update_*`, `check_*`, `dump_*`）。

## クラス名・Protocol名

- クラス名と `Protocol` 名は `PascalCase` を使ってください。
- 役割語は末尾に置き、型空間や責務は先頭で明示します。
  - 例: `LinearOperator`, `VectorOptimizationProblem`, `PyTreeOptimizationState`
- 意味の薄い短縮形は避けます。
  - 例: `OptimizeProblem`, `OptimizeProblemState`, `PytrreeOptim` は使いません。
- 制約付きの最適化契約は `WithConstraint` 系ではなく `Constrained*` 系で統一します。
  - 例: `ConstrainedOptimizationProblem`, `ConstrainedVectorOptimizationProblem`

## 最適化 Protocol の命名

- 汎用の最適化契約は `python/jax_util/base/protocols.py` に置きます。
  - `OptimizationProblem`
  - `ConstrainedOptimizationProblem`
  - `OptimizationState`
  - `ConstrainedOptimizationState`
- 空間ごとの特殊化は、対象空間を先頭に付けて命名します。
  - ベクトル空間: `VectorOptimizationProblem`, `VectorOptimizationState`
  - PyTree パラメータ空間: `PyTreeOptimizationProblem`, `PyTreeOptimizationState`
  - 関数空間: `FunctionalOptimizationProblem`
- 制約付きの特殊化も同じ規則で命名します。
  - `ConstrainedVectorOptimizationProblem`
  - `ConstrainedPyTreeOptimizationProblem`
  - `ConstrainedFunctionalOptimizationProblem`
- 同じ概念に対して `OptimizeProblem` 系と `*OptimizationProblem` 系を併存させません。
- 既存名の読み替えに頼る互換 alias は置きません。命名を変える場合は参照側も同時に書き換えます。

### 公開API と内部関数

- **公開API**: モジュールの `__all__` に載せる関数/クラス。
  - 名前は先頭 `_` なし。
  - 例: `pcg_solve`, `minres_solve`, `kkt_block_solver`
- **内部実装**: 先頭 `_`。
  - 例: `_pdipm_solve`, `_sym_ortho`

### module import 規約

- concrete problem module の canonical import は `jax_util.differential_equations.<problem_name>.<symbol>` に固定します。
- `differential_equations` の concrete problem は package root の `__init__.py` で平坦に再 export しません。
- 各 concrete problem module は少なくとも `problem` を公開し、必要なら `*_term` と `make_*` を公開します。
- したがって、利用側は `from jax_util.differential_equations.lotka_volterra import problem` のように読み込みます。

### テスト用補助

- テストファイルでは `test_*` を使います。
- `python file.py` 直実行用の補助を置く場合は、外部から使わないことが分かるよう `_run_*` を推奨します。

## 例

- `pdipm_solve`（公開）→ `_pdipm_solve`（内部）
- `initialize_kkt_state`（公開）→ `_kkt_block_solver`（内部）
- `OptimizationProblem`（汎用）→ `VectorOptimizationProblem` / `PyTreeOptimizationProblem` / `FunctionalOptimizationProblem`（空間特殊化）
- `ConstrainedOptimizationProblem`（汎用）→ `ConstrainedVectorOptimizationProblem` / `ConstrainedPyTreeOptimizationProblem`
- `jax_util.differential_equations.lotka_volterra.problem`（canonical import）
