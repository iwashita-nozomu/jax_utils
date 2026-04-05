# `differential_equations` API 詳細設計

この文書は、`python/jax_util/differential_equations/` に置く
problem metadata、微分方程式 operator protocol、1 問題 1 ファイルの配置規則を整理します。

## 1. 対象

- `python/jax_util/differential_equations/` の公開 API を対象にします。
- 当面は、微分方程式の問題メタデータと `L(f)` 形式の operator protocol を扱います。
- concrete problem module は `python/jax_util/differential_equations/<problem_name>.py` に 1 問題 1 ファイルで置きます。
- solver 実装、離散化実装、benchmark 実行、experiment report は対象外です。

## 2. 公開 API の位置づけ

- `EquationKind`
  - ODE / PDE / DAE などの大分類を表す語彙です。
- `ConditionKind`
  - 初期値問題、境界値問題、終端条件つき問題などの条件種別を表す語彙です。
- `StateFunction` / `ResidualFunction`
  - 未知関数 `f` と残差関数 `L(f)` の最小 callable 契約です。
- `DifferentialEquationOperator`
  - 関数を引き数に取り、残差関数を返す演算子契約です。
  - 標準形は `operator(f) = residual_function` です。
- `DifferentialEquationTerm`
  - operator と tag を束ねる最小単位です。
  - `equation` tag を含む場合は `L(f)=0` を暗黙に仮定します。
- `DifferentialEquationProblem`
  - 1 つの問題を識別するための最小メタデータです。
  - `name`、方程式種別、条件種別、状態次元、空間次元、時間区間を持ちます。
  - `tags` と `references` で後続の分類や文献参照を付与できます。
  - `terms` で問題に属する operator term を保持できます。
- `<problem_name>.py`
  - 問題ごとの concrete module です。
  - 各 module は `problem` と、必要な `*_term` や `make_*` 関数を export します。
  - import 入口は `jax_util.differential_equations.<problem_name>.<symbol>` に固定します。
  - この import 規約の正本は `documents/conventions/python/11_naming.md` に置きます。

## 3. 共通設計

- problem catalog は import-safe を必須にします。
- operator protocol は JAX autodiff と transform に乗せやすい callable 形を必須にします。
- 具体的な問題 term は `jax.grad`、`jax.jvp`、`jax.jacfwd`、`jax.jacrev` などの autodiff primitive を使って記述しなければなりません。
- concrete problem module は 1 問題 1 ファイルとし、module 名は `snake_case` を必須にします。
- concrete problem module は少なくとも `problem` を export し、必要な term を file-local に閉じ込めず公開名で参照できるようにします。
- データ構造は `frozen=True` と `slots=True` の dataclass を使い、軽量で不変なメタデータとして扱います。
- validation は constructor 境界で行い、不正な `state_dimension`、`spatial_dimension`、`time_interval` を早期に拒否します。
- `time_interval` は任意です。定常 PDE や時間を明示しない問題を表現できるようにします。
- `equation` tag を含む term は、右辺ゼロの残差方程式として扱います。
- 具体的な離散化済み solver や runtime state はこの段階では持ち込みません。

## 4. 依存関係

- `differential_equations` は `base` の `Vector` に依存します。
- `differential_equations` は `neuralnetwork`、`functional`、`solvers` に依存しません。
- concrete problem module 同士は相互依存しません。
- 下流の solver、benchmark、experiment code からは、この package の problem catalog を参照して構いません。
- 依存方向は `problem catalog -> downstream consumer` の一方向に固定します。

## 5. 拡張方針

- 最初の追加対象は、ODE / PDE / BVP / IVP の代表問題を catalog として段階的に増やすことです。
- starter modules として `lotka_volterra.py` と `heat_1d_dirichlet.py` を置き、この形を後続 problem の雛形にします。
- 右辺非ゼロの問題が必要になった場合は、`equation` tag 以外の tag と補助 metadata を追加して表現します。
- 共通メタデータが不足した場合は、既存 problem の互換性を壊さない default 付き field 追加を優先します。
- 問題メタデータを超える責務が必要になった場合は、metadata と execution 用コードを別 module に分離します。

## 6. 禁止事項

- この package に solver 固有の runtime state や optimizer 設定を混在させることを禁止します。
- 1 つの file に複数 problem を混在させることを禁止します。
- 具体 problem を手書きの有限差分や数値微分で定義することを禁止します。
- 実験 run 結果、report、figure を `python/jax_util/differential_equations/` に置くことを禁止します。
- `neuralnetwork` や既存 `functional` 本体の都合で problem catalog の責務を広げることを禁止します。
