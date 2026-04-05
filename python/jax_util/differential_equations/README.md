# Differential Equations

この package は、微分方程式の問題定義と paper-friendly な問題モジュールを置く入口です。

当面の責務は次です。

- ODE / PDE / DAE などの問題メタデータを表す
- `L(f)` の形で関数を取り残差関数を返す微分作用素 Protocol を表す
- 1 問題 1 ファイルで、`jax_util.differential_equations.<problem_name>` から直接 import できるようにする
- 後続の solver / benchmark / neuralnetwork 実験から参照しやすい置き場を用意する

この段階では、個別問題を無秩序に増やすのではなく、
論文や benchmark で繰り返し使う問題から整理します。
まずは import-safe かつ autodiff-friendly な problem catalog の土台と
starter modules を用意します。

## Public API

- `StateFunction` / `ResidualFunction`
  - 未知関数 `f` と残差関数 `L(f)` の最小 callable 契約です。
- `DifferentialEquationOperator`
  - 関数を引き数に取り、残差関数を返す演算子契約です。
- `DifferentialEquationTerm`
  - 演算子と tag を束ねる最小単位です。
  - `equation` tag を含む場合は `L(f)=0` を暗黙に仮定します。
- `DifferentialEquationProblem`
  - 1 つの問題の名前、方程式種別、条件種別、次元、時間区間、tag、term を持つ不変メタデータです。
- `EquationKind` / `ConditionKind`
  - catalog を絞り込むための最小語彙です。
- `<problem_name>.py`
  - 1 問題 1 ファイルの concrete module です。
  - 各ファイルは少なくとも `problem` と代表 term を export します。

## Starter Modules

- `jax_util.differential_equations.lotka_volterra`
  - predator-prey ODE の canonical module です。
- `jax_util.differential_equations.heat_1d_dirichlet`
  - 1 次元 heat equation の canonical module です。

## Example

- `operator(f)` が残差関数 `L(f)` を返す形を標準にします。
- `DifferentialEquationTerm(name="governing_equation", operator=operator, tags=("equation",))` のように tag つきで束ねます。
- `equation` tag が付いた term は `L(f)=0` を解く対象として解釈します。
- 具体的な問題は `jax.grad`、`jax.jvp`、`jax.jacfwd`、`jax.jacrev` などの自動微分で記述します。
- import は `jax_util.differential_equations.<problem_name>.<symbol>` を標準にします。
- 例: `jax_util.differential_equations.lotka_volterra.problem`

## Design Boundary

- この package は metadata と operator protocol の layer に留めます。
- concrete module は 1 問題 1 ファイルの単位で追加します。
- 具体 problem の微分作用素は autodiff で組み、手書きの有限差分へ逃がしません。
- solver 実装、離散化実装、experiment result はここに置きません。
- 詳細設計は `documents/design/jax_util/differential_equations.md` を参照します。
