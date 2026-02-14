# 依存関係（pydeps ベース）

この章は `pydeps` の解析結果（`/workspace/deps.dot`。実体は Graphviz の SVG）を一次情報として、
`python/jax_util/` 配下の依存関係を整理します。

## 方針

- 依存の向きは **下位（基盤）→ 上位（機能）** を原則とします。
  - ここでの「下位」は `base`、上位は `solvers` / `optimizers` / `neuralnetwork` / `hlo` です。
- 依存は極力シンプルに保ち、循環依存を作りません。
- この表は「import（依存）の存在」を示します。
  - “呼び出す/使う” とは別に、import している限り依存とみなします。

## レイヤ別の概要

- `jax_util.base`:
  - 基盤です。他レイヤから参照されます。
- `jax_util.solvers`:
  - `base` を利用して数値ソルバを提供し、必要に応じて `optimizers` にも提供します。
- `jax_util.optimizers`:
  - `base` と `solvers` を利用して最適化アルゴリズムを提供します。
- `jax_util.neuralnetwork`:
  - `base` を利用して NN 関連ユーティリティ/学習を提供します。
- `jax_util.hlo`:
  - `base` を利用して HLO ダンプなどを提供します。

## 内部依存（jax_util 内）

`pydeps` の結果から、`jax_util` 内部の依存（`src -> dst`）は概ね次の通りです。

### base（コア）

- `jax_util.base_protocols` -> `jax_util.base`
- `jax_util.base_protocols` -> `jax_util.base__env_value`
- `jax_util.base_protocols` -> `jax_util.base_LinearOperator`
- `jax_util.base_protocols` -> `jax_util.base_linearoperator`
- `jax_util.base_protocols` -> `jax_util.base_nonlinearoperator`
- `jax_util.base__env_value` -> `jax_util.base`
- `jax_util.base_linearoperator` -> `jax_util.base`
- `jax_util.base_linearoperator` -> `jax_util.base_nonlinearoperator`
- `jax_util.base_nonlinearoperator` -> `jax_util.base`

補足:

- ノード名の `_` は、ファイルパスの `.` や `/` が Graphviz 表現に変換されている可能性があります。
  - 例: `jax_util_base__env_value` は `jax_util.base._env_value` に対応します。

### solvers

- `jax_util.solver` パッケージの集約:
  - `jax_util_solvers__check_mv_operator` -> `jax_util_solvers`
  - `jax_util_solvers__minres` -> `jax_util_solvers`
  - `jax_util_solvers__test_jax` -> `jax_util_solvers`
  - `jax_util_solvers_kkt_solver` -> `jax_util_solvers`
  - `jax_util_solvers_lobpcg` -> `jax_util_solvers`
  - `jax_util_solvers_matrix_util` -> `jax_util_solvers`
  - `jax_util_solvers_pcg` -> `jax_util_solvers`

- solvers 内の主な依存:
  - `jax_util_solvers__minres` -> `jax_util_solvers_kkt_solver`
  - `jax_util_solvers__check_mv_operator` -> `jax_util_solvers_kkt_solver`
  - `jax_util_solvers_lobpcg` -> `jax_util_solvers_kkt_solver`
  - `jax_util_solvers_matrix_util` -> `jax_util_solvers_lobpcg`

- base から solvers への参照（基盤から見た依存）:
  - `jax_util_base` -> `jax_util_solvers__check_mv_operator`
  - `jax_util_base` -> `jax_util_solvers__minres`
  - `jax_util_base` -> `jax_util_solvers__test_jax`
  - `jax_util_base` -> `jax_util_solvers_kkt_solver`
  - `jax_util_base` -> `jax_util_solvers_lobpcg`
  - `jax_util_base` -> `jax_util_solvers_matrix_util`
  - `jax_util_base` -> `jax_util_solvers_pcg`
  - `jax_util_base` -> `jax_util_solvers_slq`

### optimizers

- `jax_util_optimizers_protocols` -> `jax_util_optimizers`
- `jax_util_optimizers_protocols` -> `jax_util_optimizers_pdipm`
- `jax_util_optimizers_pdipm` -> `jax_util_optimizers`

補足:

- `pydeps` の内部表現では、`jax_util_optimizers_pdipm` のように module 名がフラット化されます。

### neuralnetwork

- `jax_util_neuralnetwork_layer_utils` -> `jax_util_neuralnetwork_neuralnetwork`
- `jax_util_neuralnetwork_protocols` -> `jax_util_neuralnetwork_neuralnetwork`
- `jax_util_neuralnetwork_protocols` -> `jax_util_neuralnetwork_train`
- `jax_util_neuralnetwork_train` -> `jax_util_neuralnetwork`
- `jax_util_neuralnetwork_neuralnetwork` -> `jax_util_neuralnetwork`

### hlo

- `jax_util_hlo_dump` -> `jax_util_hlo`

## レイヤ制約（守ること）

- `base` は `solvers` / `optimizers` / `neuralnetwork` / `hlo` に依存しない。
- `solvers` は `optimizers` を import しない（循環依存の回避）。
- `optimizers` は `base` と `solvers` を利用してよい。
- `neuralnetwork` は `base` を利用してよい（`solvers`/`optimizers` への依存は原則避ける）。

## 更新手順

- 依存関係を更新した場合は、次を更新します。
  - `deps.dot`（再生成）
  - この文書（差分を反映）

再生成（例）:

- `pydeps ./python/jax_util/ -o deps.dot`
- `./scripts/extract_deps_from_svg.sh deps.dot --internal --prefix jax_util > deps_internal.txt`
