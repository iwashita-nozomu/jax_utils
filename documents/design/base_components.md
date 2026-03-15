# Base Components 設計

この文書は、`python/jax_util/base/` にある共通基盤の設計をまとめます。

## 1. 対象

- 型エイリアス
- 作用素 Protocol
- `LinOp`
- `linearize` / `adjoint`
- 環境設定と数値定数

## 2. 型エイリアス

- `Scalar`: 0 次元スカラー
- `Vector`: 1 次元ベクトル
- `Matrix`: 2 次元配列。`(n, batch)` の列バッチを含む
- `Boolean` / `Integer`: 0 次元の論理値 / 整数値

補足は `documents/type-aliases.md` に集約します。

## 3. 作用素 Protocol

- `Operator`: 非線形作用素。適用は `()`、合成は `*`
- `LinearOperator`: 線形作用素。適用は `@`、合成は `*`
- `SolverLike`: ソルバーの共通呼び出し契約
- `ScalarFn` / `VectorFn`: 最適化や制約で使う関数契約

## 4. `LinOp`

- 単一ベクトル向けの `mv: Vector -> Vector` を受け取り、内部で `Vector` / `Matrix` の両方に対応します。
- 線形作用素の適用は `@`、合成は `*` で統一します。
- 安定サブモジュール側は、原則として `LinOp` または `LinearOperator` を介して線形作用素を扱います。

## 5. 微分作用素

- `linearize(f, x0)`: `f(x0)` と Jacobian の線形作用素を返します。
- `adjoint(f, x0)`: `f(x0)` と Jacobian の随伴作用素を返します。
- `solvers` / `optimizers` は、明示行列ではなくこれらの作用素 API を優先して使います。

## 6. 環境設定

- `DEFAULT_DTYPE`, `EPS`, `WEAK_EPS`, `AVOID_ZERO_DIV`, `DEBUG`, `ENABLE_HLO_DUMP` は `base/_env_value.py` に集約します。
- 環境変数は `JAX_UTIL_` プレフィックスで統一します。
- 既定値はコード側で保持し、環境変数は上書き専用とします。
