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

### 2.1 実務上の使い分け

- `Scalar`
  - 単一の実数値や停止判定に使います。
- `Vector`
  - 1 本の状態ベクトル、残差ベクトル、探索方向に使います。
- `Matrix`
  - 行列そのものだけでなく、列方向へ並べたベクトルバッチにも使います。
- `Boolean` / `Integer`
  - 0 次元のフラグやカウンタに使います。

### 2.2 最小例

- `Scalar`: `jnp.asarray(1.0)`
- `Vector`: `jnp.asarray([1.0, 2.0])`
- `Matrix`: `jnp.asarray([[1.0, 0.0], [0.0, 1.0]])`

### 2.3 運用ルール

- 公開 API の引数・戻り値には、できるだけ `Scalar` / `Vector` / `Matrix` / `Boolean` を使います。
- 意味が曖昧な `Array` 単独の注釈は避けます。
- `Matrix` をバッチとして扱う場合は、列バッチかどうかをコメントで明示します。
- `Vector` と `Matrix` のどちらかで迷うときは、入出力の次元を優先して選びます。

## 3. 作用素 Protocol

- `Operator`: 非線形作用素。適用は `()`、合成は `*`
- `LinearOperator`: 線形作用素。適用は `@`、合成は `*`
- `SolverLike`: ソルバーの共通呼び出し契約
- `ScalarFn` / `VectorFn`: 最適化や制約で使う関数契約

## 3.1 最適化 Protocol

- `OptimizationProblem[T]`
  - 変数 `T` を取りスカラー値を返す目的関数を持つ最小契約です。
- `ConstrainedOptimizationProblem[T, EqT, IneqT]`
  - 等式制約と不等式制約を追加した汎用契約です。
- `OptimizationState[T]`
  - 現在の変数 `x` を保持する状態契約です。
- `ConstrainedOptimizationState[T, DualT]`
  - 双対変数と slack を追加した状態契約です。

### 3.2 命名の分担

- 汎用契約は `base/protocols.py` に置き、空間名を付けません。
- 空間ごとの特殊化は、各サブモジュールの `protocols.py` で `Vector` / `PyTree` / `Functional` を先頭に付けて命名します。
  - `VectorOptimizationProblem`
  - `PyTreeOptimizationProblem`
  - `FunctionalOptimizationProblem`
- 制約付きも同様に `Constrained*` を先頭に付けます。
- `WithConstraint` や `OptimizeProblem` のような旧系命名は、可読性と検索性を損なうため使いません。

## 4. `LinOp`

- 単一ベクトル向けの `mv: Vector -> Vector` を受け取り、内部で `Vector` / `Matrix` の両方に対応します。
- 線形作用素の適用は `@`、合成は `*` で統一します。
- 安定サブモジュール側は、原則として `LinOp` または `LinearOperator` を介して線形作用素を扱います。

### 4.1 型エイリアスとの関係

- `LinOp` の基本契約は `Vector -> Vector` です。
- 呼び出し側が `Matrix` を渡すときは、列バッチとして解釈されることを前提にします。
- したがって、`Matrix` は「2 次元行列」と「ベクトル列のバッチ」の両方を表す共通型として扱います。

## 5. 微分作用素

- `linearize(f, x0)`: `f(x0)` と Jacobian の線形作用素を返します。
- `adjoint(f, x0)`: `f(x0)` と Jacobian の随伴作用素を返します。
- `solvers` / `optimizers` は、明示行列ではなくこれらの作用素 API を優先して使います。

## 6. 環境設定

- `DEFAULT_DTYPE`, `EPS`, `WEAK_EPS`, `AVOID_ZERO_DIV`, `DEBUG`, `ENABLE_HLO_DUMP` は `base/_env_value.py` に集約します。
- 環境変数は `JAX_UTIL_` プレフィックスで統一します。
- 既定値はコード側で保持し、環境変数は上書き専用とします。

## 7. 追記方針

- 新しい型ルールや実務上の判断は、原則としてこの文書へ追加します。
- Python 側の型注釈規約は `documents/conventions/python/02_type_aliases.md` と役割分担して保ちます。
