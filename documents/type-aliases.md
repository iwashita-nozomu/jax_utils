# 型エイリアス一覧（base/protocols.py）

このドキュメントは `python/jax_util/base/protocols.py` の型エイリアスと、
それらを使うときの実務的な注意点をまとめます。

## 基本型
- `Scalar`: 0 次元スカラー（`Float[Array, ""]`）
- `Vector`: 1 次元ベクトル（`Float[Array, "n"]`）
- `Matrix`: 2 次元行列（`Float[Array, "m n"]`）
- `Boolean`: 0 次元ブール（`Bool[Array, ""]`）
- `Integer`: 0 次元整数（`Int[Array, ""]`）

## 使い分け
- `Vector`: 1 つのベクトルを表します。
- `Matrix`: 2 次元配列で、行列またはバッチ化されたベクトル列を表します。
- `Scalar`: 0 次元のスカラー量を表します。
- `Boolean`: 0 次元の論理値を表します。
- `Integer`: 0 次元の整数値を表します。

### 例（最小）
- `Scalar`: `jnp.asarray(1.0)`
- `Vector`: `jnp.asarray([1.0, 2.0])`
- `Matrix`: `jnp.asarray([[1.0, 0.0], [0.0, 1.0]])`

## 運用指針
- 関数の引数・戻り値には必ず `Scalar` / `Vector` / `Matrix` / `Boolean` を使います。
- `Array` を直接注釈に使うのは避けます。
- 作用素の契約は `Operator` / `LinearOperator` を使用し、演算子（`__mul__`, `__matmul__`）で合成します。
- `Matrix` をバッチとして扱う場合は、コメントで意図を明示します。
- `Vector` と `Matrix` のどちらかで迷う場合は、入出力の次元を優先して選びます。

## LinOp との関係
- `python/jax_util/base/linearoperator.py` の `LinOp` は `Vector` / `Matrix` を受け取ります。
- 1D/2D を内部で分岐するため、呼び出し側は `Vector` と `Matrix` のみを意識します。

### `LinOp` での典型的な使い方
- `LinOp(lambda v: A @ v)` のように、`Vector` を受ける関数を包みます。
- 返り値は `Vector` を想定します。バッチ入力は `Matrix` として処理されます。

## プロトコルの位置づけ
- `Operator` は一般作用素の合成 (`__mul__`) と適用 (`__call__`) を表します。
- `LinearOperator` は線形作用素の合成 (`__mul__`) と適用 (`__matmul__`) を表します。
- 実装側は、これらのプロトコルに沿った演算子を提供します。

## 参照元
- 定義: `python/jax_util/base/protocols.py`
- 運用指針: `documents/coding-conventions-python.md`
