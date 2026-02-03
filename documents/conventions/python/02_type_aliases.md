# 型エイリアスの方針

## 要約
- 型は `protocols.py` に集約します。
- `Scalar` / `Vector` / `Matrix` / `Boolean` を使います。

## 規約
- 型エイリアスは `python/jax_util/base/protocols.py` に集約します。
- 配列の意味は以下の通りに統一します。
	- `Scalar`: 0 次元（`Float[Array, ""]`）
	- `Vector`: 1 次元（`Float[Array, "n"]`）
	- `Matrix`: 2 次元（`Float[Array, "m n"]`）
	- `Boolean`: 0 次元（`Bool[Array, ""]`）
- `Array` は型エイリアス定義内でのみ使用し、関数の引数・戻り値では使いません。
- `Scalar` / `Vector` / `Matrix` / `Boolean` のいずれかで意味を必ず明示します。
