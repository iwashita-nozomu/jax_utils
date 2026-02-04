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
- ただし **`python/jax_util/base/` 内部実装** と **`python/jax_util/base/protocols.py`** では、
	型境界の定義として `Array` を使ってもよいものとします。
- `Scalar` / `Vector` / `Matrix` / `Boolean` のいずれかで意味を必ず明示します。
- `Matrix` は **バッチ化されたベクトル列**も含みます。
- バッチ軸（`axis=-1`）などの共通定義は `documents/coding-conventions-project.md` の **5.2** に集約します。
- `neuralnetwork` 固有の型（例: `Params`）は `python/jax_util/neuralnetwork/protocols.py` に置きます。
- `LinOp` はバッチ対応を明示するため、`Matrix` へ作用できる設計とします。
