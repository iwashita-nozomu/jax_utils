# 関数の型注釈

## 要約
- 引数・戻り値は `Scalar` / `Vector` / `Matrix` を使います。
- `Array` 単独の注釈は避けます。

## 規約
- 引数・戻り値の型は必ず `Scalar` / `Vector` / `Matrix` のいずれかで明示します。
- `Array` のみの注釈は避けてください。
- 複数の意味を持つ引数は、`Vector` と `Matrix` を分けて表現します。

## dtype の受け渡し
JAX の数値計算では dtype が結果の安定性・再現性に直結します。

- **公開API**（`__all__` で公開する関数）は、**必ず `dtype` 引数を持ち**、その既定値を `DEFAULT_DTYPE` とします。
	- 例: `def solve(..., dtype: DTypeLike = DEFAULT_DTYPE) -> ...: ...`
- **内部実装関数**（先頭が `_` の関数）は、**`dtype` を必ず引数で受け取り**、`DEFAULT_DTYPE` を既定引数にしません。
	- 例: `def _step(..., dtype: DTypeLike, ...) -> ...: ...`
	- 必要な dtype は **公開APIから必ず引数で渡します**（内部で環境変数や `DEFAULT_DTYPE` を参照しません）。
	- 目的は「暗黙の dtype 決め打ち」を避け、呼び出し元で一括制御できるようにすることです。
