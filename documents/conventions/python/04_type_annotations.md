# 関数の型注釈

## 要約
- 引数・戻り値は `Scalar` / `Vector` / `Matrix` を使います。
- `Array` 単独の注釈は避けます。

## 規約
- 引数・戻り値の型は必ず `Scalar` / `Vector` / `Matrix` のいずれかで明示します。
- `Array` のみの注釈は避けてください。
- 複数の意味を持つ引数は、`Vector` と `Matrix` を分けて表現します。
