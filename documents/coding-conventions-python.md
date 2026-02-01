## Python コーディング規約（型注釈）

### 1. 型エイリアスの方針
- 型エイリアスは `_type_aliaces.py` に集約します。
- 配列の意味は以下の通りに統一します。
	- `Scalar`: 0 次元（`Shaped[Array, ""]`）
	- `Vector`: 1 次元（`Shaped[Array, "n"]`）
	- `Matrix`: 2 次元（`Shaped[Array, "m n"]`）
	- `Batch[T]`: 任意の型 `T` のバッチ
- `Array` は型エイリアス定義内でのみ使用し、関数の引数・戻り値では使いません。

### 2. 射（関数）と継承関係
- 射は `Hom` を基準に表現します。
- `Map` は `Vector -> Vector` を表す射です。
- `BMap` は `Batch[Vector] -> Batch[Vector]` を表す射で、`Map` を継承します。
- `LinearMap` は線形性を仮定した射で、`Map` を継承します。
- `BLinearMap` は `BMap` と `LinearMap` を継承します。
- 同じ対象を表す冗長な型は作らず、継承によって依存関係を明示します。

### 3. 関数の型注釈
- 引数・戻り値の型は必ず `Scalar` / `Vector` / `Matrix` / `Batch[Vector]` のいずれかで明示します。
- `Batch[Vector]` を使うことでバッチ対応を明示します。
- `Array` のみの注釈は避けてください。

### 4. バッチ対応
- バッチ対応は `ensure_batch` を使って明示します。
- `Batch[T]` でバッチ次元の存在を表現します。

### 5. コメント
- コメントは丁寧に書き、意図と前提を明確にします。
- 実装の重複や複雑な分岐を避け、シンプルな記述を心がけます。

### 6. 型チェッカの活用
- cast等のプログラマによる型安全性の確保は避け、pyrightによる型安全性の確保を
- 型変換関数は_type_aliaces.py内で書き、型安全性は単一ファイル＋pyrightで保証する

### 7.合成型の定義
- 合成型の定義は、_type_mixin.pyにすべて定義
- TypeGuardも定義します