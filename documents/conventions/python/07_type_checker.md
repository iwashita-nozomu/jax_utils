# 型チェッカの活用

## 要約
- cast ではなく pyright を優先します。
- 型の境界は `base` に集約します。

## 規約
- cast 等のプログラマによる型安全性の確保は避け、pyright による型安全性の確保を優先します。
- 型の境界は `python/jax_util/base/protocols.py` と `python/jax_util/base/linearoperator.py` に集約し、単一の基準で整合を保ちます。
- `pyright: ignore` / `# type: ignore` の使用は避け、型注釈や設計側で解消します。

### 例外（最小限の ignore）
- JAX の制御フロー（例: `jax.lax.cond`）や `jnp.where` を含む式は、型チェッカが追従できず **実装上どうしても** `ignore` が必要になる場合があります。
- その場合は、次の条件をすべて満たす範囲で **最小限**に許可します。
	- まず `jnp.asarray(..., dtype=DEFAULT_DTYPE)` などで **型を正規化**し、`ignore` を不要にできないか試す。
	- それでも解消できない場合のみ、`pyright: ignore` を 1 行単位で付ける。
	- `ignore` の直前に「なぜ解消できないか」を **丁寧にコメント**で説明する。
