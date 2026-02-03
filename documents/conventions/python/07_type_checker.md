# 型チェッカの活用

## 要約
- cast ではなく pyright を優先します。
- 型の境界は `base` に集約します。

## 規約
- cast 等のプログラマによる型安全性の確保は避け、pyright による型安全性の確保を優先します。
- 型の境界は `python/jax_util/base/protocols.py` と `python/jax_util/base/linearoperator.py` に集約し、単一の基準で整合を保ちます。
