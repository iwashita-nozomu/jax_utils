# 作用素と継承関係

この章は、作用素の表現と継承関係の基準をまとめます。

## 要約

- `Operator` / `LinearOperator` を基準にします。
- 合成は `__mul__` / `__matmul__` を使います。

## 規約

- 作用素は `python/jax_util/base/protocols.py` の `Operator` / `LinearOperator` を基準に表現します。
- 作用素の合成は `__mul__` / `__matmul__` を通して表します。
- 同じ対象を表す冗長な型は作らず、継承によって依存関係を明示します。
- `Operator` は `Callable[[Matrix], Matrix]` として定義されていますが、現状の実装では積極的に意識しません。
