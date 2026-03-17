# 作用素と継承関係

この章は、作用素の表現と継承関係の基準をまとめます。

## 要約

- `Operator` / `LinearOperator` を基準にします。
- Protocol と実装の役割分担を明確にします。

## 規約

- 作用素は `python/jax_util/base/protocols.py` の `Operator` / `LinearOperator` を基準に表現します。
- 同じ対象を表す冗長な型は作らず、継承によって依存関係を明示します。
- `Operator` は `Callable[[Matrix], Matrix]` として定義されていますが、現状の実装では積極的に意識しません。
- `LinOp` は Python 実装で線形作用素を扱う標準の薄い実装とし、`LinearOperator` 契約に沿って使います。
- 演算子記法そのものの厳格ルールは `documents/conventions/python/12_operator_rules.md` を正本とします。
