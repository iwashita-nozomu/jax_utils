# JAX/Equinox の運用規約

## 要約
- 反復は `jax.lax.while_loop` を使います。
- JIT 文脈での Python 変換を避けます。

## 規約
- 反復は `jax.lax.while_loop` を使い、Python の `if/for` に依存しません。
- JIT 文脈での `bool/int/float` 変換は避け、**JAX 配列のまま扱う**ことを優先します。
- デバッグ出力は `DEBUG` ガードと `jax.debug.print` を使います。
