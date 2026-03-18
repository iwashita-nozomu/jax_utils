# 型チェッカの活用

この章は、pyright を中心とした型検査の運用方針を定めます。

## 要約

- cast ではなく pyright を優先します。
- 型の境界は `base` に集約します。

## 規約

- `pyright` の設定は `pyproject.toml` の `[tool.pyright]` を正本とし、並行した `pyrightconfig.json` は持ちません。
- 既定の `pyright` 実行は repo root で行い、`pyproject.toml` に含めた対象だけを baseline として常時 clean に保ちます。
- 現在の baseline は `python/jax_util/` のうち `neuralnetwork` と `solvers/archive` を除く実装です。
- `python/tests/` や `python/jax_util/neuralnetwork/` を触った場合は、既定実行とは別に対象 path を明示して `pyright` を追加実行します。
- 実験段階やテストで `pyright` エラーが残る場合は、黙って放置せず `task.md`、`reviews/`、または関連 note に未解消として残します。
- cast 等のプログラマによる型安全性の確保は避け、pyright による型安全性の確保を優先します。
- 型の境界は `python/jax_util/base/protocols.py` と `python/jax_util/base/linearoperator.py` に集約し、単一の基準で整合を保ちます。
- `pyright: ignore` / `# type: ignore` の使用は避け、型注釈や設計側で解消します。

## 実行例

- baseline 全体: `pyright`
- 実験段階の NN を触ったとき: `pyright python/jax_util/neuralnetwork`
- テストを触ったとき: `pyright python/tests/<subdir-or-file>`
- 特定モジュールだけを確認したいとき: `pyright python/jax_util/<module>`

### 例外（最小限の ignore）

- JAX の制御フロー（例: `jax.lax.cond`）や `jnp.where` を含む式は、型チェッカが追従できず **実装上どうしても** `ignore` が必要になる場合があります。
- その場合は、次の条件をすべて満たす範囲で **最小限**に許可します。
  - まず `jnp.asarray(..., dtype=DEFAULT_DTYPE)` などで **型を正規化**し、`ignore` を不要にできないか試す。
  - それでも解消できない場合のみ、`pyright: ignore` を 1 行単位で付ける。
  - `ignore` の直前に「なぜ解消できないか」を **丁寧にコメント**で説明する。
