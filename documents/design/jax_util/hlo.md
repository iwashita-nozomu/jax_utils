# `hlo` API 詳細設計

この文書は、`hlo` の安定 API の役割と依存関係を整理します。

## 1. 対象

- `python/jax_util/hlo/` の安定 API を対象にします。

## 2. 公開 API の位置づけ

- `jax_util.hlo.dump`: 0 引数 callable の JAX lowering から HLO を取得し、JSONL に追記する正面 API
- `jax_util.hlo.dump_hlo_jsonl`: 引数付き lowering を直接扱いたい場合の互換 API

## 3. 共通設計

- HLO 取得は重いため、`JAX_UTIL_ENABLE_HLO_DUMP` で明示的に有効化します。
- 出力は JSONL（1 行 1 JSON）に統一します。
- `from jax_util.hlo import dump` として `dump(f, path)` を標準の呼び方とし、引数付き関数は `lambda: f(x)` で閉じ込めます。
- `tag` により解析対象を識別できるようにします。省略時は関数名から自動生成します。

## 4. 依存関係

- `hlo` は `base` の環境設定だけを利用します。
- `hlo` から `solvers` / `optimizers` へ依存しません。
