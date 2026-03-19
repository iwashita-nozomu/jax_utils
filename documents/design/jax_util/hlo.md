# `hlo` API 詳細設計

この文書は、`hlo` の安定 API の役割と依存関係を整理します。

## 1. 対象

- `python/jax_util/hlo/` の安定 API を対象にします。

## 2. 公開 API の位置づけ

- `dump_hlo_jsonl`: JAX lowering から HLO を取得し、JSONL に追記するユーティリティ

## 3. 共通設計

- HLO 取得は重いため、`JAX_UTIL_ENABLE_HLO_DUMP` で明示的に有効化します。
- 出力は JSONL（1 行 1 JSON）に統一します。
- `tag` により解析対象を識別できるようにします。

## 4. 依存関係

- `hlo` は `base` の環境設定だけを利用します。
- `hlo` から `solvers` / `optimizers` へ依存しません。
