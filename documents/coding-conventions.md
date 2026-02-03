# コーディング規約（共通）

この文書は共通規約の**目次**です。
詳細は各章および領域別の規約に分割しています。

## 共通規約（章ごとの要約）
1. [基本方針](./conventions/common/01_principles.md) — 読みやすさ・保守性・依存最小を最優先にします。
2. [命名](./conventions/common/02_naming.md) — 役割が伝わる名前を使い、省略を最小限にします。
3. [コメント](./conventions/common/03_comments.md) — 意図と前提を明確にし、数式や安定性の注意を優先します。
4. [演算子記法（共通）](./conventions/common/04_operators.md) — 適用は `@`、合成は `*` を基本にします。
5. [ドキュメント運用](./conventions/common/05_docs.md) — 実装変更に合わせて文書も更新します。

## 言語・領域別の規約（要約）
- [Python 規約](./coding-conventions-python.md) — `python/jax_util/` の型注釈・演算子記法・JAX 運用を整理します。
- [C++ 規約](./coding-conventions-cpp.md) — C++ 実装に関する命名・設計・注意点を整理します。
- [テスト規約](./coding-conventions-testing.md) — テスト配置、ログ出力、実行方法を定めます。
- [ログ規約](./coding-conventions-logging.md) — JSONL ログの形式・出力先・運用手順を定めます。
- [ソルバー規約](./coding-conventions-solvers.md) — 反復法の戻り値・情報出力・安定化を定めます。

## 補助資料
- [型エイリアス一覧](./type-aliases.md) — `Scalar` / `Vector` / `Matrix` などの指針と一覧。
## 5. ドキュメント運用

- 変更が入った場合は、該当する `documents/` 内の文書を同時に更新します。

