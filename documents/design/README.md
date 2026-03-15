# 設計書

このディレクトリは、安定サブモジュールの設計書を置く場所です。
規約本文とは分けて、追記しやすい単位で整理します。

## 構成

- `base_components.md`: base の型・クラス・Protocol・環境設定
- `apis/solvers.md`: `solvers` の API 詳細設計
- `apis/optimizers.md`: `optimizers` の API 詳細設計
- `apis/hlo.md`: `hlo` の API 詳細設計

## 追記ルール

- base の共通概念を増やすときは `base_components.md` を更新します。
- 安定サブモジュールの API を増やすときは `apis/<module>.md` を更新します。
- 実験段階の設計は、このディレクトリへ混ぜずに別文書として扱います。
