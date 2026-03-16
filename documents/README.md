# documents

`documents/` は、次の 3 層で整理します。

## 1. General なコーディング方針

- `documents/coding-conventions.md`
- `documents/coding-conventions-experiments.md`
- `documents/conventions/common/`
- `documents/conventions/python/`
- 対象: 命名、コメント、型注釈、演算子記法、テスト、ログ、JAX 運用、実験環境運用などの共通ルール

## 2. ベースとなるクラス / Protocol / 型の設計

- `documents/design/base_components.md`
- `documents/type-aliases.md`
- 対象: `base` の型エイリアス、作用素 Protocol、`LinOp`、`linearize` / `adjoint`、環境設定

## 3. 安定サブモジュールの API 詳細設計

- `documents/design/README.md`
- 対象: `solvers` / `optimizers` / `hlo`
- 追記単位: サブモジュールごとに `documents/design/apis/<module>.md` を増やします。

## 現在の扱い

- 安定サブモジュール: `base` / `solvers` / `optimizers` / `hlo`
- 実験段階: `neuralnetwork`
- 保管領域: `solvers/archive`

## 追記ルール

- general な規約を足すときは `documents/conventions/common/` または `documents/conventions/python/` に追記します。
- base の型・クラス・Protocol を整理するときは `documents/design/base_components.md` を更新します。
- 安定 API の設計を足すときは `documents/design/apis/` にサブモジュール単位で追記します。
- 実験段階の設計は、安定 API 文書へ混ぜず、独立した補足として扱います。
- 文書は参照の連鎖で読ませず、できるだけそのファイル単体で読めるように保ちます。
