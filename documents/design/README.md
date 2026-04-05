# 設計ドキュメント

このディレクトリは、設計の正本を集約する入口です。
存在しない旧パスを経由せず、ここから現在の設計文書へ直接辿れる状態を保ちます。

## 現在の正本

- [protocols.md](/workspace/documents/design/protocols.md)
  - Protocol 層の責務分割
  - 型パラメータ化の方針
- [jax_util/README.md](/workspace/documents/design/jax_util/README.md)
  - `jax_util` 配下のモジュール設計入口
- [experiment_runner.md](/workspace/documents/experiment_runner.md)
  - `experiment_runner` の契約と実行モデル

## `jax_util` 詳細設計

- [base_components.md](/workspace/documents/design/jax_util/base_components.md)
- [solvers.md](/workspace/documents/design/jax_util/solvers.md)
- [optimizers.md](/workspace/documents/design/jax_util/optimizers.md)
- [hlo.md](/workspace/documents/design/jax_util/hlo.md)
- [lagrangian_df_zero.md](/workspace/documents/design/jax_util/lagrangian_df_zero.md)

## 更新ルール

- base の型・Protocol を変えた場合は [base_components.md](/workspace/documents/design/jax_util/base_components.md) と [protocols.md](/workspace/documents/design/protocols.md) を更新します。
- `solvers` / `optimizers` / `hlo` の公開 API や責務を変えた場合は、対応する `documents/design/jax_util/*.md` を更新します。
- `experiment_runner` の契約を変えた場合は [experiment_runner.md](/workspace/documents/experiment_runner.md) を更新します。

## 禁止事項

- `documents/design/base_components.md` のような存在しない旧パスを正本として参照することを禁止します。
- `documents/design/apis/` のような削除済み階層を案内することを禁止します。
- 設計の正本を `notes/` や `reports/` に置くことを禁止します。
