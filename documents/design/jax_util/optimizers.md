# `optimizers` API 詳細設計

この文書は、`optimizers` の安定 API の役割と依存関係を整理します。

## 1. 対象

- `python/jax_util/optimizers/` の安定 API を対象にします。

## 2. 公開 API の位置づけ

- `protocols.py`: 最適化問題と状態の契約
- `pdipm.py`: 制約付き最適化向け primal-dual interior point method

## 3. 共通設計

- 目的関数・制約は `ScalarFn` / `VectorFn` 系の契約で受け渡します。
- `PDIPMState` は KKT ソルバ状態と primal / dual 変数をまとめます。
- `initialize_pdipm_state` はウォームスタート可能な初期状態を返します。
- `pdipm_solve` は Newton 方向計算に `kkt_solver.py` を利用します。

## 4. 依存関係

- このサブモジュールの依存関係は概ね `base` と `solvers` です。詳細な方針は [documents/design/jax_util/README.md](README.md) を参照してください。
