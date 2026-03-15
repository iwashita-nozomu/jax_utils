# `solvers` API 詳細設計

この文書は、`solvers` の安定 API の役割と依存関係を整理します。

## 1. 対象

- `python/jax_util/solvers/` の安定 API を対象にします。
- `python/jax_util/solvers/archive/` は対象外です。

## 2. 公開 API の位置づけ

- `pcg.py`: SPD 系向けの前処理付き共役勾配法
- `_minres.py`: 対称系向けの MINRES
- `lobpcg.py`: 固有値推定とランク制限スペクトル前処理
- `kkt_solver.py`: KKT ブロックソルバ
- `slq.py`: Stochastic Lanczos Quadrature
- `_check_mv_operator.py`: 作用素検査ユーティリティ
- `matrix_util.py`: 直交化などの補助

## 3. 共通設計

- 線形作用素は `LinearOperator` / `LinOp` で受け渡します。
- 戻り値は将来的に `ans, state, info` の形へ揃えます。
- `state` は `eqx.Module`、`info` は辞書を基本にします。
- 悪条件ケースでは `res_norm` と `rel_res` の両方を残します。

## 4. 依存関係

- `solvers` は `base` に依存します。
- `kkt_solver.py` は `_minres.py` と `lobpcg.py` を利用します。
- `solvers` から `optimizers` へは依存しません。

## 5. 非対象

- FGMRES などの退避実装は `archive/` に置き、安定 API 設計には含めません。
