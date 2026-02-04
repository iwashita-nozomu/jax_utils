# 依存関係の概要

## 要約
- `base` を基盤として `Algorithms` が依存します。
- KKT/PDIPM は複数モジュールを統合します。

## 規約
| 対象 | 依存 | 目的 |
| --- | --- | --- |
| `python/jax_util/base/__init__.py` | `base/protocols.py`, `base/linearoperator.py`, `base/_env_value.py` | 型と定数の集約エクスポート。 |
| `python/jax_util/base/linearoperator.py` | `base/protocols.py` | `LinOp` の基盤型を共有する。 |
| `python/jax_util/Algorithms/_check_mv_operator.py` | `base` | 作用素チェック用の型・定数を利用する。 |
| `python/jax_util/Algorithms/_fgmres.py` | `base` | **アーカイブ済み**（現在は未使用）。 |
| `python/jax_util/Algorithms/_minres.py` | `base` | 反復解法の基盤型・定数を利用する。 |
| `python/jax_util/Algorithms/pcg.py` | `base` | 反復解法の基盤型・定数を利用する。 |
| `python/jax_util/Algorithms/lobpcg.py` | `base`, `Algorithms/matrix_util.py` | 固有値計算の補助を利用する。 |
| `python/jax_util/Algorithms/kkt_solver.py` | `base`, `Algorithms/_check_mv_operator.py`, `Algorithms/_minres.py`, `Algorithms/lobpcg.py` | KKT 解法で各モジュールを統合する。 |
| `python/jax_util/Algorithms/pdipm.py` | `base/_env_value.py`, `Algorithms/kkt_solver.py` | 内点法で KKT 解法と定数を利用する。 |
| `python/jax_util/neuralnetwork/protocols.py` | `base` | 型エイリアスを利用してプロトコルを定義する。 |
| `python/jax_util/neuralnetwork/layer_utils.py` | `base`, `neuralnetwork/protocols.py` | 層の実装と初期化に基盤型を利用する。 |
| `python/jax_util/neuralnetwork/neuralnetwork.py` | `base`, `neuralnetwork/protocols.py`, `neuralnetwork/layer_utils.py` | ネットワークの構成と forward を提供する。 |
| `python/jax_util/neuralnetwork/train.py` | `base`, `neuralnetwork/protocols.py` | 学習ループで共通型と損失プロトコルを利用する。optax を前提とする。 |
