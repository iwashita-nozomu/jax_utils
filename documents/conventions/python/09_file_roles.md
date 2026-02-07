# ファイルごとの役割

## 要約
- `base` と `Algorithms` を明確に分離します。
- 各ファイルの役割を表で示します。

## 規約
| ファイル | 役割 |
| --- | --- |
| `python/jax_util/base/protocols.py` | 型エイリアスと作用素プロトコルの定義。`Scalar` / `Vector` / `Matrix` / `Boolean` / `Integer` を集約し、`Operator` / `LinearOperator` の契約を示す。 |
| `python/jax_util/base/linearoperator.py` | `LinOp` 実装。`Vector`/`Matrix` を受け取り 1D/2D を内部で分岐する。合成（`__mul__` / `__matmul__`）の基本挙動を提供する。 |
| `python/jax_util/base/nonlinearoperator.py` | 現状は **微分作用素（`linearize`）と随伴（`adjoint`）のみ**を提供する。`Operator`（`Callable[[Matrix], Matrix]`）は `base/protocols.py` に存在するが、現状は明示的に意識しない。 |
| `python/jax_util/base/_env_value.py` | 既定の `dtype` と数値定数（`EPS` など）を提供する。**JAX 実行環境の設定**もここに集約する。環境変数（`JAX_UTIL_ENABLE_X64` / `JAX_UTIL_DEFAULT_DTYPE` / `JAX_UTIL_DEBUG` / `JAX_UTIL_WEAK_EPS` / `JAX_UTIL_EPS` / `JAX_UTIL_AVOID_ZERO_DIV` / `JAX_UTIL_ENABLE_HLO_DUMP`）で上書きを許可する。 |
| `python/jax_util/base/__init__.py` | `base` 系の型・定数・`LinOp` を集約してエクスポートする窓口。利用側は原則ここから import する。 |
| `python/jax_util/solvers/` | 反復解法・KKT・LOBPCG などのソルバ群をまとめるサブパッケージ。 |
| `python/jax_util/solvers/_check_mv_operator.py` | 作用素の自己随伴性・SPD 性の簡易レポートを返すユーティリティ。安全確認の最小チェックに使う。 |
| `python/jax_util/solvers/_fgmres.py` | **アーカイブ済み**（現在は未使用）。必要になった場合のみ再導入する。 |
| `python/jax_util/solvers/_minres.py` | MINRES の実装。前処理付き反復解法を提供し、数値安定性の補助処理を含む。 |
| `python/jax_util/solvers/pcg.py` | PCG（前処理付き共役勾配法）の実装。投影付きの反復構成と収束判定を行う。 |
| `python/jax_util/solvers/kkt_solver.py` | KKT 系のブロックソルバ。`_minres.py` / `_fgmres.py` / `lobpcg.py` を組み合わせ、前処理を含めた解法を提供する。 |
| `python/jax_util/solvers/lobpcg.py` | LOBPCG 系の固有値推定。ブロック固有値更新、部分空間更新、前処理構築を扱う。 |
| `python/jax_util/solvers/matrix_util.py` | 行列ユーティリティ（直交化など）。小さく単機能の補助関数を置く。 |
| `python/jax_util/solvers/_test_jax.py` | JAX の動作確認と最小の疎通チェック。環境の健全性を確認する。 |
| `python/jax_util/optimizers/` | 最適化アルゴリズム群（例: PDIPM）をまとめるサブパッケージ。 |
| `python/jax_util/optimizers/pdipm.py` | 内点法（PDIPM）の実装。目的関数・制約の構成、KKT 解法の呼び出し、簡易な問題生成を含む。 |
| `python/jax_util/optimizers/protocols.py` | 最適化問題の Protocol（目的関数・制約・状態など）を定義する。 |
| `python/jax_util/optimizers/__init__.py` | 最適化系の公開 API を集約する。 |
| `python/jax_util/neuralnetwork/protocols.py` | ニューラルネットワークの最小プロトコル（`Ctx` / `Carry` / `NeuralNetworkLayer`）を定義する。 |
| `python/jax_util/neuralnetwork/layer_utils.py` | 標準 NN と ICNN の層・初期化関数を提供する。 |
| `python/jax_util/neuralnetwork/neuralnetwork.py` | ネットワーク本体と forward、ファクトリ関数をまとめる。 |
| `python/jax_util/neuralnetwork/__init__.py` | ニューラルネットワーク系の公開 API を集約する。 |
| `python/jax_util/neuralnetwork/train.py` | 学習ループの最小実装を提供する。 |
| `python/jax_util/__init__.py` | パッケージのメタ情報（`__version__`）を保持する。 |
| `documents/type-aliases.md` | 型エイリアスの運用指針と一覧を整理する。 |
