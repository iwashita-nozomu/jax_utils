## Python コーディング規約（型注釈）

この文書は、`python/jax_util/` 配下の実装を対象に、型注釈と構造の指針をまとめます。

### 1. 型エイリアスの方針
- 型エイリアスは `python/jax_util/base/protocols.py` に集約します。
- 配列の意味は以下の通りに統一します。
	- `Scalar`: 0 次元（`Float[Array, ""]`）
	- `Vector`: 1 次元（`Float[Array, "n"]`）
	- `Matrix`: 2 次元（`Float[Array, "m n"]`）
	- `Boolean`: 0 次元（`Bool[Array, ""]`）
- `Array` は型エイリアス定義内でのみ使用し、関数の引数・戻り値では使いません。
- `Scalar` / `Vector` / `Matrix` / `Boolean` のいずれかで意味を必ず明示します。

### 2. 作用素と継承関係
- 作用素は `python/jax_util/base/protocols.py` の `Operator` / `LinearOperator` を基準に表現します。
- 作用素の合成は `__mul__` / `__matmul__` を通して表します。
- 同じ対象を表す冗長な型は作らず、継承によって依存関係を明示します。

### 3. 関数の型注釈
- 引数・戻り値の型は必ず `Scalar` / `Vector` / `Matrix` のいずれかで明示します。
- `Array` のみの注釈は避けてください。
- 複数の意味を持つ引数は、`Vector` と `Matrix` を分けて表現します。

### 4. バッチ対応
- バッチ対応は `python/jax_util/base/linearoperator.py` の `LinOp` を通して明示します。
- 入力は `Vector` / `Matrix` で受け、内部で 1D/2D の分岐を行います。
- `Matrix` を「ベクトルのバッチ」とみなす場合は、その意図をコメントで明示します。

### 5. コメント
- コメントは丁寧に書き、意図と前提を明確にします。
- 実装の重複や複雑な分岐を避け、シンプルな記述を心がけます。

### 6. 型チェッカの活用
- cast 等のプログラマによる型安全性の確保は避け、pyright による型安全性の確保を優先します。
- 型の境界は `python/jax_util/base/protocols.py` と `python/jax_util/base/linearoperator.py` に集約し、単一の基準で整合を保ちます。

### 7. 合成型の定義
- 合成は `Operator` / `LinearOperator` の演算子として定義します。
- 新しい合成型は、既存のプロトコルに沿って最小限で追加します。

### 8. ファイルごとの役割
| ファイル | 役割 |
| --- | --- |
| `python/jax_util/base/protocols.py` | 型エイリアスと作用素プロトコルの定義。`Scalar` / `Vector` / `Matrix` / `Boolean` / `Integer` を集約し、`Operator` / `LinearOperator` の契約を示す。 |
| `python/jax_util/base/linearoperator.py` | `LinOp` 実装。`Vector`/`Matrix` を受け取り 1D/2D を内部で分岐する。合成（`__mul__` / `__matmul__`）の基本挙動を提供する。 |
| `python/jax_util/base/_env_value.py` | 既定の `dtype` と数値定数（`EPS` など）を提供する。数値安定性のための定数を一元化する。 |
| `python/jax_util/base/__init__.py` | `base` 系の型・定数・`LinOp` を集約してエクスポートする窓口。利用側は原則ここから import する。 |
| `python/jax_util/_check_mv_operator.py` | 作用素の自己随伴性・SPD 性の簡易レポートを返すユーティリティ。安全確認の最小チェックに使う。 |
| `python/jax_util/_fgmres.py` | 右前処理付き FGMRES の実装。再起動、ワークスペース管理、ウォームスタートの状態保持を担う。 |
| `python/jax_util/_minres.py` | MINRES の実装。前処理付き反復解法を提供し、数値安定性の補助処理を含む。 |
| `python/jax_util/pcg.py` | PCG（前処理付き共役勾配法）の実装。投影付きの反復構成と収束判定を行う。 |
| `python/jax_util/kkt_solver.py` | KKT 系のブロックソルバ。`_minres.py` / `_fgmres.py` / `lobpcg.py` を組み合わせ、前処理を含めた解法を提供する。 |
| `python/jax_util/lobpcg.py` | LOBPCG 系の固有値推定。ブロック固有値更新、部分空間更新、前処理構築を扱う。 |
| `python/jax_util/matrix_util.py` | 行列ユーティリティ（直交化など）。小さく単機能の補助関数を置く。 |
| `python/jax_util/pdipm.py` | 内点法（PDIPM）の実装。目的関数・制約の構成、KKT 解法の呼び出し、簡易な問題生成を含む。 |
| `python/jax_util/_test_jax.py` | JAX の動作確認と最小の疎通チェック。環境の健全性を確認する。 |
| `python/jax_util/__init__.py` | パッケージのメタ情報（`__version__`）を保持する。 |
| `documents/type-aliases.md` | 型エイリアスの運用指針と一覧を整理する。 |

### 9. 依存関係の概要
| 対象 | 依存 | 目的 |
| --- | --- | --- |
| `python/jax_util/base/__init__.py` | `base/protocols.py`, `base/linearoperator.py`, `base/_env_value.py` | 型と定数の集約エクスポート。 |
| `python/jax_util/base/linearoperator.py` | `base/protocols.py` | `LinOp` の基盤型を共有する。 |
| `python/jax_util/_check_mv_operator.py` | `base` | 作用素チェック用の型・定数を利用する。 |
| `python/jax_util/_fgmres.py` | `base` | 反復解法の基盤型・定数を利用する。 |
| `python/jax_util/_minres.py` | `base` | 反復解法の基盤型・定数を利用する。 |
| `python/jax_util/pcg.py` | `base` | 反復解法の基盤型・定数を利用する。 |
| `python/jax_util/lobpcg.py` | `base`, `matrix_util.py` | 固有値計算の補助を利用する。 |
| `python/jax_util/kkt_solver.py` | `base`, `_check_mv_operator.py`, `_fgmres.py`, `_minres.py`, `lobpcg.py` | KKT 解法で各モジュールを統合する。 |
| `python/jax_util/pdipm.py` | `_env_value.py`, `kkt_solver.py` | 内点法で KKT 解法と定数を利用する。 |

### 10. テストの書き方
- `if __name__ == "__main__":` で関数を実行する形式とします。
- 例外が起きないことと、既知解に一致することを確認します。
- 大規模な乱数テストは避け、決定的な小規模入力を使います。