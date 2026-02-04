# プロジェクト全体の運用規約

この文書は、Docker 環境・ドキュメント群・サブモジュール依存関係の運用方針をまとめます。

## 1. 対象
- 対象は `/workspace` 配下の全体構成です。
- 実装の詳細規約は `documents/coding-conventions-*.md` と
  `documents/conventions/` の各章を参照します。

## 2. Docker 環境の方針
- 既定の実行環境は `docker/Dockerfile` を基準にします。
- 追加パッケージが必要な場合は、**Dockerfile と requirements を同時に更新**します。
  - Python パッケージは `docker/requirements.txt` に追記します。
  - JAX などのシステム依存は `docker/Dockerfile` のインストール手順で管理します。
- 仮想環境の作成は **一切行いません**（Docker 環境前提）。
- `venv` / `virtualenv` / `conda` などの導入は **禁止**します。
- **Docker の Bash 環境を前提**とし、ローカル環境との差分は規約に明記します。

## 3. ドキュメント運用
- 変更が入った場合は、該当する `documents/` 内の文書を同時に更新します。
- 目次は以下を基準に整理します。
  - 共通規約: `documents/coding-conventions.md`
  - Python 規約: `documents/coding-conventions-python.md`
  - ログ規約: `documents/coding-conventions-logging.md`
  - テスト規約: `documents/coding-conventions-testing.md`
  - ソルバー規約: `documents/coding-conventions-solvers.md`

## 4. サブモジュールと依存関係
- 依存関係は「基盤 → 上位」の一方向を維持します。
- 依存関係は **プロジェクト全体** と **ファイル単位** の 2 段で整理します。

### 4.1 `python/jax_util/base`
- 型・定数・基本作用素の基盤です。
- 依存元は `base` に集約します。

### 4.2 `python/jax_util/Algorithms`
- 数値アルゴリズム群の実装です。
- `base` に依存し、`Algorithms` 同士の依存は最小限に抑えます。

### 4.3 `python/tests`
- テストは `python/tests/` に集約します。
- `base` / `Algorithms` / `neuralnetwork` に依存するだけに留めます。

### 4.4 `scripts`
- 実行補助スクリプト群です。
- 既存のテスト・ログ規約と整合するよう運用します。

### 4.5 `documents`
- 規約と運用方針の一次情報源です。
- 実装変更と同時に更新します。

### 4.6 プロジェクト全体の依存関係
| レイヤ | 依存先 | 役割 |
| --- | --- | --- |
| `docker/` | OS / CUDA / Python / JAX | 実行環境の基盤。バージョンと依存の源泉。 |
| `documents/` | なし | ルールの一次情報源。実装に合わせて更新する。 |
| `python/jax_util/base` | なし（標準ライブラリと JAX 等のみ） | 型・定数・基本作用素の基盤。 |
| `python/jax_util/Algorithms` | `base` | 数値アルゴリズムの実装層。 |
| `python/tests` | `base`, `Algorithms`, `neuralnetwork` | 検証とログの出力。 |
| `scripts` | `python/tests`, `documents` | 実行補助（テスト・ログ運用）。 |

### 4.7 ファイルごとの依存関係（要点）
- 詳細な表は `documents/conventions/python/10_dependencies.md` に集約します。
- 代表的な依存関係は以下の通りです。
  - `python/jax_util/base/__init__.py` → `base/protocols.py`, `base/linearoperator.py`, `base/_env_value.py`
  - `python/jax_util/Algorithms/_minres.py` → `base`
  - `python/jax_util/Algorithms/pcg.py` → `base`
  - `python/jax_util/Algorithms/kkt_solver.py` → `base`, `Algorithms/_check_mv_operator.py`, `Algorithms/_minres.py`, `Algorithms/lobpcg.py`
  - `python/jax_util/Algorithms/pdipm.py` → `base/_env_value.py`, `Algorithms/kkt_solver.py`
  - `python/jax_util/neuralnetwork/neuralnetwork.py` → `base`, `neuralnetwork/protocols.py`, `neuralnetwork/layer_utils.py`

## 5. プロジェクト共通の実装規約
この節は **Algorithms 以外にも適用される**共通ルールです。

### 5.1 適用範囲
- `python/jax_util/` 配下の全サブパッケージに適用します。
- `Algorithms` / `neuralnetwork` / `base` / `tests` を含みます。

### 5.2 型・形状の統一
- `Scalar` / `Vector` / `Matrix` を優先します。
- `Matrix` は **ベクトルのバッチ**を含む型として扱います。
- `Matrix` のバッチ軸は **最後の軸（axis=-1）** を原則とします。
- バッチ軸の定義は **この節に集約**し、他文書は参照のみとします。

#### 5.2.1 形状の見方（Solver / NN 共通）
- 本プロジェクトでは「ベクトルは列ベクトル」とみなし、`Matrix` は **列ベクトルを並べたもの**として扱います。
- 具体的には、次を原則とします。
  - `Vector`: `shape == (n,)`
  - `Matrix`: `shape == (n, batch)`（最後の軸がバッチ）
- この統一により、明示行列 `A: Matrix` を用いた `A @ x` と、線形作用素 `Mv @ x` の読み替えが自然になります。
- NN も同様に `x.shape == (in_dim, batch)` を入力の原則とし、Solver 系と形状規約を共有します。

### 5.3 作用素・演算子の統一
- 適用は `@`、合成は `*` を原則とします。
- 例外や特殊用途は `documents/conventions/python/12_operator_rules.md` に従います。

### 5.4 JAX のルール
- 反復は `jax.lax.scan` / `jax.lax.while_loop` を優先します。
- Python の `for` は **JIT 対象外の範囲のみ**で使います。

### 5.5 ログとデバッグ
- ログ出力は `DEBUG` ガードの内側でのみ行います。
- `jax.debug.print` を優先します。

### 5.6 テストと実行
- テストは `python/tests/` に集約します。
- ライブラリ配下に `__main__` を置かず、実行はテスト側で行います。

## 6. 環境設定の運用
### 6.1 集約ルール
- 実行環境の設定は `python/jax_util/base/_env_value.py` に集約します。
- 設定の追加・変更時は **規約とセット**で更新します。

### 6.2 環境変数の扱い
- 環境変数は **上書き専用**です。
- 既定値はコード側で保持し、環境変数がない場合は既定値を使います。

### 6.3 実行手順の明文化
- 実行時に必要な環境変数は `documents/` に記載します。
- 変更があれば **必ず更新**します。
- 仮想環境の導入は **禁止**します。依存は Docker と `docker/requirements.txt` で管理します。

## 7. タスク運用
### 7.1 `task.md` の位置づけ
- `task.md` は **実装の進行管理**に使います。
- 粒度は **ファイル単位**で統一します。

### 7.2 更新ルール
- 完了した項目は `task.md` から削除します。
- 詳細設計や仕様は `documents/` に追記します。
