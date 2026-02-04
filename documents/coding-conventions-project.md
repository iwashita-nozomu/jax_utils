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
- 仮想環境の作成は行いません（Docker 環境前提）。
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
- `base` / `Algorithms` に依存するだけに留めます。

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
| `python/tests` | `base`, `Algorithms` | 検証とログの出力。 |
| `scripts` | `python/tests`, `documents` | 実行補助（テスト・ログ運用）。 |

### 4.7 ファイルごとの依存関係（要点）
- 詳細な表は `documents/conventions/python/10_dependencies.md` に集約します。
- 代表的な依存関係は以下の通りです。
  - `python/jax_util/base/__init__.py` → `base/protocols.py`, `base/linearoperator.py`, `base/_env_value.py`
  - `python/jax_util/Algorithms/_minres.py` → `base`
  - `python/jax_util/Algorithms/pcg.py` → `base`
  - `python/jax_util/Algorithms/kkt_solver.py` → `base`, `Algorithms/_check_mv_operator.py`, `Algorithms/_minres.py`, `Algorithms/lobpcg.py`
  - `python/jax_util/Algorithms/pdipm.py` → `base/_env_value.py`, `Algorithms/kkt_solver.py`

## 5. 参照
- 依存関係の詳細: `documents/conventions/python/10_dependencies.md`
- ファイル役割: `documents/conventions/python/09_file_roles.md`
