# `jax_util` 詳細設計インデックス

このディレクトリは `python/jax_util/` に対応する詳細設計を集約します。目的は:

- サブモジュールごとに設計責任を明確にすること
- ドキュメントを 3 参照以内に辿れるようにインデックスを整備すること

構成とナビゲーション:

- `README.md` — このファイル。各設計の目次と依存関係の中央集約を含みます。
- `hlo.md` — HLO ダンプの設計（`python/jax_util/hlo/` に対応）。
- `differential_equations.md` — 微分方程式 problem catalog の設計（`python/jax_util/differential_equations/`）。
- `optimizers.md` — 最適化器（`python/jax_util/optimizers/`）。
- `solvers.md` — 線形ソルバ（`python/jax_util/solvers/`）。
- `base_components.md` — 型エイリアス、共通プロトコル、`LinOp` など基盤要素。

依存関係（中央集約）:

- `jax_util` のサブモジュール間依存の指針:
  - `base` は基盤（型、protocols、環境設定）として他のサブモジュールの第一の依存先です。
  - `differential_equations` は import-safe な metadata layer とし、solver や experiment code から参照される側に置きます。
  - `solvers` は数値線形代数を提供し、`optimizers` が KKT などで利用します。
  - `hlo` は副次的なツール（ダンプ/解析）であり、通常は実行時に明示的に有効化します。

編集方針（重複削減）:

- サブドキュメント(`optimizers.md` / `solvers.md` / `hlo.md`) に記載されている共通的な依存関係・方針はここに集約し、個々のファイルには実装固有の要点のみを残します。

3 参照以内のルールの担保:

1. リポジトリルート → `documents/design/` → `documents/design/jax_util/README.md`（このファイル）
1. `README.md` から該当サブドキュメントへ直接リンク
1. 個別ドキュメントは `公開API`・`対象コードパス` を先頭に置く

メンテナンス:

- 新しい設計を追加する場合は、既存の個別設計文書と同じ見出し構成を踏襲し、`README.md` にリンクを追加してください。
