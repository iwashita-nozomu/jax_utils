# 命名規約（Python）

対象: `python/jax_util/` 配下。

## 要約

- **ファイル名**は `snake_case.py`。
- **関数名**は `snake_case`。
- **公開API** と **内部実装** を名前で区別します。

## ファイル名

- `snake_case.py` を使ってください。
- 略語は一般的なもののみ許可します（例: `pcg`, `minres`, `kkt`, `hlo`）。
- 役割が分かる語を置き、意味のない接尾辞（`_util` の乱用等）は避けます。

### 禁止/注意

- 大文字を含むファイル名（例: `MyAlgo.py`）は避けます。
- OS差分が出やすい表記（空白・ハイフン混在）は避けます。

## 関数名

- `snake_case` を使ってください。
- **動詞で始める**ことを推奨します（例: `solve_*`, `init_*`, `make_*`, `update_*`, `check_*`, `dump_*`）。

### 公開API と内部関数

- **公開API**: モジュールの `__all__` に載せる関数/クラス。
  - 名前は先頭 `_` なし。
  - 例: `pcg_solve`, `minres_solve`, `kkt_block_solver`
- **内部実装**: 先頭 `_`。
  - 例: `_pdipm_solve`, `_sym_ortho`

### テスト用補助

- テストファイルでは `test_*` を使います。
- `python file.py` 直実行用の補助を置く場合は、外部から使わないことが分かるよう `_run_*` を推奨します。

## 例

- `pdipm_solve`（公開）→ `_pdipm_solve`（内部）
- `initialize_kkt_state`（公開）→ `_kkt_block_solver`（内部）
