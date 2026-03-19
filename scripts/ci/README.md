# CI helper scripts

配置: `scripts/ci/`

目的: 再利用可能な小さなスクリプト群を置き、GitHub Actions などのワークフローはこれらを呼ぶだけにする。

主要スクリプト:

- `run_static_checks.sh [report_dir]`:
  - 実行: `ruff`, `pyright`, `black --check`, `pytest --junitxml`
  - 出力: レポートを `reports/static-analysis/` に保存

- `safe_file_extractor.py`:
  - 入力: `reports/static-analysis/ruff.json`
  - 出力: `/tmp/safe_files.txt`, `reports/static-analysis/safe_files.txt`（リポジトリ相対パス）
  - 役割: 全ブランチの差分と突合して auto-fix 可能なファイルを抽出する

- `safe_fix.sh [--commit]`:
  - 役割: `safe_file_extractor.py` を呼び出し、安全ファイルに `ruff --fix` → `black` を実行
  - `--commit` を渡すと変更をコミットして push する（CI では通常 false）

- `collect_reports.sh [outdir]`:
  - 役割: `reports/static-analysis` を圧縮して CI アーティファクト用に保存

実行方針:
- スクリプトは idempotent を目指し、CI からもローカルからも実行できること。
- 自動コミット・push はオプションとし、デフォルトではログ・レポートのみ生成する。
