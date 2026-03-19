# Prioritized Static-Analysis Fixes

作業方針（遵守）:
- main 側のコードは変更しない。すべてワークツリー/ブランチで修正を行い、レビュー記録を残す。
- 自動で安全に直せるものは専用ブランチで実行（`ruff --fix` 等）。手作業が必要なものはこのレビューに詳細を残す。

概要（上位カテゴリ）:
- インポートの整頓（`I001` / safe fix が多い） — 多くは `ruff --fix` で対応可能。
- モジュールレベルの import がトップにない（`E402`） — コードの意味を壊さないよう手動移動が必要。
- 行長超過（`E501`） — 自動改行が難しい箇所あり。簡易修正は `black`（設定: `--line-length 100`）で軽減、残りは手動で改行・リファクタ。

上位ファイル（`ruff` 出力から件数上位 20 件）:

- 81  /workspace/experiments/functional/smolyak_scaling/render_smolyak_scaling_report.py
- 42  /workspace/python/jax_util/base/linearoperator.py
- 30  /workspace/python/jax_util/base/__init__.py
- 28  /workspace/experiments/functional/smolyak_scaling/run_smolyak_scaling.py
- 24  /workspace/experiments/functional/smolyak_hlo/run_smolyak_hlo_case.py
- 14  /workspace/python/jax_util/base/nonlinearoperator.py
- 8   /workspace/python/tests/experiment_runner/test_subprocess_scheduler_unit.py
- 8   /workspace/python/tests/solvers/test_solver_internal_branches.py
- 7   /workspace/python/jax_util/optimizers/pdipm.py
- 6   /workspace/python/jax_util/base/protocols.py
- 6   /workspace/python/jax_util/experiment_runner/subprocess_scheduler.py
- 5   /workspace/python/test.py
- 5   /workspace/python/tests/experiment_runner/test_subprocess_scheduler.py
- 5   /workspace/python/tests/functional/test_protocols_and_smolyak_helpers.py
- 5   /workspace/python/tests/neuralnetwork/test_layer_utils_and_training.py
- 5   /workspace/python/tests/solvers/test_slq.py
- 4   /workspace/python/jax_util/solvers/lobpcg.py
- 4   /workspace/python/tests/functional/test_smolyak.py
- 3   /workspace/python/jax_util/functional/smolyak.py
- 3   /workspace/python/jax_util/solvers/_minres.py

推奨優先度と対応方針（短期 -> 中期）:

1) 優先度: 高 — 自動で安全に修正できるもの
   - 対象: `I001`（imports の整頓）等。多くは `fix` が提供されている。
   - 手順:
     - ブランチを作成: `fix/ruff-imports-<YYYYMMDD>`（既存ブランチがあれば使う）
     - 実行:

```bash
ruff --fix --select I001,E401,E402,E501 .
black --line-length 100 .
ruff .
```

   - 成果物: 修正差分（インポート整形）と `reports/` に再実行ログ。

2) 優先度: 中 — 自動化で一部対応できるが手作業が必要なもの
   - 対象: `E402`（module-level import not at top）、一部の `E501`（長い行）
   - 方針: 自動化で移動させる箇所は慎重に。影響が出る箇所はテスト（ユニット）を小分けで実行して確認。
   - 役割: `reviewer` が影響範囲を検査し、`integrator` が小さな統合ブランチで適用。

3) 優先度: 低〜検討 — 設計/API 観点の修正
   - 対象: 長過ぎる式や可読性の低い部分、GPU テスト依存の設計変更。
   - 方針: 影響が大きい場合は提案ドキュメント（`reviews/` に PR 提案）を作成して合意後に着手。

特記事項（既知の環境依存）:
- GPU に依存するテストは現在の CI では失敗しやすい（環境不足）。これらは分離して `integration/gpu` 用のワークフローを作成するか、ローカル実行手順を `notes/` に記載する。

短期アクションプラン（実行者向け）:
 - step A: `fix/ruff-imports-YYYYMMDD` ブランチで `ruff --fix` を実行し、自動修正を適用（コミットは小単位）。
 - step B: `black --line-length 100` を run してから `ruff` を再実行。
 - step C: 残った `E402`・`E501` をファイル単位でレビューリスト化し、各ファイルの担当者を割当てる（この `reviews/PRIORITIZED_FIXES.md` に担当と期限を追記）。
 - step D: 各修正ブランチで CI を実行し、結果（`reports/`）を証跡として添付。

再現コマンド（チェック用）:

```bash
# 解析を再現するためのコマンド例
ruff --format json . > reports/static-analysis/ruff_after_fix_run.json
pyright > reports/static-analysis/pyright_after_run.txt || true
pytest -q || true
```

次の作業候補（このリポジトリ上で実行可能）:
- 自動修正ブランチの作成と `ruff --fix` 実行（コード編集はブランチ内のみ）。
- 自動修正後のレポートを `reports/` に保存し、差分を `reviews/` に記録。

---
作成: Agent 自動集計
