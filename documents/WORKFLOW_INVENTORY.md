# ワークフロー目録（自動化現状と未自動化項目）

この文書は現状のスクリプト・ワークフローを一覧化し、未自動化の作業を洗い出すための目録です。

自動化済みスクリプト / ワークフロー（ローカル・CI 共通）:

- `scripts/ci/run_static_checks.sh` — `ruff` / `pyright` / `black --check` / `pytest` を実行して `reports/static-analysis/` に出力。
- `scripts/ci/safe_file_extractor.py` — ruff レポートと全ブランチ差分を突合して安全ファイルを列挙。
- `scripts/ci/safe_fix.sh` — 安全ファイルに対して `ruff --fix` と `black` 実行、オプションでコミット。
- `scripts/ci/collect_reports.sh` — `reports/static-analysis/` をアーカイブ。
- `scripts/tools/organize_designs.py` — 設計ファイルをサブモジュール別にコピー（保守的）。
- `scripts/tools/create_design_template.py` — サブモジュール用テンプレートを作成。
- `scripts/tools/find_redundant_designs.py` — 完全一致の重複設計ファイルを検出・削除（オプション）。

未自動化（要対応）:

1. ブランチスコープの CI 自動検出（PR が複数サブモジュールを跨いでいる場合に警告する GitHub Action）。
   - 理由: 1 ブランチ = 1 サブモジュール方針を技術的に補強するため。
1. ワークツリー `WORKTREE_SCOPE.md` の存在チェックとオーナー通知（自動作成は既存 `create_worktree.sh` があるが、既存ワークツリー未整備を検出していない）。
1. 設計ファイル類似度検出（完全一致でないが内容が重複しているファイルの候補抽出）。
1. 設計移行の dry-run → PR 自動作成フロー（organize_designs の実行結果を元に PR を用意）。
1. 設計ドキュメントインデックス自動生成（`documents/design/README.md` と各サブモジュール README の更新）。

優先度（提案）:

- 高: (1) ブランチスコープの PR 時警告、(2) ワークツリー scope チェック + owner 通知
- 中: (3) 類似度検出ツール、(4) 設計移行の dry-run → PR 自動化
- 低: (5) インデックス自動生成（cron で十分）

次ステップ（短期）:

1. `scripts/tools/check_worktree_scopes.sh` を追加して全ワークツリーの `WORKTREE_SCOPE.md` を検出・レポート化（既に追加実行済み、結果は `reports/worktree_scope_report.txt`）。
1. GitHub Actions で (1) を実装するための `.github/workflows/branch-scope-check.yml` を作成。
1. (3) 類似度検出はまずローカルツールで候補抽出し、レビュワーに割り当てる。
