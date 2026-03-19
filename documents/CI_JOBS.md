# CI / 自動化ジョブ仕様

目的: 静的解析・自動修正・レポート収集・衝突検出などの自動化ジョブを仕様化し、ジョブごとの入力／出力／制約／失敗時の扱いを明確化する。

共通前提
- 実行環境は可能な限り Docker コンテナで統一すること（`docker/` に追記）。
- 主要ツール: `ruff`, `black`, `pyright`, `pytest`。
- すべての成果物（解析 JSON, junit, logs）は `reports/` 以下に保存する。
- main への直接変更は禁止。すべての自動修正は feature ブランチで行う。

ジョブ一覧

1) auto-fix-imports
- 目的: `ruff` が提示する安全な自動修正（主に `I001` 等の import 整理）を限定的に適用する。
- トリガ: 手動／スケジュール／チャットOps コマンド。
- 手順:
  1. `origin/main` の最新コミットをベースに新ブランチ作成 (`fix/ruff-imports-YYYYMMDD`)
  2. `reports/static-analysis/ruff.json` を読み、対象ファイル一覧を取得
  3. 全ブランチの変更一覧と照合して『安全ファイル』を抽出（他ブランチで変更されていないこと）
  4. `ruff --fix` を上記安全ファイルに対して実行（ファイル単位に分割して実行）
  5. `black --line-length 100` を適用
  6. `ruff`/`pyright`/`pytest` を実行してレポートを `reports/` に保存
  7. 変更を小さなコミットで push し、PR を作成（PR ボディ自動生成）
- 入力: `reports/static-analysis/ruff.json`, `origin/main`、全ブランチの差分情報
- 出力: ブランチ + `reports/static-analysis/ruff_after_fix_run_safe.json`, PR
- 制約:
  - 決して main を更新しない。
  - 自動修正対象は『安全ファイル』のみ（他ブランチとファイル重複がないもの）。
  - テストが完全に通らない場合は自動マージしない（CI 緑が前提）。
  - 大きなリファクタや API 変更は自動化対象外。

2) report-collection
- 目的: リポジトリ全体の静的解析とテスト結果を収集・保存する。
- トリガ: PR 開始、手動、cron
- 手順: `ruff --format json`, `pyright`, `pytest --junitxml` を実行し、`reports/static-analysis/` と `reports/test-results/` に保存
- 制約: 解析はワークツリーに影響を与えない（読み取り専用）。失敗時はログを保存して終了コードは 0 を返す（集計ジョブとして失敗で pipeline を止めない）。

3) conflict-detection
- 目的: 自動修正前に、解析対象ファイルが他ブランチで編集されていないかを検出する。
- トリガ: auto-fix 前、あるいは手動
- 手順: 全ブランチの `git diff origin/main..branch` を走査し、解析ファイルの交差を `reviews/CONFLICT_REPORT.md` に出力
- 制約: レポートは必ず人が確認すること。自動で conflict を解消して main に push しない。

4) safe-file-extraction
- 目的: `ruff` 結果と全ブランチ差分を突合して、auto-fix 実行可能な『安全ファイル』を生成する。
- 出力: `/tmp/safe_files.txt`（CI 内部）および `reports/static-analysis/safe_files.txt`（恒久保存）
- 制約: ファイル名正規化（リポジトリ相対パス）を必ず行う。

5) worktree-scope-check
- 目的: `work/*` ワークツリーに `WORKTREE_SCOPE.md` があるかを確認し、欠落があれば `notes/worktrees/worktree_scope_report.md` を更新する。
- トリガ: 手動／定期
- 制約: スコープの追加はワークツリー所有者が行う。スクリプトは依頼文を生成するのみ。

6) gpu-test-separation
- 目的: GPU 依存テストを通常 CI から分離し、`integration/gpu` 用 workflow を提案する。
- 手順: `pytest -k "gpu"` を個別ワークフローに移し、通常 CI では `-m "not gpu"` を適用する。
- 制約: GPU ワークフローは専用 runner またはセルフホストでのみ実行する。

7) periodic-report
- 目的: 定期的に静的解析を実行し、未処理の項目を優先度付きで `reports/` に保存する。
- トリガ: cron（例: daily）
- 制約: 定期実行は main ベースで解析のみ行い、自動修正は行わない。

8) pr-template-fill
- 目的: 自動修正 PR を作る際に、変更概要、影響範囲、解析レポートへのリンクを自動で PR 本文に埋める。
- 制約: 自動生成内容はレビューで必ず改変・承認されること。

共通の失敗ポリシー
- 解析（`ruff`/`pyright`）がエラーを返した場合: レポートを `reports/` に保存し、PR を作る場合は「要修正」としてラベル付けする。
- テストが失敗した場合: 自動マージ禁止。PR に CI ログを添付し、担当者が手動で調査する。
- ジョブが内部エラーで停止した場合: ジョブログを `reports/logs/` に保存し、`auditor` 宛に通知する（メール/Issue）。

セキュリティ・運用上の注意
- 自動修正ジョブは必ず sandboxed 環境で実行し、任意のリポジトリコードを root 権限で実行しない。
- 外部アクセス（ネットワーク）を最小化し、認証情報は Secrets に格納する。

監査と記録
- すべてのジョブ実行は `reports/logs/<job>-<timestamp>.log` として保存する。
- 重要な意思決定（自動マージ、ルール変更）は `.github/agents/discussion.md` に記録する。

---
この仕様をもとに、必要な GitHub Actions ワークフローと補助スクリプトを順次作成してください。
