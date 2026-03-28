# エージェントチーム運用と役割分担

目的: このリポジトリで恒久的に使えるエージェントチームを定義し、毎回同じ入口、同じ証跡、同じ品質ゲートで運用すること。

## 正本

- チーム概要: `agents/README.md`
- 構成ファイル: `agents/agents_config.yaml`
- タスク workflow カタログ: `agents/TASK_WORKFLOWS.md`
- GitHub 側の運用指針: `.github/AGENTS.md`

## 恒久チームの役割

- Intent Analyst: ユーザー要求を読み取り、意図、受け入れ条件、曖昧点を `intent_brief.md` に落とす担当。
- Coordinator: スコープ固定、受け入れ条件定義、役割分担、エスカレーション担当。
- Editor: スコープ内の実装・文書変更担当。
- Change Reviewer: 実装中の各チャンクを逐次レビューする担当。
- Final Reviewer: editor と独立して最終レビューする担当。
- Verifier: `scripts/ci/pre_review.sh` と必要な CI コマンド実行担当。
- Auditor: 実行ログ、判断根拠、検証結果、ふりかえりの集約担当。

## 条件付き専門ロール

- Researcher: アルゴリズム調査、外部資料調査、インターネット検索担当。
- Scheduler: 大規模変更のマイルストーン管理、順序管理、依存関係管理担当。
- Infra Steward: experiment runner、CI、Docker、基盤自動化の保守・拡張担当。

## コンテキスト分離

- Editor と reviewer 群は private context を共有しない。
- reviewer は `intent_brief.md`、`research_notes.md`、`schedule.md`、diff、`verification.txt` などの明示的成果物だけを見る。
- この分離により、逐次レビューと最終レビューの独立性を守る。

## 実行フロー

1. Coordinator が run を作成する。
1. `python3 scripts/agent_tools/bootstrap_agent_run.py --task "<task>" --owner "<owner>"` を実行する。
1. 必要に応じて `--enable researcher --enable scheduler --enable infra_steward` または `--full-team` を付ける。
1. Intent Analyst が `intent_brief.md` を作成する。
1. `reports/agents/<run-id>/team_manifest.yaml` に沿って handoff を進める。
1. Editor が変更し、Change Reviewer が逐次 findings を返す。
1. Final Reviewer が独立レビューを行う。
1. Verifier が `scripts/ci/pre_review.sh` または `make ci` を実行する。
1. Auditor が `decision_log.md`、`verification.txt`、`retrospective.md` を完成させる。

## 標準出力

各 run は必ず以下を持つ:

- `reports/agents/<run-id>/intent_brief.md`
- `reports/agents/<run-id>/decision_log.md`
- `reports/agents/<run-id>/review_log.md`
- `reports/agents/<run-id>/team_manifest.yaml`
- `reports/agents/<run-id>/verification.txt`
- `reports/agents/<run-id>/retrospective.md`

必要に応じて追加される:

- `reports/agents/<run-id>/research_notes.md`
- `reports/agents/<run-id>/schedule.md`
- `reports/agents/<run-id>/infra_notes.md`

`AGENT_REPORT_DIR=reports/agents/<run-id>` を指定して `scripts/ci/pre_review.sh` を実行すると、検証結果を `verification.txt` に追記できます。

## 安全ルール

- `WORKTREE_SCOPE.md` がない worktree では開始しない。
- `main` へ直接 push しない。
- スコープ外変更が必要になったら、Coordinator が明示的に再スコープする。
- 自動化が削除や大規模移動を行う場合、人間承認を先に取る。
- close 前に少なくとも 1 回は検証を実行し、結果を保存する。

## 専門エージェント

タスクに応じて一時的な専門エージェントを追加してよい。ただし、必ず恒久チームのどれかが最終責任を持つ。

例:

- Design organizer: `scripts/tools/organize_designs.py` を扱う。
- Template generator: `scripts/tools/create_design_template.py` を扱う。
- Experiment reviewer: `experiments/` と `reports/` の整合性を確認する。
