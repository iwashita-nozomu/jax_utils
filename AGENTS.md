# Agent Instructions

このファイルは、Codex、Claude、GitHub Copilot などのエージェント向け共通入口です。
当面の primary runtime は Codex とします。
共有の正本は `agents/` に置きます。tool 固有の入口はここへ重い説明を複製しません。

## Read Order

1. `README.md`
1. `agents/README.md`
1. `agents/CODEX_WORKFLOWS.md`
1. `agents/TASK_WORKFLOWS.md`
1. `agents/skills/README.md`
1. `documents/AGENTS_COORDINATION.md`

## Request Routing

ユーザーの自然言語の依頼を受けたら、実装の前に次を決めます。

1. `agents/TASK_WORKFLOWS.md` から workflow family を選ぶ
1. `agents/skills/README.md` から skill family を選ぶ
1. 完了判定の前に必要な review を決める
1. 必要なときだけ `.codex/agents/*.toml` の subagent を選ぶ
1. parent agent が最終編集責任を持つ

## Default Routing

- 局所バグ修正、テスト修正、README 追随
  - workflow: `Scoped Correction`
  - skill: `static-check`, `code-review`
  - required review: 実装後に `code-review`。Python 差分なら `python-review` も通す。必要なら `reviewer`
  - subagent: 通常は不要。Python 差分なら `python_reviewer`。必要なら `explorer`、修正後に `reviewer`
- 外部調査つき実装、性能改善、比較実験
  - workflow: `Research-Driven Change`
  - skill: `research-workflow`, `experiment-lifecycle`, `critical-review`, `report-review`
  - required review: 外部調査、比較条件、exit criteria を先に固定する。各反復で `implement -> run -> critical-review -> report-review` を通し、`report_rewrite_required`、`extra_validation_required`、`rerun_required` のいずれかが解消するまで loop を閉じない。methodology、benchmark protocol、artifact、reporting policy を大きく変える場合は `research-perspective-review` も通す
  - subagent: `docs_researcher` または `explorer` で調査、`experiment_planner` で protocol と run layout を確認、必要なら research perspective reviewers を並列実行、実装後に `reviewer`、report draft 後に `report_reviewer`
- HLO 解析、XLA flag 調査、compiler behavior 比較
  - workflow: `Research-Driven Change`
  - skill: `research-workflow`, `experiment-lifecycle`, `python-review`, `critical-review`
  - required review: baseline と change 後で HLO dump、runtime metric、`xla_env` / `XLA_*` 方針を同じ protocol で比較する。flag か code は 1 回の反復で 1 種類だけ変え、`critical-review` を通す。user-facing report を閉じるなら `report-review` も通す
  - subagent: `hlo_investigator` で HLO dump と flag 仮説を整理し、必要なら `docs_researcher` で XLA / JAX の仕様確認、実装後に `python_reviewer`
- 大規模 refactor、新 API
  - workflow: `Large Delivery`
  - skill: `code-review`
  - required review: 設計差分と公開面を `code-review`。Python 公開面なら `python-review`。実装後に `reviewer`
  - subagent: `explorer` で影響範囲確認。Python 差分なら `python_reviewer`。必要なら `reviewer`
- 文書整理、workflow 整理、notes 吸収
  - workflow: `Scoped Correction` または `Platform and Infra`
  - skill: `docs-completeness-review`, `md-style-check`, `docs-consistency-review`
  - required review: 変更後に `docs-consistency-review`。Markdown 変更なら `md-style-check`。必要なら `comprehensive-review`
  - subagent: `docs_workflow_steward`、`docs_completeness_reviewer`、`docs_consistency_reviewer`
- 実験準備、run、result/report 整理
  - workflow: `Research-Driven Change` または `Platform and Infra`
  - skill: `experiment-lifecycle`, `critical-review`, `report-review`
  - required review: run 前に比較条件を確認し、run 後に `critical-review`。ユーザー向け report の完了前に `report-review`
  - subagent: `experiment_planner`、report draft 後に `report_reviewer`
- repo 全体レビュー、棚卸し
  - workflow: `Platform and Infra`
  - skill: `project-review`
  - required review: `project-review` を正本にし、必要なら `comprehensive-review` と `project-health` を併用する。scope に experiments、reports、benchmark protocol、research docs が入る場合は `research-perspective-review` を追加する
  - subagent: `project_reviewer`、必要なら `docs_researcher`、`worktree_health_reviewer`、`reproducibility_reviewer`、`scientific_computing_reviewer`、`benchmark_reviewer`、`artifact_reviewer`、`fair_data_reviewer`、`ml_science_reviewer`

## Canonical Layout

- `agents/` は team shape、workflow、template、shared skill の正本です。
- `agents/CODEX_WORKFLOWS.md` は Codex-first の workflow と subagent 運用の正本です。
- `agents/skills/` は tool 非依存の skill 文書正本です。
- `.agents/skills/` は Codex / Copilot などの自動発見向け shim です。
- `.codex/config.toml` と `.codex/agents/*.toml` は Codex の project-scoped runtime config です。
- `CLAUDE.md` は Claude 用の薄い adapter です。
- `.github/AGENTS.md` と `.github/copilot-instructions.md` は GitHub 用の薄い adapter です。
- `.github/skills/` は GitHub Actions や CLI から呼ぶ実装面です。skill 説明の正本にはしません。

## Repo Rules

- 日本語で作業します。
- ローカル仮想環境は作りません。Docker 構成を正本にします。
- 正本の文書は `documents/`、補助知見は `notes/`、日付ログは `diary/` に置きます。
- agent system を更新するときは、まず `agents/` を直し、その後に tool adapter を追随させます。
- role 一覧、handoff、skill 説明を tool 固有ファイルへ重複記載しません。
- Codex subagent は explicit に頼む場合だけ使い、parent が最終編集責任を持ちます。
- Codex subagent は routing で必要性が出たときだけ使います。常時 spawn する運用にはしません。
- `Research-Driven Change` と実験結果を伴う依頼では、`critical-review` を省略して完了扱いにしません。
- `Research-Driven Change` では、`baseline or current state -> implement -> run -> critical-review -> report-review -> rewrite or extra validation or rerun` の loop を省略して完了扱いにしません。
- methodology、benchmark protocol、artifact、reporting policy を大きく変える研究系変更では、`research-perspective-review` を通さずに完了扱いにしません。
- `experiments/report/<run_name>.md` を生成する依頼では、`report-review` を省略して完了扱いにしません。
- repo-wide な整理や workflow 改造では、変更後の review を省略して完了扱いにしません。
- `python/` 配下を触る依頼では、`pyright` と `ruff` を無視した review を完了扱いにしません。
- Markdown を触る依頼では、`md-style-check` を無視した review を完了扱いにしません。
- 文書整理では、`docs-consistency-review` を通さずに完了扱いにしません。
- worktree を使う依頼では、必要に応じて `worktree-health` を通し、scope drift を放置して完了扱いにしません。

## If You Edit The Agent System

- team / role / workflow は `agents/README.md`、`agents/TASK_WORKFLOWS.md`、`agents/task_catalog.yaml` を更新します。
- Codex subagent の使い分けは `agents/CODEX_WORKFLOWS.md` と `.codex/agents/*.toml` を更新します。
- skill の目的、入出力、使い分けは `agents/skills/` を更新します。
- tool 固有ファイルでは、共有正本への入口と、その tool で必要な最小補足だけを書きます。
