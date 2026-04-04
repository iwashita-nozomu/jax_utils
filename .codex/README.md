# Codex Project Setup

このディレクトリは Codex-first の project-scoped 設定をまとめます。

## Layout

- `config.toml`
  - Codex の project 設定と subagent の入口
- `agents/*.toml`
  - project-scoped subagent 定義

## Shared Canon

- 共通の agent / workflow / skill 正本は `AGENTS.md` と `agents/` にあります。
- `.codex/` は Codex 実行面の設定だけを持ちます。

## Agents

- `explorer`
  - built-in `explorer` を repo 用に上書きした read-only agent
- `worker`
  - built-in `worker` を repo 用に上書きした implementation agent
- `docs_researcher`
  - official docs と外部仕様の確認を担当する
- `docs_workflow_steward`
  - 文書、workflow、notes の整備を担当する
- `docs_completeness_reviewer`
  - 文書の不足、入口不足、説明不足を確認する
- `docs_consistency_reviewer`
  - 文書間の矛盾、曖昧性、canon drift を確認する
- `reviewer`
  - diff と設計の妥当性を確認する
- `python_reviewer`
  - Python diff を pyright、ruff、型追跡で確認する
- `reproducibility_reviewer`
  - provenance、seed、command、environment、rerunability を確認する
- `scientific_computing_reviewer`
  - incremental change、testing、automation、prototype discipline を確認する
- `benchmark_reviewer`
  - benchmark fairness、confounder、anti-pattern を確認する
- `artifact_reviewer`
  - artifact package、rerunability、raw result の十分性を確認する
- `fair_data_reviewer`
  - metadata、命名、再利用性、result / note path を確認する
- `ml_science_reviewer`
  - assumptions、limitations、uncertainty、checklist closure を確認する
- `project_reviewer`
  - repo-wide な棚卸し、workflow health、tooling health を確認する
- `worktree_health_reviewer`
  - 現在の worktree の scope と cleanup readiness を確認する
- `experiment_planner`
  - 実験の cases、resource estimate、skip、report layout を確認する
- `hlo_investigator`
  - HLO dump、compiler behavior、XLA flag 仮説を整理する
- `report_reviewer`
  - user-facing experiment report の根拠、数値、図表、結論 traceability を確認する

## Usage

- まず `AGENTS.md` を読む
- `AGENTS.md` の routing table で workflow family と skill family を決める
- 必要なら `agents/CODEX_WORKFLOWS.md` で Codex-specific routing を確認する
- 次に必要な subagent を選ぶ
- 役割をまたぐ作業は、探索・実装・文書整備・レビューを分ける

## Chat Intake

- 局所修正なら parent がまず処理し、必要なら `explorer` と `reviewer` だけ使う
- 文書整理や workflow 整理なら `docs_workflow_steward` を使う
- 文書の不足は `docs_completeness_reviewer`、矛盾や曖昧性は `docs_consistency_reviewer` を使う
- bounded な実装を切り出したいときだけ `worker` を使う
- Python diff の厳密レビューは `python_reviewer` を使う
- 研究 scope の repo-wide review では research perspective reviewers を並列で使う
- repo 全体レビューや棚卸しは `project_reviewer` を使う
- worktree の健全性確認は `worktree_health_reviewer` を使う
- 実験計画や layout 確認は `experiment_planner` を使う
- HLO dump や XLA flag の仮説整理は `hlo_investigator` を使う
- report draft の独立レビューは `report_reviewer` を使う
- 常に parent が最終編集責任を持つ
- 書き込みは parent か `worker` / `docs_workflow_steward` に集約する
