# Codex Workflows

この文書は、Codex を primary runtime として使うときの workflow と subagent 運用の正本です。

## Scope

- `AGENTS.md` から最初に読む Codex-specific supplement
- `.codex/config.toml` と `.codex/agents/*.toml` の意味を示す
- built-in agent と custom agent の使い分けを固定する
- ユーザーの依頼文から workflow と subagent をどう選ぶかを固定する

## Primary Runtime

- 当面の primary runtime は Codex です。
- Claude と GitHub Copilot は adapter として残します。
- shared canon は引き続き `agents/` に置きます。

## Intake Flow

1. ユーザーの依頼を `agents/TASK_WORKFLOWS.md` の workflow family に分類する
1. `agents/skills/README.md` から必要な skill family を選ぶ
1. 完了前に必要な review を決める
1. その場で解けるなら parent が進める
1. 調査、文書整理、レビュー、実験計画を分けたいときだけ subagent を使う
1. parent が最終方針、最終編集、最終応答を持つ

## Built-In Agents

- `default`: 汎用 fallback
- `worker`: 実装や修正を進める execution-focused agent
- `explorer`: 読み取り中心の codebase exploration agent

この repo では `.codex/agents/explorer.toml` と `.codex/agents/worker.toml` を置き、built-in agent に repo-specific instruction を上書きします。

## Project-Scoped Custom Agents

- `.codex/agents/docs_workflow_steward.toml`
  - `AGENTS.md`、`agents/`、adapter file、repo canon の整理と文書更新
- `.codex/agents/docs_researcher.toml`
  - official docs の確認、外部仕様の照合、主張を弱める根拠探し
- `.codex/agents/docs_completeness_reviewer.toml`
  - 文書の欠落、説明不足、入口不足を潰す
- `.codex/agents/docs_consistency_reviewer.toml`
  - 文書間の矛盾、曖昧性、canon drift を潰す
- `.codex/agents/reviewer.toml`
  - findings-first review。主張にはかなり批判的に当たる
- `.codex/agents/python_reviewer.toml`
  - Python diff を pyright、ruff、型追跡で潰す
- `.codex/agents/project_reviewer.toml`
  - repo-wide な inventory、workflow health、tooling health を潰す
- `.codex/agents/worktree_health_reviewer.toml`
  - 現在の worktree の scope drift と cleanup risk を潰す
- `.codex/agents/experiment_planner.toml`
  - 実験設計、resource estimate、skip、result/report 整理
- `.codex/agents/hlo_investigator.toml`
  - HLO dump、compiler behavior、XLA flag 仮説、`xla_env` の使い方を整理する
- `.codex/agents/report_reviewer.toml`
  - user-facing experiment report の独立レビュー

## Subagent Rules

- Codex は explicit に頼まれたときだけ subagent を spawn します。
- parent agent が write responsibility を持ちます。
- subagent は基本 read-only で使います。
- 並列化は探索、レビュー、資料確認に使います。
- 並列の write-heavy 実装は避けます。
- `max_depth` は `1` に固定し、再帰的 fan-out を防ぎます。
- review では「通す理由」より「止める理由」を優先して探します。
- 実験や設計の主張がある場合は、必要に応じて `docs_researcher` を足して文献や仕様の反証を探します。
- `Research-Driven Change` では、外部調査、比較条件固定、実装、run、`critical-review`、`report-review` を 1 回で終わらせず、`report_rewrite_required`、`extra_validation_required`、`rerun_required` が解消するまで loop を回します。
- `experiments/report/<run_name>.md` を閉じる前には、`report_reviewer` か同等の `report-review` 実施を要求します。
- `python/` 配下の差分では、必要に応じて `python_reviewer` を使い、pyright と ruff を review 根拠に含めます。
- repo 全体レビューや棚卸しでは、必要に応じて `project_reviewer` を使います。
- 文書整理では、必要に応じて `docs_completeness_reviewer` と `docs_consistency_reviewer` を使います。
- worktree 運用では、必要に応じて `worktree_health_reviewer` を使います。

## Routing Table

| User Request Pattern | Workflow Family | Skill Family | Required Review | Default Codex Flow |
| -------------------- | --------------- | ------------ | --------------- | ------------------ |
| 局所バグ修正、テスト修正、README 追随 | `Scoped Correction` | `static-check`, `code-review` | 実装後に `code-review`。Python 差分なら `python-review`。必要なら `reviewer` | parent 直処理。必要なら `explorer` → parent 実装 → `python_reviewer` または `reviewer` |
| 外部調査つき実装、性能改善、比較検証 | `Research-Driven Change` | `research-workflow`, `experiment-lifecycle`, `critical-review`, `report-review` | 外部調査、比較条件、exit criteria を先に固定し、各反復で `critical-review` と `report-review` を通す。`report_rewrite_required`、`extra_validation_required`、`rerun_required` が残る限り loop を閉じない。実装後に `reviewer`。必要なら `docs_researcher` で反証探し | `docs_researcher` または `explorer` で調査 → `experiment_planner` で protocol 固定 → parent 実装 → `reviewer` → run → `report_reviewer`。decision に応じて rewrite / extra validation / rerun / re-implement を反復 |
| HLO 解析、XLA flag 調査、compiler behavior 比較 | `Research-Driven Change` | `research-workflow`, `experiment-lifecycle`, `python-review`, `critical-review` | baseline と change 後で HLO dump、runtime metric、`xla_env` / `XLA_*` 方針を同じ protocol で比較する。flag か code は 1 回の反復で 1 種類だけ変え、`critical-review` を通す。report を閉じるなら `report-review` も通す | `hlo_investigator` で dump / flag 仮説を整理 → 必要なら `docs_researcher` で仕様確認 → parent 実装 → `python_reviewer` → run → `critical-review`。decision に応じて HLO 再採取、追加検証、fresh rerun を反復 |
| 大規模 refactor、新 API | `Large Delivery` | `code-review` | 設計差分と公開面を `code-review`。Python 公開面なら `python-review`。実装後に `reviewer` | `explorer` で影響範囲確認 → parent または `worker` 実装 → `python_reviewer` または `reviewer` |
| Python module change、pyright warning cleanup、type boundary refactor | `Scoped Correction` または `Large Delivery` | `python-review`, `static-check` | `pyright` と `ruff` を確認し、`python-review` を通す | parent 実装 → `python_reviewer`。必要なら `reviewer` を追加 |
| 実験準備、run、result/report 整理 | `Research-Driven Change` または `Platform and Infra` | `experiment-lifecycle`, `critical-review`, `report-review` | run 前に比較条件確認、run 後に `critical-review`。report 完了前に `report-review`。主張が強いなら `docs_researcher` で文献確認 | `experiment_planner` で既存構成と計画を確認 → parent 更新 → report draft 後に `report_reviewer` |
| 文書整理、workflow 整理、notes 吸収 | `Scoped Correction` または `Platform and Infra` | `docs-completeness-review`, `md-style-check`, `docs-consistency-review` | 変更後に `docs-consistency-review`。Markdown 変更なら `md-style-check`。必要なら `comprehensive-review` | `docs_workflow_steward` → `docs_completeness_reviewer` / `docs_consistency_reviewer` |
| 現在の worktree 健全性確認、scope drift 確認 | `Platform and Infra` | `worktree-health` | `worktree-health` を通し、必要なら `project-review` へ拡張 | `worktree_health_reviewer` |
| repo 全体レビュー、棚卸し | `Platform and Infra` | `project-review` | `project-review` を正本にし、必要なら `comprehensive-review` と `project-health` を併用 | `project_reviewer`、必要なら `docs_researcher` を並列 |

## Recommended Patterns

### 1. Repo Exploration

- parent が task を分割する
- `explorer` に affected area の調査を依頼する
- parent が結果を統合して実装方針を決める

### 2. Docs And Workflow Cleanup

- `docs_workflow_steward` に shared canon と adapter の重複を洗わせる
- `docs_completeness_reviewer` に missing section、missing command、missing path を洗わせる
- `docs_consistency_reviewer` に canon drift、矛盾、曖昧性を洗わせる
- Markdown 変更があるなら `md-style-check` を通す
- 必要なら `docs_researcher` に official docs の確認を並列で依頼する
- parent が正本へ反映する

### 2.5. Repo-Wide Project Review

- `project_reviewer` に inventory、workflow health、tooling health を見させる
- worktree を含む作業なら `worktree_health_reviewer` に scope drift と cleanup risk を見させる
- 必要なら `docs_researcher` を並列で使い、外部仕様との差分を確認する
- findings を `fix now`、`follow-up`、`delete-ok` に分ける
- parent が cleanup、notes 吸収、削除判断を反映する

### 3. Implementation And Review

- 実装前に `explorer` で影響範囲を確認する
- parent が実装する
- 実装後に `reviewer` へ review を依頼する
- Python diff では `python_reviewer` に、pyright、ruff、型追跡、warning 処理を重点確認させる
- 主張が強い場合や upstream 根拠が必要な場合は `docs_researcher` を追加する

### 4. Research-Driven Change Loop

- `docs_researcher` または `explorer` で問い、比較対象、関連文献、反証候補を確認する
- `experiment_planner` に cases、resource estimate、skip 条件、result / report layout、exit criteria を確認させる
- parent が 1 つの change だけを実装する
- `reviewer` と必要なら `python_reviewer` を通してから run する
- `experimenter` が同じ protocol で baseline または current state と change 後を比較する
- `critical-review` で比較公平性、evidence sufficiency、overclaim を潰す
- report draft が出たら `report_reviewer` に、概要、数値、図表、結論と根拠の対応を確認させる
- `report_reviewer` が `report_rewrite_required` を返したら、同じ result で report だけを書き直して再レビューする
- `critical-review` か `report-review` が `extra_validation_required` を返したら、同じ仮説のまま追加 case、追加 figure、追加集計を行って再レビューする
- `critical-review` か `report-review` が `rerun_required` を返したら、fresh `run_name` で再実行し、必要なら parent が code か protocol を修正してから loop を回し直す
- 両 review が通り、exit criteria を満たしたときだけ loop を閉じる
- 結論を閉じる前に、必要なら `docs_researcher` で関連文献や仕様の反証を追加で探す
- parent が実験コード、report、notes を更新する

### 4.5. HLO Analysis And Compiler-Tuning Subflow

- `hlo_investigator` に、対象関数、HLO dump path、比較対象 run、想定 bottleneck を整理させる
- baseline と change 後で `jax_util.hlo.dump` の出力を同じ protocol で採取する
- `scripts/hlo/summarize_hlo_jsonl.py` などで op 頻度、dialect、主要差分を集計する
- `jax_util.xla_env` と `XLA_*` / `JAX_PLATFORMS` 方針を確認し、script 側の ad hoc env 分岐を禁止する
- 1 回の反復では code change か flag change のどちらか 1 種類だけを変える
- 実装後は `python_reviewer` に、型と env 初期化境界、warning 処理を重点確認させる
- run 後は HLO 差分だけでなく runtime metric、failure kind、backend 条件も比較する
- HLO が変わったことだけを根拠に改善と扱わず、`critical-review` で定量 evidence を要求する
- report を閉じる場合は、代表 HLO 差分と runtime difference の両方を trace できるようにして `report-review` を通す

## Prompting Patterns

- `Spawn an explorer to map the affected files and tests before editing.`
- `Spawn docs_workflow_steward and docs_researcher in parallel, then merge their findings before editing.`
- `Use reviewer after the patch and report findings first.`
- `Use python_reviewer for Python diffs, especially when pyright warnings, type boundaries, or dtype flow changed.`
- `Use docs_completeness_reviewer and docs_consistency_reviewer after doc-heavy changes.`
- `Use worktree_health_reviewer before closing or deleting a worktree, or when scope drift is suspected.`
- `Use project_reviewer for repo-wide review, inventory, and workflow-health audits.`
- `Use experiment_planner to inspect cases, resource estimates, skip logic, and report layout.`
- `Use hlo_investigator when HLO dumps, XLA flags, or compiler behavior are part of the hypothesis.`
- `Use report_reviewer after the report draft and require a rewrite, extra validation, or rerun decision.`
- `When claims look strong, spawn docs_researcher to find caveats and contrary sources before accepting them.`

## Codex Control Points

- project-scoped config: `.codex/config.toml`
- project-scoped custom agents: `.codex/agents/*.toml`
- shared instructions: `AGENTS.md`
- shared skills: `.agents/skills/`
- running thread inspection: `/agent`
