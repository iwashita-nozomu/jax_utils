# Project Review

## Purpose

repo 全体を横断して、構成、文書、skills、ツール、静的健全性、運用 drift をまとめてレビューします。

## Use This For

- repo-wide な棚卸し
- workflow、agent system、tools、docs をまたぐ全体レビュー
- 大きな整理や統合変更の完了判定
- stale 文書、重複定義、未参照資産、運用 drift の検出
- project health の定点確認

## Must Read Before Reviewing

- `agents/skills/comprehensive-review.md`
- `agents/skills/project-health.md`
- scope に research / experiment が入る場合は `agents/skills/research-perspective-review.md`
- `agents/CODEX_WORKFLOWS.md`
- `documents/REVIEW_PROCESS.md`
- 必要なら `documents/AGENTS_COORDINATION.md`
- 必要なら `documents/coding-conventions-project.md`

## Inputs

- review scope
- 対象ディレクトリ
- 必要なら phase 指定
- 必要なら前回 review の結果

## Outputs

- findings-first の project-wide review
- `inventory`, `static-health`, `workflow-health`, `research-evidence-health`, `tooling-health`, `follow-up` の各 section
- `fix now`, `follow-up`, `delete-ok` の切り分け

## Mandatory Phases

1. `Inventory`
   - 何が現役で、何が stale で、何が正本かを確認する
1. `Static Health`
   - `pyright`、`ruff`、リンク整合、machine-readable config の破綻を確認する
   - Markdown 変更がある場合は `md-style-check` を含める
   - Python 変更がある場合は `python-review` の観点を含める
1. `Workflow Health`
   - routing、skill canon、subagent、adapter の導線が揃っているか確認する
   - 文書整理を伴う場合は `docs-completeness-review` と `docs-consistency-review` を含める
1. `Research Evidence Health`
   - scope に experiments、reports、benchmark protocol、research docs が入る場合は `research-perspective-review` を含める
   - reproducibility、scientific computing、benchmark、artifact、FAIR data、ML-science reporting の観点を別々に確認する
1. `Tooling Health`
   - scripts、CI、GitHub implementation surface、runtime helper の破綻を確認する
1. `Worktree Health`
   - worktree を使う作業では `WORKTREE_SCOPE.md`、scope drift、cleanup risk を確認する
1. `Follow-Up Decision`
   - 直す、削除する、notes に吸収する、監視だけ続ける、のどれかに分類する

## Mandatory Checklist

- 正本と adapter の責務が混ざっていない
- shared canon と machine-readable config が矛盾していない
- stale file、duplicate file、unreferenced file が残っていないか確認した
- review 対象に応じた `pyright`、`ruff`、整合 check を見た
- workflow 改造後に routing と subagent 導線を追った
- research / experiment scope では perspective review pack を通し、観点ごとの差分を分けて記録した
- tool 実装面が shared canon から取り残されていない
- worktree を使う作業では scope drift と cleanup risk を確認した
- health issue と one-off cleanup を混同していない
- findings を `fix now`、`follow-up`、`delete-ok` に分けた

## Default Placement In Workflow

- repo 全体レビュー、棚卸し、workflow 改造後の完了判定ではこれを最上位に使います。
- `comprehensive-review` と `project-health` はこの workflow の下位 review として使います。

## Boundary

- 局所 diff のレビューだけなら `agents/skills/code-review.md` を使います。
- Python 差分の静的解析中心レビューなら `agents/skills/python-review.md` を使います。
- 実験 result や report の妥当性は `critical-review` と `report-review` を使います。

## Implementation Surface

- `.github/skills/20-project-review/`
- `.github/skills/06-comprehensive-review/`
- `.github/skills/17-project-health/`
