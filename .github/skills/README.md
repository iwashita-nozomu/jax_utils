# GitHub Skill Implementation Surface

このディレクトリは、GitHub Actions と GitHub 側 CLI から呼ぶ skill 実装面です。
skill の説明正本は `agents/skills/` に置きます。

## Read First

1. `agents/skills/README.md`
1. `agents/skills/catalog.yaml`
1. 必要な family に対応する `agents/skills/*.md`

## What Lives Here

- GitHub から直接呼ぶ Python スクリプト
- GitHub Actions で使う実装入口
- 実装詳細に近い README
- family ごとの narrow automation helper

## Family Mapping

| Shared Family | GitHub Implementation Paths |
| ------------- | --------------------------- |
| `static-check` | `01-static-check/`, `09-type-checking/`, `10-documentation-validation/`, `11-test-execution/`, `15-docker-environment/` |
| `code-review` | `02-code-review/`, `12-code-review-refactoring/` |
| `python-review` | `19-python-review/` |
| `docs-completeness-review` | `21-docs-completeness-review/` |
| `md-style-check` | `22-md-style-check/`, `10-documentation-validation/` |
| `docs-consistency-review` | `23-docs-consistency-review/`, `10-documentation-validation/` |
| `worktree-health` | `24-worktree-health/` |
| `experiment-lifecycle` | `03-run-experiment/`, `08-experiment-initialization/`, `13-experiment-execution/`, `14-result-validation/` |
| `critical-review` | `04-critical-review/` |
| `report-review` | `18-report-review/` |
| `research-workflow` | `05-research-workflow/` |
| `research-perspective-review` | `25-research-perspective-review/` |
| `comprehensive-review` | `06-comprehensive-review/` |
| `project-health` | `07-health-monitor/`, `16-ci-cd-integration/`, `17-project-health/` |
| `project-review` | `20-project-review/`, `06-comprehensive-review/`, `17-project-health/` |

## Maintenance Rules

- shared description をこの README や adapter file に複製しません。
- family を追加したら、先に `agents/skills/` を更新します。
- GitHub Actions から呼ぶスクリプト名や入出力だけをこの層で管理します。

## Related Docs

- `documents/SKILL_IMPLEMENTATION_GUIDE.md`
- `documents/experiment-workflow.md`
- `documents/experiment-critical-review.md`
- `documents/REVIEW_PROCESS.md`
- `documents/research-workflow.md`
