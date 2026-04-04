# Shared Skill Canon

このディレクトリは、Codex、Claude、GitHub Copilot などで共有する skill 文書の正本です。

## Rules

- skill の目的、入力、出力、使い分けは `agents/skills/` にだけ書きます。
- tool 固有の入口は shared canon を参照し、長い説明を複製しません。
- `.agents/skills/` は auto-discovery 用 shim です。
- `.github/skills/` は GitHub Actions や CLI から呼ぶ実装面です。
- 新しい skill を追加するときは、`catalog.yaml` と該当 skill 文書を同時に更新します。

## Skill Families

| Family | Purpose | Canonical Doc | Implementation Surface |
| ------ | ------- | ------------- | ---------------------- |
| `static-check` | 速い検査と基礎品質確認 | `agents/skills/static-check.md` | `.github/skills/01-static-check/`, `.github/skills/09-type-checking/`, `.github/skills/10-documentation-validation/`, `.github/skills/11-test-execution/`, `.github/skills/15-docker-environment/` |
| `code-review` | 変更レビューと設計妥当性確認 | `agents/skills/code-review.md` | `.github/skills/02-code-review/`, `.github/skills/12-code-review-refactoring/` |
| `python-review` | Python 差分の静的解析・型追跡レビュー | `agents/skills/python-review.md` | `.github/skills/19-python-review/` |
| `docs-completeness-review` | 文書の欠落や説明不足のレビュー | `agents/skills/docs-completeness-review.md` | `.github/skills/21-docs-completeness-review/` |
| `md-style-check` | Markdown の体裁と lint の確認 | `agents/skills/md-style-check.md` | `.github/skills/22-md-style-check/`, `.github/skills/10-documentation-validation/` |
| `docs-consistency-review` | 文書間の矛盾と曖昧性のレビュー | `agents/skills/docs-consistency-review.md` | `.github/skills/23-docs-consistency-review/`, `.github/skills/10-documentation-validation/` |
| `worktree-health` | worktree の scope と健全性の確認 | `agents/skills/worktree-health.md` | `.github/skills/24-worktree-health/` |
| `experiment-lifecycle` | 単一 run とその review / rerun 分岐 | `agents/skills/experiment-lifecycle.md` | `.github/skills/03-run-experiment/`, `.github/skills/08-experiment-initialization/`, `.github/skills/13-experiment-execution/`, `.github/skills/14-result-validation/` |
| `critical-review` | 実験結果や主張の批判的評価 | `agents/skills/critical-review.md` | `.github/skills/04-critical-review/` |
| `report-review` | ユーザー向け実験レポートの独立レビュー | `agents/skills/report-review.md` | `.github/skills/18-report-review/` |
| `research-workflow` | 外部調査、実装、実験反復、仮説更新を含む outer loop | `agents/skills/research-workflow.md` | `.github/skills/05-research-workflow/` |
| `research-perspective-review` | 研究系変更を複数の独立視点で並列レビュー | `agents/skills/research-perspective-review.md` | `.github/skills/25-research-perspective-review/` |
| `comprehensive-review` | repo 全体の横断レビュー | `agents/skills/comprehensive-review.md` | `.github/skills/06-comprehensive-review/` |
| `project-health` | 継続運用、監視、CI 健全性 | `agents/skills/project-health.md` | `.github/skills/07-health-monitor/`, `.github/skills/16-ci-cd-integration/`, `.github/skills/17-project-health/` |
| `project-review` | repo-wide な棚卸しと全体レビュー | `agents/skills/project-review.md` | `.github/skills/20-project-review/`, `.github/skills/06-comprehensive-review/`, `.github/skills/17-project-health/` |

## Tool Adapters

- Shared entrypoint: `AGENTS.md`
- Claude adapter: `CLAUDE.md`
- GitHub Copilot adapter: `.github/copilot-instructions.md`
- GitHub entrypoint: `.github/AGENTS.md`
- Codex / Copilot skill shim: `.agents/skills/*/SKILL.md`

## Adding Or Updating A Skill

1. 共有の目的と境界を `agents/skills/<family>.md` に書く。
1. `agents/skills/catalog.yaml` を更新する。
1. 必要なら `.agents/skills/<family>/SKILL.md` を更新する。
1. GitHub 向け実装や CI から使う場合は `.github/skills/` 配下の実装を更新する。
1. `AGENTS.md` などの adapter に大きな説明を増やさない。
