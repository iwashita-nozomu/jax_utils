# エージェント用ガイド

この文書は、人間と各種 AI エージェントが共通の正本へたどり着くための日本語入口です。

## 最初に見る場所

1. `AGENTS.md`
1. `agents/README.md`
1. `agents/CODEX_WORKFLOWS.md`
1. `agents/TASK_WORKFLOWS.md`
1. `agents/skills/README.md`

## 目的別の入口

### 実装を進めたい

1. `agents/README.md` で role と権限を見る
1. `agents/TASK_WORKFLOWS.md` で task family を選ぶ
1. Python 差分なら `agents/skills/python-review.md` を読む
1. `documents/coding-conventions-project.md` と対象分野の規約を読む

### 実験を回したい

1. `agents/TASK_WORKFLOWS.md` で experimenter を含む workflow を選ぶ
1. `agents/skills/experiment-lifecycle.md` を読む
1. `agents/skills/critical-review.md` と `agents/skills/report-review.md` を読む
1. `documents/experiment-workflow.md` と `documents/research-workflow.md` を読む

### レビューしたい

1. `agents/COMMUNICATION_PROTOCOL.md` で handoff / review / response を確認する
1. `agents/skills/code-review.md`、`agents/skills/python-review.md`、`agents/skills/docs-completeness-review.md`、`agents/skills/md-style-check.md`、`agents/skills/docs-consistency-review.md`、`agents/skills/worktree-health.md`、`agents/skills/critical-review.md`、`agents/skills/report-review.md`、`agents/skills/project-review.md` のどれかを選ぶ
1. `documents/REVIEW_PROCESS.md` を読む

### agent system 自体を直したい

1. `agents/README.md` を shared canon として更新する
1. `agents/CODEX_WORKFLOWS.md` と `.codex/agents/*.toml` を見直す
1. `agents/skills/README.md` と該当 skill 文書を更新する
1. その後に `AGENTS.md`、`CLAUDE.md`、`.github/AGENTS.md`、`.github/copilot-instructions.md` を追随させる

## Tool ごとの入口

- primary runtime: Codex
- 共通入口: `AGENTS.md`
- Codex subagent config: `.codex/config.toml` と `.codex/agents/`
- Claude: `CLAUDE.md`
- GitHub Copilot: `.github/copilot-instructions.md`
- auto-discovery される skill shim: `.agents/skills/`

## 置き場のルール

- shared canon は `agents/` に置く
- Codex の custom subagent は `.codex/agents/` に置く
- skill の説明正本は `agents/skills/` に置く
- tool 固有の補足は adapter にだけ置く
- 実装用スクリプトや GitHub Actions 連携は `.github/skills/` と `.github/workflows/` に置く
