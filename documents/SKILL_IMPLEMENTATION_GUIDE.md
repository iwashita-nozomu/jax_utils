# Skill Implementation Guide

目的: shared canon と tool-specific implementation surface を分離したまま、skill を追加・更新するための正本を示すこと。

## Canonical Layout

- `agents/skills/`
  - skill の目的、入力、出力、使い分けを書く shared canon
- `agents/skills/catalog.yaml`
  - family と implementation path の対応表
- `.agents/skills/`
  - Codex / Copilot などの auto-discovery 向け shim
- `.codex/agents/`
  - Codex の project-scoped custom agent
- `.github/skills/`
  - GitHub Actions や CLI から直接呼ぶ実装スクリプト
- `AGENTS.md`、`CLAUDE.md`、`.github/AGENTS.md`、`.github/copilot-instructions.md`
  - 共有正本へ導く thin adapter

## Rules

- skill 説明の正本は `agents/skills/` だけに置きます。
- tool 固有ファイルに、長い skill 説明や workflow 説明を複製しません。
- GitHub 向けのスクリプトやテストは `.github/skills/` に置きます。
- auto-discovery で必要な最小説明は `.agents/skills/<name>/SKILL.md` に置きます。

## Add A New Skill

1. `agents/skills/<family>.md` を追加する。
1. `agents/skills/catalog.yaml` に family を追加する。
1. 必要なら `.agents/skills/<family>/SKILL.md` を追加する。
1. GitHub Actions や CLI から実行するなら `.github/skills/<impl>/` を追加する。
1. adapter file には入口だけを追記する。

## Update An Existing Skill

1. 先に `agents/skills/` の shared canon を更新する。
1. 次に `.agents/skills/` の shim を更新する。
1. 最後に `.github/skills/` の実装 README やスクリプトを更新する。

## What Belongs Outside The Skill Layer

- role、handoff、review loop は `agents/README.md` と `agents/COMMUNICATION_PROTOCOL.md`
- task family は `agents/TASK_WORKFLOWS.md`
- repo-wide coding rules は `documents/coding-conventions-project.md`
- experiment layout は `documents/experiment-workflow.md`
