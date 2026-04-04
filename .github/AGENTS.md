# GitHub Agent Entry Point

このファイルは GitHub 側の薄い入口です。team shape、role 一覧、skill 説明の正本はここへ再掲しません。

## Read First

1. `AGENTS.md`
1. `agents/README.md`
1. `agents/TASK_WORKFLOWS.md`
1. `agents/skills/README.md`
1. `documents/AGENTS_COORDINATION.md`

## GitHub-Specific Surfaces

- GitHub Copilot adapter: `.github/copilot-instructions.md`
- GitHub-facing skill implementations: `.github/skills/`
- GitHub Actions automation: `.github/workflows/agent-coordination.yml`
- Repo-wide discussion log: `.github/agents/discussion.md`

## Rules

- role permission と handoff は `agents/agents_config.json` と `agents/COMMUNICATION_PROTOCOL.md` を正本にします。
- skill の用途説明は `agents/skills/` を正本にします。
- GitHub 固有ファイルには adapter と automation 補足だけを書きます。
