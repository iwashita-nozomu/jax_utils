# Agent Team — Entry Point

このファイルは GitHub 上の入口です。team shape や role 一覧はここへ再掲しません。

## Canonical Sources

- Team definition and write policy: `agents/agents_config.json`
- Communication protocol: `agents/COMMUNICATION_PROTOCOL.md`
- Human-facing team summary: `agents/README.md`
- Runtime implementation: `scripts/agent_tools/agent_team.py`
- Repo integration guide: `documents/AGENTS_COORDINATION.md`

## GitHub-Side Rules

- Team summary は `agents/README.md` を読む。
- role permission と handoff は `agents/agents_config.json` と `agents/COMMUNICATION_PROTOCOL.md` を正本にする。
- GitHub Actions の automation mirror は `.github/workflows/agent-coordination.yml` を使う。
- このファイルへ role 一覧や flow の説明を複製しない。
