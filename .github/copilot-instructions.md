# GitHub Copilot Instructions

## Read First

1. `AGENTS.md`
1. `agents/README.md`
1. `agents/TASK_WORKFLOWS.md`
1. `agents/skills/README.md`

## Copilot-Specific Notes

- 日本語で作業してください。
- コメントは必要な箇所にだけ丁寧に書いてください。
- `./python` と `./jupyter` を主対象にします。
- 複雑な分岐より、単純で保守しやすい構造を優先してください。

## Repo Rules

- 規約は `documents/coding-conventions-project.md` を正本にします。
- 仮想環境の新設は禁止です。Docker と `docker/requirements.txt` を正本にします。
- skill の用途説明は `agents/skills/` を正本にします。
- GitHub 固有の実装スクリプトは `.github/skills/` にあります。

## Skill Entry Points

- Shared canon: `agents/skills/README.md`
- GitHub implementation surface: `.github/skills/README.md`
- Auto-discovery shim: `.agents/skills/`
