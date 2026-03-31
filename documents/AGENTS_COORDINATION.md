# エージェントチーム運用の入口

目的: 恒久的なエージェントチームを、毎回同じ正本と同じ権限モデルで運用すること。

この文書は薄い入口に留めます。role 一覧、handoff、権限、workflow 詳細はここへ再掲しません。

## 正本

- チーム定義と role permission: `agents/agents_config.json`
- agent 間やり取りの規約: `agents/COMMUNICATION_PROTOCOL.md`
- runtime 実装: `scripts/agent_tools/agent_team.py`
- 人間向け要約: `agents/README.md`
- task workflow カタログ: `agents/TASK_WORKFLOWS.md`

agent 間の handoff、review、response、escalation の書き方は
`agents/COMMUNICATION_PROTOCOL.md` を参照する。

## この文書が扱うこと

- repo 運用から見た agent team の入口を示す
- 実行コマンドと CI 接続点を示す
- 正本以外へ role 定義を複製しないルールを明記する

## 実行入口

```bash
python3 scripts/agent_tools/bootstrap_agent_run.py \
  --task "<task>" \
  --owner "<owner>" \
  --workspace-root "$PWD"
```

```bash
python3 scripts/agent_tools/validate_role_write_scope.py \
  --report-dir reports/agents/<run-id> \
  --workspace-root "$PWD" \
  --report-snapshot-out /tmp/agent-report-before.json \
  --workspace-snapshot-out /tmp/agent-workspace-before.json
```

```bash
python3 scripts/agent_tools/validate_role_write_scope.py \
  --role change_reviewer \
  --report-dir reports/agents/<run-id> \
  --report-snapshot-in /tmp/agent-report-before.json \
  --workspace-snapshot-in /tmp/agent-workspace-before.json \
  --workspace-root "$PWD"
```

`pre_review.sh` では、必要なら以下で verifier の write scope も検査できる。

```bash
AGENT_REPORT_DIR=reports/agents/<run-id> \
AGENT_ROLE=verifier \
AGENT_ENFORCE_WRITE_SCOPE=1 \
bash scripts/ci/pre_review.sh
```

GitHub Actions の automation mirror は `.github/workflows/agent-coordination.yml` を参照する。

## repo 側の運用ルール

- role 定義は `agents/agents_config.json` だけで管理する。
- 人間向けの team summary は `agents/README.md` だけに置く。
- この文書と `.github/AGENTS.md` は thin entrypoint とし、role 一覧や flow を再掲しない。
- GitHub Actions では reviewer-return loop を含む handoff spine を通し、specialist role も workflow input から有効化できるようにする。
- `implementer` 以外が repo ファイルを触る workflow は作らない。
- artifact-only role を検査するときは、role 実行前の report snapshot と workspace snapshot の両方を取り、role 実行後の差分で write scope を判定する。
- `WORKTREE_SCOPE.md` がある作業では、repo 変更を editable directories に限定する。
