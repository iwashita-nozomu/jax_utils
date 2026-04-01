# Agent Communication Protocol

この文書は、恒久エージェントチームの agent-to-agent communication の正本です。

## 目的

- エージェント間の handoff を暗黙知にしない
- reviewer の指摘とその修正反映を追跡可能にする
- hidden context に依存しない受け渡しを徹底する
- disagreement や scope expansion を manager へ正しく戻す

## 基本原則

- 次の role が判断に使う情報は、必ず artifact に書く
- 口頭相当の短いやり取りで合意した内容も、判断に効くなら artifact に残す
- review のあと、対象 role は `resolved`, `rejected`, `escalated` のどれかで必ず応答する
- reviewer は実装や設計を直接直さず、指摘と要求変更を artifact に残す
- manager 以外は、scope や permission を勝手に変えない
- repo 編集は `implementer` だけが行う

## 主要な通信面

優先順位は以下です。

1. role ごとの run artifact
1. `decision_log.md`
1. `team_manifest.yaml`
1. `.github/agents/discussion.md`

使い分け:

- role ごとの作業結果、review、応答は `reports/agents/<run-id>/` の artifact に書く
- scope、acceptance criteria、specialist activation、escalation は `decision_log.md` に残す
- team shape や active roles は `team_manifest.yaml` を参照する
- `.github/agents/discussion.md` は cross-run の運用メモや、repo 全体に残す価値がある議論だけに使う

## Handoff Packet

handoff は最低限、以下の情報を含む。

- `from`
- `to`
- `stage`
- `summary`
- `requested_action`
- `artifacts`
- `repo_changes`
- `open_questions`
- `status`

推奨テンプレート:

```md
## Handoff

- from: manager
- to: designer
- stage: approved intake -> design
- summary: solver bug fix scoped to 2 files; no specialist roles required
- requested_action: produce a minimal design brief before coding
- artifacts:
  - intent_brief.md
  - management_review.md
- repo_changes:
  - none
- open_questions:
  - confirm whether API behavior is observable to users
- status: ready
```

## Review Packet

review は「何が悪いか」だけでなく、「何を直せば前に進めるか」を含む。

- `finding`
- `severity`
- `required_change`
- `evidence`
- `status`

推奨ステータス:

- `open`
- `resolved`
- `rejected`
- `escalated`

推奨テンプレート:

```md
## Findings

| Finding | Severity | Required Change | Evidence | Status |
| ------- | -------- | --------------- | -------- | ------ |
| Design couples unrelated modules | high | split the boundary and update design_brief.md | design_review.md | open |
```

## Review Response

review を受けた role は、次へ進む前に response を返す。

- `resolved`: 指摘を反映した
- `rejected`: 指摘を採用しない理由がある
- `escalated`: 自分では解けないため manager 判断が必要

推奨テンプレート:

```md
## Review Response

| Finding | Response | Evidence | Status |
| ------- | -------- | -------- | ------ |
| Design couples unrelated modules | split design into manager-facing and solver-facing paths | design_brief.md | resolved |
```

## Escalation Rules

以下では manager へ戻す。

- reviewer と execution role で合意できない
- scope 外の変更が必要
- write permission の拡張が必要
- research だけでは根拠が不足する
- schedule が現実的でない
- infra change に rollback が用意できない

escalation 時は `decision_log.md` に以下を書く。

- conflict summary
- options considered
- requested decision
- impact if delayed

## Communication Etiquette

- 要約は先に、根拠は後に書く
- blocker は 1 行目で分かるようにする
- reviewer は曖昧な「気になる」ではなく、required change を書く
- execution role は silence で流さず、必ず response を返す
- 次の role が読む artifact を明示する

## Minimal Examples

`manager -> manager_reviewer -> manager`:

- intake を書く
- review で scope / acceptance の不足を指摘する
- manager が不足を補って `resolved` にする

`designer -> design_reviewer -> designer`:

- design_brief を出す
- design_reviewer が boundary / risk を指摘する
- designer が設計を直して再提出する

`implementer -> change_reviewer -> implementer`:

- code change を出す
- change_reviewer が regression risk を指摘する
- implementer が修正と test evidence を返す

`experimenter -> experiment_reviewer -> implementer`:

- experimenter が baseline または比較 run を出す
- experiment_reviewer が fairness、overclaim、比較対象不足、再実行要否を指摘する
- implementer が必要な修正を入れ、次の実験条件を明示して experimenter へ戻す

## Repository-Wide Discussion Log

`.github/agents/discussion.md` へ書くときも、最低限以下を守る。

- role 名を明記する
- decision / concern / request を分ける
- run 固有の詳細は run artifact への参照に留める
- repo-wide に残す価値のある内容だけを書く
