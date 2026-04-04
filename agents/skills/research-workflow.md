# Research Workflow

## Purpose

単発の run ではなく、比較、反復、仮説更新を含む長期研究フローを整理します。

## Use This For

- baseline と改造案の比較設計
- 外部調査つき実装や性能改善の反復 loop
- 反復ごとの branch / note / report 整理
- 次の仮説への引き継ぎ
- 複数 run をまたぐ知見の吸い上げ

## Must Read Before Working

- `agents/skills/experiment-lifecycle.md`
- `agents/skills/critical-review.md`
- `agents/skills/report-review.md`
- `documents/research-workflow.md`
- `documents/experiment-workflow.md`

## Inputs

- 研究課題
- baseline
- 比較計画
- exit criteria
- 既存 notes / reports

## Outputs

- 反復計画
- 比較表
- 次の実験条件
- notes への知見吸収
- 現在の loop decision

## Standard Outer Loop

1. `Question`、比較対象、exit criteria を固定する
1. 外部調査を行い、採用候補と反証候補を整理する
1. 比較プロトコル、metrics、resource estimate、run layout を固定する
1. baseline または current state を `experiment-lifecycle` で記録する
1. `implementer` が 1 つの change を入れる
1. 同じ protocol で run し、result と report draft を作る
1. `critical-review` で evidence sufficiency と overclaim を潰す
1. `report-review` で reader-facing な report を潰す
1. decision に応じて loop を分岐する

## Loop Decisions

- `report_rewrite_required`
  - 同じ result を使って report だけを書き直し、`report-review` へ戻る
- `extra_validation_required`
  - 同じ仮説のまま追加 case、追加集計、追加 figure を行い、`critical-review` へ戻る
- `rerun_required`
  - fresh `run_name` で rerun する。必要なら code か protocol を修正して loop 先頭へ戻る
- `approved`
  - exit criteria を満たしていれば loop を閉じる。満たしていなければ次の 1 change を設計する

## Required Records

- 現在の仮説
- 比較対象と fairness rule
- どの review が loop を戻したか
- `report_rewrite_required`、`extra_validation_required`、`rerun_required`、`approved` の最終判断
- 次の変更案か、終了判断

## Niche Subflow: HLO Analysis And Compiler-Tuning

次のような依頼では、この subflow を使います。

- HLO dump を比較して bottleneck を探したい
- `XLA_*` や `JAX_PLATFORMS` の方針変更を検証したい
- compiler behavior の変化を code change と一緒に追いたい

手順は次です。

1. 対象関数、case、backend 条件、run_name、dump path を固定する
1. baseline と change 後で、同じ protocol の HLO dump を採取する
1. `scripts/hlo/summarize_hlo_jsonl.py` で差分を集計する
1. `jax_util.xla_env` と child initializer を使い、env 初期化点を固定する
1. code change か flag change のどちらか 1 種類だけを変える
1. HLO 差分、runtime metric、failure kind を同じ比較表で扱う
1. HLO の見た目ではなく、`critical-review` で定量 evidence を要求する
1. user-facing report を閉じる場合は、代表 HLO 差分と実測差分を両方残す

## Implementation Surface

- `.github/skills/05-research-workflow/`

## Boundary

- 単発の実験実行と 1 run 内の分岐は `agents/skills/experiment-lifecycle.md` を使います。
- 実装差分のレビューは `agents/skills/code-review.md` を使います。
