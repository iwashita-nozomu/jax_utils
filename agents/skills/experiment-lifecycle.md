# Experiment Lifecycle

## Purpose

実験の準備、初期化、実行、結果整理、review、再実行判断を一続きの運用として扱います。

## Use This For

- experiment directory の初期化
- case 群の実行
- result / report 生成
- `critical-review` と `report-review` を挟んだ実験反復
- rerun、追加検証、report 書き直しの分岐
- 実験実行の再現手順整理

## Inputs

- 実験目的
- cases
- task 実装
- 実行時の resource estimate
- 必要なら skip や環境初期化の方針
- report draft

## Outputs

- 実行ログ
- result ディレクトリ
- report や summary
- `critical-review` と `report-review` の結果
- `report_rewrite_required`、`extra_validation_required`、`rerun_required` の判断
- 次の比較実験に必要な手順

## Role In Research-Driven Change

- この skill は `Research-Driven Change` の inner loop です。
- 外側の仮説更新、外部調査、次の change 決定は `agents/skills/research-workflow.md` が扱います。
- この skill は 1 つの protocol と 1 回の run、またはその直後の rewrite / extra validation / rerun 分岐を扱います。

## Required Review Chain

1. `experimenter` が run を完了させ、result と report draft を作る
1. `critical-review` が比較公平性、evidence sufficiency、overclaim を潰す
1. `report-review` が概要、数値、図表、結論と根拠の対応を潰す
1. `report-review` が `report_rewrite_required` を返した場合、同じ result を使って report を書き直す
1. `critical-review` または `report-review` が `extra_validation_required` を返した場合、同じ比較方針で追加検証を行う
1. `critical-review` または `report-review` が `rerun_required` を返した場合、新しい run_name で fresh rerun を行う
1. 両方の review が通るまで report を閉じない

## Implementation Surface

- `.github/skills/03-run-experiment/`
- `.github/skills/08-experiment-initialization/`
- `.github/skills/13-experiment-execution/`
- `.github/skills/14-result-validation/`

## Boundary

- 実験結果の批判的読解は `agents/skills/critical-review.md` を使います。
- user-facing report の独立レビューは `agents/skills/report-review.md` を使います。
- 長期の研究反復管理は `agents/skills/research-workflow.md` を使います。
