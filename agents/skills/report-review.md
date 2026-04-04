# Report Review

## Purpose

ユーザーに見せる実験レポートが、根拠に即していて、数値と図表が読めて、誤解を生まない形になっているかを独立に確認します。

## Use This For

- `experiments/report/<run_name>.md` の完成前レビュー
- `Abstract`、`Results`、`Discussion`、`Conclusion` の reader-facing quality 確認
- 数値、table、figure、caption の根拠追跡確認
- evidence は足りているが report の書き方が弱いケースの書き直し判断
- evidence 自体が足りないケースの追加検証 / rerun 判定

## Must Read Before Reviewing

- `documents/experiment-report-style.md`
- `documents/experiment-critical-review.md`
- `agents/templates/experiment_report.md`
- 対象の report draft
- 対応する `summary.json`、`cases.jsonl`、主要 figure / table
- 既にあれば `experiment_review.md`

## Inputs

- report draft
- `summary.json`
- `cases.jsonl`
- figure / table
- 比較対象と protocol の要約
- `critical-review` の findings

## Outputs

- findings-first の `report_review.md`
- `approved`、`report_rewrite_required`、`extra_validation_required`、`rerun_required` のいずれか
- report だけで直る問題と、evidence を増やす必要がある問題の切り分け

## Mandatory Checklist

- 実験の概要、問い、比較対象、protocol が report 冒頭で分かる
- `Abstract` が strongest result を数値つきで述べ、scope も明記する
- headline number の近くに case 数、success / failure 数、failure kind がある
- 各 major claim が figure または table を明示的に参照する
- 図表が単体で読める。軸名、単位、scale、legend、caption が欠けない
- `Results` が観測事実を述べ、`Discussion` が解釈を述べる
- limitations と missing evidence が report 内に残っている
- user-facing wording が evidence より強くなっていない
- summary table と representative figure の両方で主要主張を支えている

## Decision Rules

- `report_rewrite_required`
  - 根拠は足りているが、説明順、数値の見せ方、図表導線、結論の書き方が弱い
- `extra_validation_required`
  - 同じコードと同じ比較方針のまま、追加 table、追加 figure、追加 narrow run で足りる
- `rerun_required`
  - 比較条件の破綻、case set 不一致、partial run 混入、条件変更、protocol 汚染がある
- `approved`
  - reader-facing な report として閉じてよい

## Response Order

1. `rerun_required`
   - report の書き直しを止めます。
   - protocol、比較条件、partial run 混入を直し、新しい run_name で fresh rerun します。
1. `extra_validation_required`
   - rerun までは不要でも、追加 figure、追加 table、追加 narrow run を先に足します。
   - この段階では report の wording fix だけで閉じません。
1. `report_rewrite_required`
   - evidence が揃ってから、report の構成、抽象、図表導線、結論文を直します。
1. `approved`
   - report を閉じてよいです。

## Default Placement In Workflow

- `critical-review` の後に使います。
- `experiments/report/<run_name>.md` を閉じる前に必ず使います。
- `report-review` が `extra_validation_required` か `rerun_required` を返した場合、report を閉じません。

## Boundary

- 比較公平性、evidence sufficiency、overclaim の中核判定は `critical-review` と連携して扱います。
- `report-review` は reader-facing な構成、数値の根拠、図表の見せ方、結論と根拠の対応を主担当にします。
- ただし review 中に evidence 欠落を見つけた場合は、追加検証や rerun を要求します。

## Implementation Surface

- `.github/skills/18-report-review/`
