# Skill: `report-review` — ユーザー向け実験レポートレビュー

このディレクトリは GitHub 側の implementation surface です。skill の説明正本は `agents/skills/report-review.md` に置きます。

## Read First

1. `agents/skills/report-review.md`
1. `documents/experiment-report-style.md`
1. `documents/experiment-critical-review.md`
1. 対象の `experiments/report/<run_name>.md`

## Review Focus

- 実験の概要が冒頭で分かるか
- `Abstract` が strongest result を numbers つきで述べるか
- headline metrics に case 数、success / failure、failure kind が添えられているか
- 各 major claim が figure または table を参照しているか
- 図表に軸名、単位、scale、legend、caption があるか
- report rewrite だけで直るか、追加検証や rerun が必要か
