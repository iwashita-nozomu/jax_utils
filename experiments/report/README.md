# Experiment Reports

このディレクトリには、1 回の実験 run に対応する Markdown report を置きます。

## ルール

- 1 report は 1 run に対応させます。
- ファイル名は `<run_name>.md` に固定します。
- 対応する結果ディレクトリは `experiments/<topic>/result/<run_name>/` とします。
- report からは少なくとも `summary.json`、`cases.jsonl`、主要な図やログの場所を辿れるようにします。

## 最低限書くこと

- `Question`
- `Protocol`
- `Result Summary`
- `Interpretation`
- `Limitations`
- `Artifacts`

複数 run をまたいだまとめや長期的な知見は、`notes/experiments/` または `notes/themes/` に分けて残します。
