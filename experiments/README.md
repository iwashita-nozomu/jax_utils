# Experiment Hub

`experiments/` は、実験コード、実行ごとの生成物、実行ごとのレポートをまとめる入口です。

## 目的

- 実験ごとの目的とコード配置を 1 か所で辿れるようにする。
- 実行結果を `experiments/<topic>/result/<run_name>/` に閉じ込めて管理する。
- 1 回の実験に対する report を `experiments/report/` にまとめる。

## 標準レイアウト

```text
experiments/
├── README.md
├── report/
│   ├── README.md
│   └── <run_name>.md
└── <topic>/
    ├── README.md
    ├── cases.py
    ├── experimentcode.py
    └── result/
        └── <run_name>/
```

## 各 topic で README に書くこと

- 実験目的
- `cases.py` と `experimentcode.py` の役割
- 実行コマンド
- `result/<run_name>/` に何が出るか
- 対応する report の場所
- 関連ドキュメントの入口

## 置き場の方針

- 実行ごとの機械生成物は `experiments/<topic>/result/<run_name>/` に置きます。
- 1 回の実験を読むための Markdown report は `experiments/report/<run_name>.md` に置きます。
- 複数 run をまたいだ振り返りや知見整理は `notes/experiments/` や `notes/themes/` に置きます。
- top-level `reports/` は project-wide な review や automation report に使い、topic ごとの実験 report の正本にはしません。

## 禁止事項

- 新規 experiment で `experiments/<topic>/` 直下に `cases.py` と `experimentcode.py` 以外の topic 固有 Python ファイルを増やすことを禁止します。
- 新規 experiment で結果を `experiments/<topic>/result/<run_name>/` 以外へ散らすことを禁止します。
- 1 回の実験 report を top-level `reports/` や `notes/experiments/` に置くことを禁止します。
- 新規 experiment で legacy な `results/` レイアウトを採用することを禁止します。

## branch / worktree

- branch は既定では分けません。同じ branch 上で実験コード、結果、report を扱います。
- 別 branch / worktree の使用は、長時間 run の隔離、巨大生成物の退避、破壊的な試行の切り分けに限って許可します。

## 関連

- [documents/experiment-workflow.md](/workspace/documents/experiment-workflow.md)
- [documents/coding-conventions-experiments.md](/workspace/documents/coding-conventions-experiments.md)
- [documents/experiment-report-style.md](/workspace/documents/experiment-report-style.md)
- [report/README.md](/workspace/experiments/report/README.md)
