# jax_util

JAX ベースの数値計算ユーティリティと、実験実行基盤をまとめたリポジトリです。
ライブラリ本体の実装、実験コード、実験結果、規約文書を 1 つのリポジトリで管理します。

この README は人間向けの作業入口です。AI エージェント向けの入口は `agents/README.md` です。

## 何が入っているか

- `python/jax_util`
  - ライブラリ本体です。線形作用素、ソルバー、最適化、functional 系の実装を置きます。
- `python/experiment_runner`
  - 実験を case ごとの fresh process で回すための共通基盤です。
- `python/tests`
  - テストスイートです。
- `experiments`
  - topic ごとの実験コード、run ごとの結果、run ごとの report を置きます。
- `documents`
  - 規約、設計、運用手順の正本です。
- `notes`
  - 実験をまたぐ知見、補助メモ、worktree から持ち帰る要約を置きます。
- `diary`
  - 日付単位の作業ログを置きます。
- `reports`
  - project-wide な review、automation、management report を置きます。
- `reviews`
  - 現在有効なレビュー文書だけを置きます。

## 最初に理解しておくルール

- `main` を長時間実験や大きな変更の作業場にしません。
- 新しい作業は worktree を切って進めます。
- ローカル仮想環境 (`.venv` など) は使いません。
- 依存を追加する場合は `docker/Dockerfile` と `docker/requirements.txt` を同時に更新します。
- コード変更では、対応するテストと必要な文書更新を同じ作業で入れます。
- `documents/` には正本だけを残します。履歴用の別文書は増やしません。

## ふだんの作業手順

### 1. worktree を作る

新しい作業は次のコマンドで始めます。

```bash
bash scripts/setup_worktree.sh work/<topic>-YYYYMMDD
```

例:

```bash
bash scripts/setup_worktree.sh work/solver-fix-20260402
```

このコマンドは次を行います。

- `origin/main` から branch を作ります
- `.worktrees/work-<topic>-YYYYMMDD/` に worktree を作ります
- `WORKTREE_SCOPE.md` のテンプレートをコピーします

### 2. scope を明確にする

worktree を作ったら、`WORKTREE_SCOPE.md` に少なくとも次を書きます。

- 何を直すか
- どのディレクトリを編集するか
- 実験結果をどこへ出すか
- `main` に何を持ち帰るか

### 3. 変更前にチェックを流す

軽量チェック:

```bash
make ci-quick
```

フルチェック:

```bash
make ci
```

### 4. 変更を入れる

通常のライブラリ修正では、主に次を触ります。

- `python/jax_util`
- `python/tests`
- `documents`

実験まわりの変更では、主に次を触ります。

- `experiments`
- `python/experiment_runner`
- `notes`
- 必要な `documents`

### 5. 持ち帰るものをそろえる

`main` に持ち帰るときは、少なくとも次をそろえます。

- 再生成可能なコード
- 対応するテスト
- 必要な規約・設計・README 更新
- 実験なら per-run report と最低限の result
- 長期に残す判断は `notes/` または `diary/`

## 実験の標準構成

新規実験は次の形にそろえます。

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

topic ディレクトリの必須要素は次です。

- `README.md`
- `cases.py`
- `experimentcode.py`
- `result/`

新規 experiment で topic 直下に topic 固有 Python ファイルを増やすことはしません。追加の topic 固有ロジックは `cases.py` か `experimentcode.py` に整理します。

## 実験結果の置き方

1 回の run の結果は次にそろえます。

- `experiments/<topic>/result/<run_name>/summary.json`
- `experiments/<topic>/result/<run_name>/cases.jsonl`
- `experiments/<topic>/result/<run_name>/run.log`
- 必要なら `figures/`
- 人が読む report は `experiments/report/<run_name>.md`

`run_name` は次の形式にそろえます。

```text
<topic>_<variant>_<YYYYMMDDTHHMMSSZ>
```

例:

```text
smolyak_gpu_20260402T061500Z
```

## 実験で守ること

- 1 回の run は fresh 実行で完走させます。
- 途中停止した run を resume の正本にしません。
- partial run を正式な比較結果として扱いません。
- topic ごとの実験 report を top-level `reports/` に置きません。
- 複数 run をまたぐ要約は `notes/experiments/` または `notes/themes/` に置きます。
- project-wide な report だけを `reports/` に置きます。

## どこに何を書くか

### `documents`

規約、設計、運用の正本を書きます。現在の正しいルールを残す場所です。

### `notes`

長期に残したい補助メモを書きます。実験をまたぐ知見、比較の要約、worktree の carry-over を置きます。

### `diary`

日付に紐づく作業の流れを書きます。その日に何をして、何を見て、何を決めたかを残します。

### `reports`

project-wide な automation report、management report、review report を置きます。topic ごとの experiment report は置きません。

### `reviews`

現在有効なレビュー文書だけを置きます。古いレビューや重複レビューは Git 履歴へ戻します。

## よく使うコマンド

```bash
# worktree を作る
bash scripts/setup_worktree.sh work/<topic>-YYYYMMDD

# 軽量チェック
make ci-quick

# フルチェック
make ci

# ツール一覧を見る
make tools-help
```

## 作業前に見るべき内容

まずこの README を読んだうえで、次の 3 つを見ればほぼ始められます。

- `documents/worktree-lifecycle.md`
- `documents/conventions/README.md`
- `experiments/README.md` もしくは対象 topic の `README.md`

## 詳細入口

- 規約と運用: `documents/README.md`
- 実験全体: `experiments/README.md`
- 補助メモ: `notes/README.md`
- エージェント運用: `agents/README.md`
- スクリプト一覧: `scripts/README.md`
