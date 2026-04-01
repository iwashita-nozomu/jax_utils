# 実験ディレクトリ構成

この章は、このリポジトリで experiment をどこへ置くかを定めます。
研究の問い、数式、比較設計、段階的改造の手順は `documents/research-workflow.md` を正本とします。

## 要約

- 再利用できる汎用 runtime は `python/experiment_runner/` に置きます。
- topic 固有のケース生成、設定、集計、実行スクリプトは `experiments/` 配下に置きます。
- 長時間実行の生成物は topic ごとの `results/` に集約し、ライブラリ本体へ混ぜません。
- experiment run は 1 回の fresh 実行で完走させ、途中停止 run を継ぎ足して完了扱いにしません。

## 規約

### 1. 役割分担

- `python/experiment_runner/` は、topic 非依存の実験実行基盤を置く場所とします。
- ここには、subprocess 実行、resource scheduling、GPU 環境変数の受け渡し、context の整形のような汎用機能だけを置きます。
- 特定 topic に閉じたケース生成、テスト関数選択、結果集計、CLI 既定値は `experiments/` 側へ置きます。

### 2. topic ごとの配置

- 単独で完結する experiment は `experiments/<topic>/` に置きます。
- 同じ area の下で複数 experiment を並べる必要がある場合だけ `experiments/<area>/<topic>/` を使います。
- `experiments/` 配下の topic ディレクトリには、少なくとも README、実行スクリプト、集計または可視化スクリプト、`results/` を置ける形にします。

### 3. 推奨レイアウト

```text
experiments/smolyak_experiment/
├── README.md
├── __init__.py
├── cases.py
├── runner_config.py
├── results_aggregator.py
├── run_smolyak_experiment_simple.py
└── results/
```

```text
experiments/functional/smolyak_scaling/
├── README.md
├── run_smolyak_scaling.py
├── render_smolyak_scaling_report.py
└── results/
```

### 4. どこへ何を置くか

- topic をまたいで再利用する protocol や scheduler は `python/experiment_runner/` に置きます。
- その topic のためだけに存在する `cases.py`、`runner_config.py`、`results_aggregator.py` は `experiments/<topic>/` に置きます。
- 可視化や report rendering は、topic 固有なら experiment ディレクトリに同居させます。
- topic 固有ディレクトリの README や note から、定式化と比較対象を必ず辿れるようにします。
- 長時間実行で生成される JSON、JSONL、HTML、SVG、ログは `results/` に集約します。
- JSONL は run 中の progress 記録として扱い、resume 用の canonical input にはしません。
- 生成物は `.gitignore` と results branch 運用で管理し、安定ライブラリやテストディレクトリへ混ぜません。

### 5. テスト

- 汎用 runtime のテストは `python/tests/experiment_runner/` に置きます。
- topic 固有 helper のうち再利用価値が高いものは、対応する unit test を `python/tests/` 側へ追加して構いません。
- 一回限りの実行スクリプトは、README の smoke command と集計結果の形式保証を優先し、無理にライブラリ級の API にしません。
- 実行スクリプトは、指定した条件を 1 回の invocation で完走する責務を持たせます。
- 途中停止した場合は、同じ結果ファイルへ継ぎ足さず、新しい run_id と出力先で 0 から再実行します。

### 6. `main` への統合

- worktree から `main` へ持ち帰るときは、コードだけでなく test と document を同時に統合します。
- worktree 側で削除された冗長ファイルは、`main` 側でも不要なら削除します。
- 実行結果そのものは raw のまま全部を `main` へ戻さず、必要な最終 JSON と note を残します。
- `main` へ持ち帰るのは完走 run の final JSON を原則とし、partial run は診断用の branch artifact に留めます。

## 補足

- `python/experiment/` という別の top-level package は、現時点では新設しません。
- 現在の repo では、汎用層はすでに `python/experiment_runner/` にあるため、そこへ責務を集約します。

## 更新手順

- benchmark の置き場と境界を変えた場合は [20_benchmark_policy.md](./20_benchmark_policy.md) を同時に更新します。
- worktree 運用や results branch 運用を変えた場合は [documents/coding-conventions-experiments.md](../../coding-conventions-experiments.md) も更新します。
