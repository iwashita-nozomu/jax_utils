# Smolyak Scaling Experiments

`SmolyakIntegrator` の計算規模探索とベンチマークを、テストとは分離して行うための実験置き場です。

結果 JSON を載せるブランチ名は `results/functional-smolyak-scaling` を想定しています。

結果メモと branch 対応は [notes/README.md](/workspace/experiments/functional/smolyak_scaling/notes/README.md) に残します。

## What It Does

- 次元レンジとレベルレンジからケース列を生成します。
- 各ケースは fresh な subprocess で実行します。
- GPU 実行では物理 GPU ごとに 1 worker を立てて並列に流します。
- 同じ GPU では、前ケースのプロセス終了後に次ケースを開始します。
- 各ケースは fresh process で実行し、積分器は CPU で初期化し、その後に対象デバイスへ転送します。
- 同じ積分問題を `fori_loop` で `100` 回解いて、平均積分時間を計測します。
- `SmolyakIntegrator(dtype=...)` を切り替えて複数の float 精度を比較できます。
- 各ケースについて、誤差平均、誤差分散、点数、保持サイズ、デバイスメモリ統計、RSS、積分器初期化時間、転送時間、実行時間を JSON に保存します。
- 各ケースは終了時に JSONL へ 1 行ずつ追記されるので、途中停止しても部分結果を回収できます。
- GPU 実験では `XLA_PYTHON_CLIENT_PREALLOCATE=false` を設定し、GPU メモリの先取りを無効化します。
- 実験全体について、`git_branch`、`git_commit`、`results_branch`、`worktree_path`、`script_path`、実行条件レンジをトップレベル JSON に保存します。
- 実験後は `render_smolyak_scaling_report.py` で SVG/HTML のレポートを生成できます。

## Files

- `run_smolyak_scaling.py`
  - レンジ指定ベースのベンチマーク実行スクリプトです。
- `render_smolyak_scaling_report.py`
  - 結果 JSON から、誤差・時間・メモリ・failure kind・frontier をまとめた可視化レポートを生成するスクリプトです。
- `results/`
  - 実行結果を保存するディレクトリです。
  - `<run>.jsonl` は case 単位の逐次保存結果です。

## Usage

CPU で小さめのレンジを確認する例です。

```bash
PYTHONPATH=/workspace/python python3 /workspace/experiments/functional/smolyak_scaling/run_smolyak_scaling.py \
  --platform cpu \
  --dimensions 8:16:4 \
  --levels 4:5 \
  --dtypes all \
  --num-accuracy-problems 9
```

3 GPU を使って広めのレンジを並列に流す例です。

```bash
PYTHONPATH=/workspace/python python3 /workspace/experiments/functional/smolyak_scaling/run_smolyak_scaling.py \
  --platform gpu \
  --gpu-indices 0,1,2 \
  --dimensions 8:48:4 \
  --levels 4:8 \
  --dtypes float16,bfloat16,float32,float64 \
  --num-accuracy-problems 9
```

時間計測の反復回数や係数レンジも変えられます。

```bash
PYTHONPATH=/workspace/python python3 /workspace/experiments/functional/smolyak_scaling/run_smolyak_scaling.py \
  --platform gpu \
  --gpu-indices 0,1,2 \
  --dimensions 8:64:4 \
  --levels 4:10 \
  --num-repeats 100 \
  --num-accuracy-problems 9 \
  --coeff-start -0.55 \
  --coeff-stop 0.65
```

実験後に結果レポートを作る例です。

```bash
python3 /workspace/experiments/functional/smolyak_scaling/render_smolyak_scaling_report.py \
  --input /workspace/experiments/functional/smolyak_scaling/results/latest.json
```
