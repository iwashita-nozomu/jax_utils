# Smolyak Scaling Experiments

`SmolyakIntegrator` の計算規模探索とベンチマークを、テストとは分離して行うための実験置き場です。

従来は専用 branch に結果を逃がす運用も想定していましたが、新標準では branch 分離は必須ではありません。

結果メモと横断的な要約は [notes/README.md](/workspace/notes/README.md) に残します。

## What It Does

- 次元レンジとレベルレンジからケース列を生成します。
- 各ケースは `StandardRunner` により fresh child process で実行します。
- GPU 実行では `StandardFullResourceScheduler` が物理 GPU と GPU slot を割り当てて並列に流します。
- 同じ GPU では、割り当て可能な slot が空いたときだけ次ケースを開始します。
- 各ケースは fresh process で実行し、積分器は CPU で初期化し、その後に対象デバイスへ転送します。
- 同じ積分問題を `fori_loop` で `100` 回解いて、平均積分時間を計測します。
- `SmolyakIntegrator(dtype=...)` を切り替えて複数の float 精度を比較できます。
- 各ケースについて、誤差平均、誤差分散、点数、保持サイズ、デバイスメモリ統計、RSS、積分器初期化時間、転送時間、実行時間を JSON に保存します。
- 各ケースは終了時に JSONL へ 1 行ずつ追記され、run 中の progress と failure 診断を残します。
- child 側の Python 例外は worker が詳細 record を JSONL へ書いてから再送出し、timeout / native failure / completion 欠落は parent 側が補完 record を JSONL へ書きます。
- GPU 実験では `jax_util.xla_env` 経由で `XLA_PYTHON_CLIENT_PREALLOCATE=false` を設定し、GPU メモリの先取りを無効化します。
- 実験全体について、`git_branch`、`git_commit`、`results_branch`、`worktree_path`、`script_path`、実行条件レンジをトップレベル JSON に保存します。
- 実験後は `render_smolyak_scaling_report.py` で SVG/HTML のレポートを生成できます。

## Files

- `run_smolyak_scaling.py`
  - レンジ指定ベースのベンチマーク実行スクリプトです。
  - `StandardRunner` / `StandardFullResourceScheduler` を使う単一路線です。
- `render_smolyak_scaling_report.py`
  - 結果 JSON から、誤差・時間・メモリ・failure kind・frontier をまとめた可視化レポートを生成するスクリプトです。
- `python/experiment_runner/`
  - fresh child process 実行、resource scheduling、monitor、JSONL helper の共通部品です。
- `python/tests/experiment_runner/test_runner_gpu.py`
  - 2 GPU を使う fresh child process scheduler smoke test です。
  - child は数秒間 matmul を回すので、`nvidia-smi` でも観測しやすい負荷になります。
- `results/`
  - 実行結果を保存するディレクトリです。
  - `<run>.jsonl` は case 単位の逐次保存結果です。
  - 途中停止した場合でも同じ `<run>.jsonl` へ継ぎ足さず、新しい run を 0 からやり直します。

## Layout Note

- この topic は簡素化前の layout で、`results/` と補助 script を持っています。
- 新規 experiment の標準構成は [experiments/README.md](/workspace/experiments/README.md) を参照してください。
- 1 回の run に対する Markdown report の正本は [experiments/report/](/workspace/experiments/report/README.md) に置きます。
- 新規 experiment でこの legacy layout を再利用することを禁止します。

## Usage

CPU で小さめのレンジを確認する例です。

````bash
PYTHONPATH=/workspace/python python3 /workspace/experiments/functional/smolyak_scaling/run_smolyak_scaling.py \
  --platform cpu \
  --dimensions 8:16:4 \
  --levels 4:5 \
  --dtypes all \
  --num-accuracy-problems 9
```text

3 GPU を使って広めのレンジを並列に流す例です。

```bash
PYTHONPATH=/workspace/python python3 /workspace/experiments/functional/smolyak_scaling/run_smolyak_scaling.py \
  --platform gpu \
  --gpu-indices 0,1,2 \
  --dimensions 8:48:4 \
  --levels 4:8 \
  --dtypes float16,bfloat16,float32,float64 \
  --num-accuracy-problems 9
```text

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
```text

実験後に結果レポートを作る例です。

```bash
python3 /workspace/experiments/functional/smolyak_scaling/render_smolyak_scaling_report.py \
  --input /workspace/experiments/functional/smolyak_scaling/results/latest.json
```text

## Run Policy

- 実験は 1 回の fresh 実行で指定レンジを完走させます。
- JSONL は progress 記録であり、resume のための canonical input ではありません。
- 途中で止まった run を診断用に残すことは許可します。ただし、正規結果として継続することを禁止します。
- 再実行するときは、新しい出力先で 0 から実行します。
- naming rule や出力先を変えた場合は、この README と `documents/coding-conventions-experiments.md` を同時に更新します。
````
