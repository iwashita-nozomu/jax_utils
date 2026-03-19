# Smolyak HLO Analysis

単一の Smolyak 積分ケースについて、既存の `jax_util.hlo.dump_hlo_jsonl` を使って HLO を保存し、`scripts/hlo/summarize_hlo_jsonl.py` で集計する実験ディレクトリです。

この実験の目的は、まず 1 ケースの lowering を丁寧に見て、

- `while` や `call` が多いか
- `gather` や `slice` などのデータ移動が支配的か
- `dot` や `multiply` が主役か
- HLO text / proto / compiled code size がどれくらい膨らむか

を把握し、Smolyak 積分器のボトルネック候補を洗い出すことです。

## 実行例

```bash
PYTHONPATH=/workspace/.worktrees/work-smolyak-tuning-20260316/python \
python3 /workspace/.worktrees/work-smolyak-tuning-20260316/experiments/functional/smolyak_hlo/run_smolyak_hlo_case.py \
  --platform cpu \
  --dimension 4 \
  --level 3 \
  --dtype float32 \
  --num-repeats 16
```text

## 出力

- `<run>.jsonl`
  - `dump_hlo_jsonl` が吐く生の HLO JSONL
- `<run>_summary.json`
  - `summarize_hlo_jsonl.py` の集計結果
  - `preferred_text.utf8_bytes`, `hlo_proto_bytes`, `generated_code_size_in_bytes`, `temp_size_in_bytes` などの size 集計を含む
- `<run>.json`
  - 実験条件、Git 情報、summary、簡易ヒント
  - `size_digest` として重要な size 指標の最大値も保持する
- `latest.json`
  - 直近 run の JSON コピー
- `latest.jsonl`
  - 直近 run の HLO JSONL コピー
- `latest_summary.json`
  - 直近 run の summary コピー

## 使い分け

- まずは `single_integral` と `repeated_integral` の 2 タグを見ます。
- `repeated_integral` は `fori_loop` を含むため、制御フローがどれくらい支配的かを見やすいです。
- `single_integral` は 1 回の Smolyak 積分そのものの構造を見るのに向いています。
- size 指標を見るときは、まず
  - `preferred_text.utf8_bytes`
  - `hlo_proto_bytes`
  - `generated_code_size_in_bytes`
  - `temp_size_in_bytes`
    の 4 つを見ると、IR と compile 後の膨らみをざっくり切り分けやすいです。
