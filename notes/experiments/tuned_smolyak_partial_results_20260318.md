# Tuned Smolyak Partial Results 2026-03-18

## 0. このメモの目的

このメモは、

`results/functional-smolyak-scaling-tuned` worktree 上で継続中の

Smolyak 実験について、

`2026-03-18T07:20:55Z` 時点の途中結果だけを使って

考察を進めるためのものである。

実験本体は継続中であり、

ここで扱う数値は

live state ではなく

スナップショットに固定した値である。

## 1. 関連 branch と生成物

- 実験 worktree / branch
  - `results/functional-smolyak-scaling-tuned`
- 途中結果の JSONL
  - [`smolyak_scaling_gpu_20260316T134600Z.jsonl`](../../experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260316T134600Z.jsonl)
- このメモの基準にした partial snapshot
  - [`smolyak_scaling_gpu_20260316T134600Z_partial_20260318T072055Z.json`](../../experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260316T134600Z_partial_20260318T072055Z.json)
- partial report
  - [`smolyak_scaling_gpu_20260316T134600Z_partial_20260318T072055Z_report/index.html`](../../experiments/functional/smolyak_scaling/results/smolyak_scaling_gpu_20260316T134600Z_partial_20260318T072055Z_report/index.html)
- 実行ログ
  - [`run_smolyak_scaling_20260316T134600Z.log`](../../experiments/functional/smolyak_scaling/results/run_smolyak_scaling_20260316T134600Z.log)

## 2. スナップショットの範囲

- 全予定ケース数は `40804 = 101 dimensions * 101 levels * 4 dtypes`
- snapshot 時点の完了件数は `1700`
- 進捗率は約 `4.17%`
- 完了済みの `dimension` は `0, 1, 2, 3` 全域と、`dimension=4` の `level<=20`
- 自明な不正入力 `dimension=0` または `level=0` を除いた valid 完了件数は `1280`
- valid 完了件数の内訳は `226 ok`, `1054 failed`

## 3. 観測結果

### 3.1 failure の全体像

snapshot 全体の内訳は次の通りである。

- `ok`: 226
- `error`: 424
- `oom`: 10
- `worker_terminated`: 872
- `timeout`: 168

Interpretation:

`error=424` の大半は

`dimension=0` または `level=0` を投げたことによる

自明失敗である。

valid case に限ると、

- `ok`: 226
- `oom`: 10
- `worker_terminated`: 872
- `timeout`: 168
- `error`: 4

となる。

Interpretation:

この tuned 実装の支配的失敗は

`worker_terminated` と `timeout` であり、

通常の Python 例外ではない。

### 3.2 次元ごとの frontier

`level>=1` に限ったときの

「全 dtype 成功の最後の level」と

「失敗が立ち上がる level」は次の通りである。

| dimension | last all-ok level | first failure | failure mode |
| --- | ---: | ---: | --- |
| 1 | 28 | 23 | `oom` が先に出て、`29+` は全 dtype `worker_terminated` |
| 2 | 13 | 14 | `14+` は全 dtype `timeout` |
| 3 | 10 | 11 | `11+` は全 dtype `timeout` |
| 4 | 8 | 9 | `9+` は全 dtype `timeout` |

Interpretation:

現在の tuned 実装では、

`dimension=1` は主にメモリ壁、

`dimension>=2` は主に時間壁で止まっている。

特に `dimension` が 1 増えるごとに

許容 `level` がかなり早く下がる。

### 3.3 dtype 比較

4 dtype すべてが成功した共通セルは `54` 個ある。

その共通セルだけで比較すると、

| dtype | mean abs err | max abs err | mean avg time [s] | max avg time [s] |
| --- | ---: | ---: | ---: | ---: |
| `float16` | `4.87e-3` | `6.74e-2` | `1.227` | `14.442` |
| `bfloat16` | `1.75e-2` | `7.54e-2` | `1.243` | `14.655` |
| `float32` | `8.38e-4` | `1.42e-2` | `1.236` | `14.347` |
| `float64` | `8.35e-4` | `1.42e-2` | `1.255` | `14.179` |

Interpretation:

この成功領域では

`float32` と `float64` の誤差差はほぼ消えている。

これは丸め誤差より

Smolyak 近似そのものの離散化誤差が支配的であることを示す。

一方で

`float16` と `bfloat16` は

精度劣化がはっきり見える。

Interpretation:

実行時間は 4 dtype でほぼ同じであり、

低精度化による明確な throughput 利得は

今の成功領域では見えていない。

### 3.4 メモリ壁の具体例

最大成功ケースは

`dimension=1, level=28` である。

このとき

- `num_points = 268435472`
- `process_rss_mb ≈ 24.6 GB`

である。

dtype ごとの `storage_bytes` は

- `float16`: `1073742672`
- `bfloat16`: `1073742672`
- `float32`: `2147484560`
- `float64`: `4294968336`

だが、

実 RSS はどの dtype でもほぼ `24.6 GB` に張り付いている。

Interpretation:

支配的なメモリ消費は

最終格納配列の dtype サイズ差より、

point / weight の明示展開や

途中バッファを含む host-side 構築コストにある可能性が高い。

そのため、

dtype を軽くするだけでは

`dimension=1` の高 level 領域を大きく押し広げられていない。

### 3.5 時間壁の具体例

高次元側の最後の成功ケースでは、

- `dimension=3, level=10`
  - `num_points = 71634`
  - `avg_integral_seconds ≈ 13.0 - 13.6 s`
- `dimension=4, level=8`
  - `num_points = 46290`
  - `avg_integral_seconds ≈ 14.2 - 14.7 s`

である。

その直後の

- `dimension=3, level=11`
- `dimension=4, level=9`

は全 dtype で `timeout(3600s)` へ落ちる。

Interpretation:

高次元側では

単純な `num_points` の増加だけでなく、

積分器初期化や評価構造の組み合わせで

急峻な時間壁が立っている。

`dimension=4` では

`dimension=3` より点数が少ない成功ケースでも

実行時間が同程度まで悪化しているため、

点数以外の構造コストが無視できない。

### 3.6 非自明 error

valid case に含まれる `error` は 4 件だけで、

すべて

`RuntimeError: Unable to initialize backend 'cuda'`

である。

発生箇所は

- `dimension=2, level=40, float16`
- `dimension=2, level=52, bfloat16/float32/float64`

である。

Interpretation:

これは主要 failure mode ではなく、

GPU backend 初期化の一時的不整合に近い。

考察の中心は

依然として `worker_terminated` と `timeout` でよい。

## 4. 現時点での判断

Consideration:

この tuned 実装は、

少なくとも現在の成功領域では

`float32` を第一候補に置く理由がある。

- `float64` とほぼ同等の誤差
- 実行時間差はごく小さい
- `float16` / `bfloat16` ほどの精度劣化がない

Consideration:

frontier を押し広げるための主論点は

dtype 選択ではなく、

explicit grid / host-side materialization をどう減らすかである。

現状の失敗様式は、

- `dimension=1` では host memory pressure に起因する kill
- `dimension>=2` では 3600 秒 timeout

に分かれており、

どちらもアルゴリズムと実装構造の改善を要求している。

## 5. 次の実験論点

Idea:

次の改訂で優先すべきなのは、

dtype 追加比較より

次のような構造改善である。

1. point / weight の全展開を避ける
2. host 上の一時配列と重複保持を減らす
3. evaluation を chunking / streaming できる形へ寄せる
4. `dimension=0`, `level=0` を最初から実験レンジから外す

Idea:

計測面では、

`num_repeats=100` は安定した timing には有用だが、

frontier 探索だけなら

repeat 数を下げた探索 run と

詳細計測 run を分離したほうが、

高次元側の境界把握は速くなる。
