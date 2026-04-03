# Smolyak Experiment Design

`experiments/smolyak_experiment/` は、`python/jax_util/functional/smolyak.py` の
積分器を「積分器として」検証するための実験ディレクトリです。

この README を、このディレクトリの設計上の正本とします。過去の tuning 履歴や
個別レポートは補助資料であり、運用設計の source of truth ではありません。

## Goal

この実験ディレクトリの目的は次の 3 点だけです。

- Smolyak 積分器の API 契約が守られているか確認する
- 積分精度、計算量、GPU 実行性能を定量評価する
- Monte Carlo との比較を、同じ被積分関数・同じ解析解のもとで行う

逆に、このディレクトリの目的ではないものは次です。

- 積分器の定義を暗黙に変えること
- tuning 履歴を実験コードの構造に埋め込むこと
- 一時的な convenience script を増やし続けること

## Integrator API Contract

このディレクトリで扱う `SmolyakIntegrator` は、まず積分器として次の前提を持つものとして運用します。

- 積分領域は `[-0.5, 0.5]^d`
- 測度は box 上の通常の Lebesgue 積分
- `integrate(f)` に渡す `f` は、1 点 `x` を受け取って値ベクトルを返す pure function
- `x` の shape は `(dimension,)`
- `f(x)` の返り値は `(m,)` shape の JAX array を想定する
- `points / indexed / lazy-indexed / batched` は、同じ quadrature rule の実装差であり、数値結果は一致しなければならない

重要なのは、canonical な `Smolyak` はまず等方 sparse grid だということです。
`dimension_weights` 付きの anisotropic 版は別の実験対象であり、標準の Smolyak を
静かに置き換えるものではありません。

## Design Principles

- 実験コードは「case layer」と「experiment engine」を中心に最小化する
- 被積分関数の定義は case layer の責務であり、engine 側に数学的前提を散らさない
- レポート生成はすべて導出物であり、engine の上位に置く
- 既定の評価軸は canonical Smolyak を壊さないことを優先する
- anisotropic や weighted な評価は、canonical Smolyak とは明示的に区別して読む

## Canonical Structure

このディレクトリの正本構造は、概念的には次の 3 層です。

### 1. Case Layer

責務: 「何を積分するか」を決める。

- `cases.py`
  - 解析解つきの被積分関数 family を定義する
  - 実験ケースから実際の `integrand` と `analytic_value` を構成する
  - case の直積集合、識別子、必要なら大まかな resource estimate を与える

ここでいう case には、少なくとも次が含まれます。

- family
- dimension
- level
- dtype
- requested mode
- 比較対象として必要な解析解パラメータ

### 2. Experiment Engine

責務: case を実行し、JSONL に記録する。

- `run_smolyak_mode_matrix.py`
  - canonical な実験機本体
  - case の列挙
  - child process 分離
  - timeout / OOM / numerical failure の記録
  - raw JSONL の生成

この script を、Smolyak 実験の正本 runner とみなします。

### 3. Derived Analysis

責務: raw JSONL や単発 case 結果を読み、比較表・図・Markdown を作る。

- `compare_smolyak_vs_mc.py`
  - 単一 case に対する Smolyak vs Monte Carlo 比較
- `report_smolyak_mode_matrix.py`
  - mode matrix JSONL から CSV / SVG / Markdown を生成

この 2 つは engine の上に乗る導出層です。

## What Should Be Reduced

現状は script が散らばっており、正本が見えにくくなっています。今後は次の方針で削減します。

### Keep As Canonical

- `cases.py`
- `run_smolyak_mode_matrix.py`
- `compare_smolyak_vs_mc.py`
- `report_smolyak_mode_matrix.py`

### Treat As Derived Or Transitional

- `report_smolyak_gpu_sweep.py`
- `report_smolyak_gpu_batch_scaling.py`
- `report_smolyak_same_budget_accuracy.py`
- `report_smolyak_theory_comparison.py`
- `report_smolyak_vs_mc.py`
- `run_smolyak_large_full_report.py`
- `run_smolyak_same_budget_levels_1_to_10.py`
- `run_smolyak_research_loops.py`

### Legacy Candidates To Merge Or Delete

- `run_smolyak_experiment_simple.py`
- `runner_config.py`
- `results_aggregator.py`

特に `run_smolyak_experiment_simple.py` は、現在の `mode matrix` 系と責務が重なっています。
今後は「legacy scaling runner」として扱い、canonical runner へ寄せていく前提です。

## Data Flow

標準的な流れは次です。

1. `cases.py` で case と解析解つき integrand を定義する
2. `run_smolyak_mode_matrix.py` で raw JSONL を生成する
3. `report_smolyak_mode_matrix.py` で CSV / SVG / Markdown に変換する
4. 必要なら `compare_smolyak_vs_mc.py` で代表 case を掘る

この順序から外れる one-off script は、原則として増やさない方針です。

## Interpretation Rules

精度評価では、次を区別して読む必要があります。

- canonical Smolyak が悪いのか
- weighted / anisotropic variant が悪いのか
- implementation mode が悪いのか
- GPU 実装上の timeout / OOM が悪いのか

したがって、少なくとも次は毎回結果に残るべきです。

- `requested_mode`
- `actual_mode`
- `num_terms`
- `num_evaluation_points`
- `storage_bytes`
- `active_axis_count`
- `axis_level_ceilings`

`active_axis_count < dimension` の case は、canonical な高次元 Smolyak ではなく
実質的に低次元へ簡約された anisotropic rule とみなして読むべきです。

## Current Operational Rule

今の運用上の正本は次です。

- canonical な Smolyak 評価では `dimension_weights=None`, `weight_scheme=none`
- weighted な run は「Smolyak そのもの」ではなく anisotropic variant として別扱い
- mode 間比較は値一致を前提に行う
- tuning 履歴は README に持ち込まない

## Minimal Commands

正本の最小コマンドは次だけです。

mode matrix:

```bash
python3 -m experiments.smolyak_experiment.run_smolyak_mode_matrix \
  --platform gpu \
  --dimensions 1,2,3,4,5,6,7,8,9,10 \
  --levels 1,2,3,4,5 \
  --dtypes float64 \
  --families gaussian \
  --requested-modes auto,points,indexed,lazy-indexed,batched
```

single-case Monte Carlo compare:

```bash
python3 -m experiments.smolyak_experiment.compare_smolyak_vs_mc \
  --platform gpu \
  --dimension 8 \
  --level 6 \
  --dtype float64 \
  --family gaussian \
  --requested-mode lazy-indexed
```

mode-matrix report:

```bash
python3 -m experiments.smolyak_experiment.report_smolyak_mode_matrix \
  --jsonl-path <results.jsonl> \
  --output-dir <report_dir>
```

## Non-Goals For The Source Tree

このディレクトリでは、次を source tree の責務にしません。

- どの tuning が過去に速かったかの履歴管理
- branch ごとの運用記録
- 研究メモの正本化

それらは `notes/` 側の補助資料であり、この README の代わりにはなりません。

## Immediate Cleanup Direction

次に行う整理は次です。

- canonical runner を `run_smolyak_mode_matrix.py` に寄せる
- legacy runner / aggregator の責務を吸収して削減する
- specialized report script を、正本 runner の導出物として再配置または削減する
- README を起点に「どれが正本か」を常に一意に保つ
