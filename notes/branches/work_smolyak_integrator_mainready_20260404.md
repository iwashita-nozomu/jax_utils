# Smolyak Integrator Main-Ready Branch

- Branch: `work/smolyak-integrator-mainready-20260404`
- Status: `active`
- Worktree: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328`
- Purpose: 最新の Smolyak integrator、Monte Carlo 比較実験コード、関連文書を main へ持ち込むための統合 branch。

## Included Areas

- `python/jax_util/functional/smolyak.py`
- `python/jax_util/functional/monte_carlo.py`
- `python/jax_util/functional/integrate.py`
- `experiments/functional/smolyak_scaling/`
- `notes/experiments/`
- `notes/knowledge/`

## Main Points

- Smolyak 側は rule storage 初期化と decode 周りの改善を含む。
- Monte Carlo 側は chunked execution へ寄せ、same-budget baseline に載せやすくした。
- `smolyak_scaling` は最新 runner 契約に追従した実験入口を含む。
- HLO / bottleneck / frontier 分析の notes をこの branch に同梱する。

## Merge Intent

- main へ持ち込むときは、この branch を指定すれば最新の積分器実装、実験コード、文書をまとめて参照できる。
- raw 実験結果や一時 dump は branch の本体には含めず、コードと文書を正本とする。
