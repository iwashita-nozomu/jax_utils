# NN DF Zero Worktree Log

## 2026-04-04

- `work/nn-df-zero-20260404` を current `main` (`6030e1e`) から作成した。
- 目的は、ラグランジアンの一回微分に基づく最急法のうち、parameter update を `DF=0` の解として与える枠組みを、別サブモジュールとして調査・検証・設計・実装すること。
- 作業対象は `python/jax_util/neuralnetwork/lagrangian_df_zero/` を仮の正本とし、既存の `neuralnetwork.py`、`train.py`、`sequential_train.py`、`dynamic_programming.py` は読み取り中心とする。
- 初期段階では、文献調査、数理的検証、設計文書、reference 整理を優先し、本実装はその後に入る。
