# NN Lagrangian Newton Worktree Log

## 2026-04-04

- `work/nn-lagrangian-newton-20260404` を current `main` (`6030e1e`) から作成した。
- 目的は、ラグランジアンの二回微分までを使う Newton 型手法の文献調査、数理検証、設計、実装を、既存 `neuralnetwork` 本線と衝突しにくい別サブモジュールとして進めること。
- 作業対象は `python/jax_util/neuralnetwork/lagrangian_newton/` を仮の正本とし、既存の `neuralnetwork.py`、`train.py`、`sequential_train.py`、`dynamic_programming.py` は読み取り中心とする。
- 初期段階では、文献調査、数理的検証、設計文書、reference 整理を優先し、本実装はその後に入る。
