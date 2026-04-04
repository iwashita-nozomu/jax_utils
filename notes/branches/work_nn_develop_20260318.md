# Neuralnetwork Layer-Wise Training Branch

- Branch: `work/nn-develop-20260318`
- Status: `archived`
- Worktree: closed on 2026-04-04
- Purpose: neuralnetwork 層別訓練の設計、実装、簡易実験を進めた旧 worktree。

## Read This First

- [documents/coding-conventions-python.md](/workspace/documents/coding-conventions-python.md)
- [documents/coding-conventions-testing.md](/workspace/documents/coding-conventions-testing.md)
- [documents/coding-conventions-experiments.md](/workspace/documents/coding-conventions-experiments.md)

## Current Scope

- `python/jax_util/neuralnetwork/` の設計メモと試作
- `notes/themes/` の Bellman / layer-wise training 系 note
- `references/` の層別訓練と Bellman backprop 系参考文献

## Carry-Over Targets

- `main` の `python/jax_util/neuralnetwork/`
- `notes/themes/`
- `references/`
- 新 worktree: `work/nn-develop-20260404`

## Notes

- 実験結果そのものは results branch 側で管理し、`main` へは note を中心に持ち帰る。
- 2026-04-04 に `neuralnetwork` の import-safe な scaffold、関連 theme note、reference index を `main` に carry-over した。
