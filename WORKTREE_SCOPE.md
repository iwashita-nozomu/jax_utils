# WORKTREE_SCOPE — ニューラルネットワーク層別訓練実装

このワークツリーは、ニューラルネットワークの層別訓練（layer-wise training）戦略を設計・実装します。

## Worktree Summary

- **Branch**: `work/nn-develop-20260318`
- **Worktree path**: `./.worktrees/nn-develop-20260318`
- **Purpose**: NN モジュール層別訓練実装・テスト・簡易実験
- **Owner or agent**: GitHub Copilot / 開発チーム

## Editable Directories

- `python/jax_util/neuralnetwork/` — NN コア実装
- `python/tests/neuralnetwork/` — NN テストスイート
- `notes/experiments/` — 実験メモ・数値結果
- `notes/worktrees/` — このワークツリーのメモ

## Read-Only Or Avoid Directories

- `python/jax_util/base/` — 基盤層（参照のみ）
- `python/jax_util/solvers/` — ソルバ（参照のみ）
- `python/jax_util/optimizers/` — 最適化（参照のみ）
- `python/jax_util/functional/` — 関数型ユーティリティ（参照のみ）
- `python/jax_util/experiment_runner/` — 実験実行機（参照のみ）
- `solvers/archive/` — 保管領域（対象外）

## Required References Before Editing

- [documents/worktree-lifecycle.md](/workspace/documents/worktree-lifecycle.md) — ワークツリー運用規約
- [documents/coding-conventions-project.md](/workspace/documents/coding-conventions-project.md) — プロジェクト全体方針
- [documents/coding-conventions-python.md](/workspace/documents/coding-conventions-python.md) — Python 規約
- [documents/coding-conventions-testing.md](/workspace/documents/coding-conventions-testing.md) — テスト規約
- [documents/coding-conventions-experiments.md](/workspace/documents/coding-conventions-experiments.md) — 実験規約
- [notes/experiment_runner.md](/workspace/notes/experiment_runner.md) — 実験実行基礎

## Main Carry-Over Targets

- `notes/experiments/` — 実験メモ・分析結果
- `notes/worktrees/nn-develop/` — このワークツリーのコンテキスト
- `documents/` — 実装過程で確認する設計書（読み取り）

## Required Checks Before Commit

- `pyright python/jax_util/neuralnetwork python/tests/neuralnetwork` — 型チェック
- `pytest python/tests/neuralnetwork -v` — ユニットテスト
- `pydocstyle python/jax_util/neuralnetwork` — Docstring 検証
- `markdownlint` — 変更 md ファイルのみ

## Additional Rules

### 実装方針

1. **責務分离**: layer-wise training と standard forward/backward を明確に分離
2. **シンプル優先**: 複雑な分岐を避け、解釈可能な設計を保つ
3. **テスト駆動**: 実装前にテストケースを定義

### コミット時の制約

- 生成結果（`.json`, `.pkl`, 画像）は **commit しない**（results ブランチで管理）
- docstring 追加なしに commit しない
- HEAD は常に `origin/work/nn-develop-20260318` と同期
- 結果の参照は `notes/` に記載のみ
- 例: 変更した Markdown は `.markdownlint.json` を基準に確認する。
