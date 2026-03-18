# ニューラルネットワーク規約（実験段階）

この章は、実験段階の `neuralnetwork` に限った補足方針をまとめます。

## 要約

- `python/jax_util/neuralnetwork/` は現在も整理中です。
- 安定サブモジュール向けの詳細規約とは分け、ここでは最小限の方針だけを残します。

## 規約

### 1. 位置づけ

- このサブモジュールは **実験段階** とします。
- 現時点では、`base` / `solvers` / `optimizers` / `hlo` のような安定規約の対象には含めません。

### 2. 最低限の設計方針

- `protocols.py` は契約だけを定義します。
- 最適化関連の契約名は `PyTreeOptimizationProblem` / `PyTreeOptimizationState` / `ConstrainedPyTreeOptimizationProblem` 系で統一します。
- `OptimizeProblemPytree` のような旧名や綴りゆれを新規導入しません。
- `neuralnetwork.py` は forward とファクトリをまとめます。
- `train.py` は最小の学習ループに限定します。
- 実験コードは公開 API と分離します。

### 3. 型と形状

- `Vector` / `Matrix` を使います。
- バッチ軸はプロジェクト共通規約に従い、`Matrix` は `(n, batch)` として扱います。

### 4. 今後の整理条件

- import が安定し、テスト収集と実行が通ること。
- 実験コードと公開 API の境界が明確になること。
- 上記を満たした段階で、必要に応じて詳細規約を再展開します。
