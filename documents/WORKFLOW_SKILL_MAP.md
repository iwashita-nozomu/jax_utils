# ワークフロー・Skill マッピング

## 10個の想定ワークフロー定義

### 1. 実験初期化ワークフロー
- **概要**: 新規実験・プロジェクトの初期セットアップと設定
- **対象者**: 研究者・エンジニア
- **主スキル**: ディレクトリ作成、config.yaml 生成、依存関係確認
- **実行時間**: 5-10分
- **Skill**: **Skill 8: Experiment Initialization**

### 2. パラメータ最適化ワークフロー
- **概要**: Hyperparameter tuning・Grid search・Bayesian optimization
- **対象者**: 機械学習エンジニア
- **主スキル**: Parameter space 定義、tuner 実行、結果追跡
- **実行時間**: 数時間～数日
- **Skill**: **Skill 9: Parameter Optimization**

### 3. 結果分析・可視化ワークフロー
- **概要**: 実験結果の統計分析・グラフ描画・比較
- **対象者**: データサイエンティスト
- **主スキル**: メトリクス計算、図表生成、レポート作成
- **実行時間**: 20-60分
- **Skill**: **Skill 10: Result Analysis & Visualization**

### 4. コード品質保証ワークフロー
- **概要**: Linting・formatting・型チェック・スタイル統一
- **対象者**: 全エンジニア
- **主スキル**: Pylint・Black・Pyright実行、修正提案
- **実行時間**: 5-15分
- **Skill**: **Skill 11: Code Quality Assurance**

### 5. パフォーマンス最適化ワークフロー
- **概要**: プロファイリング・ボトルネック検出・最適化実装
- **対象者**: HPC・パフォーマンスエンジニア
- **主スキル**: cProfile・JAX profiler・メモリ分析
- **実行時間**: 30-120分
- **Skill**: **Skill 12: Performance Optimization**

### 6. ドキュメント生成ワークフロー
- **概要**: API Doc・README生成・コード例・規約チェック
- **対象者**: テクニカルライター・リード
- **主スキル**: docstring抽出、Sphinx/MkDocs、リンク検証
- **実行時間**: 10-30分
- **Skill**: **Skill 13: Documentation Generation**

### 7. 依存関係管理ワークフロー
- **概要**: 要件確認・パッケージ更新・セキュリティスキャン
- **対象者**: DevOps・インフラエンジニア
- **主スキル**: pip audit・Poetry・requirements.txt同期
- **実行時間**: 10-20分
- **Skill**: **Skill 14: Dependency Management**

### 8. CI/CDパイプラインワークフロー
- **概要**: GitHub Actions・ワークフロー検証・デプロイ自動化
- **対象者**: DevOps・メインテナー
- **主スキル**: Workflow ファイル生成、テスト実行、ステージング
- **実行時間**: 15-40分
- **Skill**: **Skill 15: CI/CD Pipeline Management**

### 9. チーム協業・レビューワークフロー
- **概要**: PR 作成・コードレビュー・conflict 解決・マージ
- **対象者**: チーム全体
- **主スキル**: Git worktree・PR lint・マージ戦略
- **実行時間**: 20-60分
- **Skill**: **Skill 16: Team Collaboration**

### 10. リリース・デプロイワークフロー
- **概要**: バージョン管理・Changelog・リリースノート・デプロイ
- **対象者**: リリースマネージャー・メインテナー
- **主スキル**: Git tag・PyPI・バージョニング・リリース検証
- **実行時間**: 30-90分
- **Skill**: **Skill 17: Release & Deployment**

---

## Skill体系図

```
Skill 1: Static Check
  └─ Skill 8: Experiment Initialization
     ├─ Skill 9: Parameter Optimization
     └─ Skill 10: Result Analysis
  
Skill 2: Code Review
  └─ Skill 11: Code Quality Assurance
  
Skill 3: Run Experiment
  ├─ Skill 12: Performance Optimization
  └─ Skill 13: Documentation Generation
  
Skill 4: Critical Review
  
Skill 5: Research Workflow
  └─ Skill 14: Dependency Management
  
Skill 6: Comprehensive Review
  
Skill 7: Health Monitor
  ├─ Skill 15: CI/CD Pipeline Management
  ├─ Skill 16: Team Collaboration
  └─ Skill 17: Release & Deployment
```

---

## 実装順序

1. **Tier 1 (基盤)**: Skills 8, 9, 10 — 実験フロー
2. **Tier 2 (品質)**: Skills 11, 12, 13 — 開発品質
3. **Tier 3 (運用)**: Skills 14, 15, 16, 17 — 運用・デプロイ
