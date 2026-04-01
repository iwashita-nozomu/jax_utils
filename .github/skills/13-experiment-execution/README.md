# Skill 13: Experiment Execution

## 概要

実験実行の統一管理。

初期化→学習→検証→テスト→報告という標準フローを提供します。

## 主要機能

- **段階実行**: 4段階（train, validate, test, report）を順序実行
- **チェックポイント**: 各段階のメタデータ記録
- **ログ集約**: 統一ログフォーマット・集約
- **失敗復旧**: チェックポイントから再開

## 依存関係

- Skill 3: Run Experiment
- Skill 8: Experiment Initialization

## ワークフロー

```
1. 設定・環境確認
2. Train 段階実行
3. Validate 段階実行
4. Test 段階実行
5. Report 生成・記録
```
