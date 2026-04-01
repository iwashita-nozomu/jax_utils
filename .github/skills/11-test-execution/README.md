# Skill 11: Test Execution & Coverage

## 概要

pytest による包括的なテスト実行とカバレッジ測定。

ユニット・統合・エッジケーステストを実行し、
カバレッジレポートと html 形式の詳細結果を生成します。

## 主要機能

- **テスト発見・実行**: pytest で全テスト自動検出
- **カバレッジ測定**: coverage.py で行・ブランチカバレッジ記録
- **並列実行**: pytest-xdist で高速化
- **レポート生成**: HTML・JSON・XML 形式で出力

## 依存関係

- Skill 8: Experiment Initialization
- Skill 9: Type Checking

## ワークフロー

```
1. テスト検出
2. 並列実行
3. カバレッジ測定
4. レポート生成
5. 閾値判定・通知
```
