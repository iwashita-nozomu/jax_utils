# Skill 9: Type Checking & Static Analysis

## 概要

Pyright による厳密な型チェックと複数の静的解析ツール実行。

コード品質・型安全性・スタイル一貫性を複層的に検証します。

## 主要機能

- **Pyright 厳密チェック**: strict mode で型エラー検出
- **複数ツール実行**: ruff, mypy, pylint の並列実行
- **レポート生成**: HTML形式の詳細レポート
- **差分チェック**: 変更箇所のみをチェック（オプション）

## 依存関係

- Skill 1: Static Check

## ワークフロー

```
1. Pyright strict 実行
2. ruff, mypy 並列実行
3. エラー集約
4. レポート生成
5. しきい値判定
```
