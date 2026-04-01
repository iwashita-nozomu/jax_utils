# Skill 15: Docker Environment Validation

## 概要

Docker 環境の整合性・依存関係・セキュリティを検証。

Dockerfile・requirements.txt・イメージビルド を統一的に管理します。

## 主要機能

- **依存関係検証**: requirements.txt ↔ Dockerfile 同期確認
- **ビルド検証**: docker build 実行・イメージ確認
- **セキュリティチェック**: 脆弱性スキャン（trivy等）
- **レイヤー最適化**: イメージサイズ・ビルド時間分析

## 依存関係

- Skill 1: Static Check

## ワークフロー

```
1. Dockerfile・requirements 読み込み
2. 依存関係同期確認
3. イメージビルド検証
4. セキュリティチェック
5. 最適化提案・レポート
```
