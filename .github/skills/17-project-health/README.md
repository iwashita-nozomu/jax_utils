# Skill 17: Project Health Monitoring

## 概要

プロジェクトの総合健全性を継続的に監視するスキル。

メトリクス追跡・定期レポート生成・アラート機構を提供し、
プロジェクトの長期的な品質維持をサポートします。

## 主要機能

- **メトリクス追跡**: テストカバレッジ、型エラー、ドキュメント品質
- **定期レポート**: 日次・週次・月次のヘルスチェックレポート
- **トレンド分析**: メトリクスの時系列変化を追跡
- **アラート機構**: 閾値超過時の即時通知

## 依存関係

- Skill 1: Static Check
- Skill 6: Comprehensive Review

## ワークフロー

```
1. 全メトリクス収集
2. 前期比較・トレンド分析
3. 健全性スコア計算
4. しきい値判定
5. レポート生成・通知
```

## メトリクス項目

| メトリクス | 対象 | 良好基準 |
|-----------|------|--------|
| Type Errors | Pyright strict | 0 errors |
| Test Coverage | pytest + coverage | >= 80% |
| Documentation | markdown files | No broken links |
| Skills Health | Skill definitions | All PASS |
| CI/CD Status | GitHub Actions | All workflows pass |

## 実行例

```bash
python3 .github/skills/17-project-health/run-monitor.py --interval daily
```
