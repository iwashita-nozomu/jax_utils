# 実装アクションプラン — 12 週間ロードマップ

**期間**: 2026 年 4 月～6 月  
**目標**: 自動化成熟度 12% → 70%, セキュリティ対応 0% → 85%

---

## Phase 1: 緊急対応（Week 1-2）

### Week 1: セキュリティ基盤

**優先度**: 🔴 P0（即時）

#### Task 1.1: GitHub Secrets 統合
```
所要時間: 2-3 時間
成果物:
  ✅ .github/workflows/*.yml で ${{ secrets.* }} 使用
  ✅ scripts/security/secrets_manager.py (skeleton)
  ✅ ドキュメント: docs/SECRETS_MANAGEMENT.md
検収基準:
  □ CI でハードコード Credential なし
  □ ローカル開発では .env.local で管理
  □ Audit log に ACCESS ログ記録
```

#### Task 1.2: 基本監査ログシステム
```
所要時間: 4-5 時間
成果物:
  ✅ scripts/security/audit_logger.py
  ✅ reports/security/audit.log (JSON フォーマット)
  ✅ .github/workflows/ci.yml に audit checkpoint 追加
検収基準:
  □ すべてのアクション（read/write/execute）を JSON で記録
  □ タイムスタンプ・Actor・Target・Result がある
  □ 他のスクリプトから利用可能（import audit_logger）
```

#### Task 1.3: 脆弱性スキャン基本実装
```
所要時間: 3-4 時間
成果物:
  ✅ .github/workflows/security-scan.yml (weekly)
  ✅ pip-audit + trivy 統合
  ✅ 脆弱性検出時に自動 Issue 作成
検収基準:
  □ 毎週月曜 09:00 UTC に自動実行
  □ Critical/High 脆弱性で即座に GitHub Issue 作成
  □ SBOM 生成
```

**Week 1 達成基準**: ✅ セキュリティ基礎 3 つが動作中

---

### Week 2: ログ・メトリクス自動化

**優先度**: 🔴 P0

#### Task 2.1: ログ・メトリクス収集フレームワーク
```
所要時間: 5-6 時間
成果物:
  ✅ scripts/metrics/collector.py
    - test coverage
    - type check errors (pyright)
    - lint violations (ruff)
    - build time
  ✅ reports/metrics/ (daily JSON)
  ✅ .github/workflows/metrics-collect.yml
検収基準:
  □ 毎日 02:00 UTC に実行
  □ JSON 出力で CI ツール連携可能
  □ 前日との Δ（差分）を計算
  □ Dashboard 用 CSV export
```

#### Task 2.2: スケジュール自動化（基本 3 つ）
```
所要時間: 4-5 時間
成果物:
  ✅ .github/workflows/scheduled-checks.yml
    1. 日次: リポジトリ health score
    2. 週次: Skill 自動実行（Skill 6）
    3. 月次: 技術負債評価
  ✅ 自動 Slack 通知（重大 Issue のみ）
検収基準:
  □ 各スケジュール正確に実行
  □ Slack/Email 通知が送られる
  □ レポートが reports/ に保存される
```

#### Task 2.3: Skill 依存関係 DAG 構築
```
所要時間: 3-4 時間
成果物:
  ✅ scripts/dependency_graph.py
  ✅ DAG 可視化 (Graphviz)
  ✅ 実行順序の自動最適化
検収基準:
  □ Skill 1-6 + scripts の依存関係を定義
  □ 循環依存なし
  □ 並列実行可能部分を自動検出
```

**Week 2 達成基準**: ✅ ログ・メトリクス + スケジュール + DAG が稼働

---

## Phase 2: 短期実装（Week 3-6）

### Week 3: エラーハンドリング・検証体系

#### Task 3.1: 統一エラーハンドリング標準

```python
所要時間: 4-5 時間
成果物:
  ✅ scripts/shared/error_handler.py (ExecutionResult class)
  ✅ すべてのスクリプトが統一形式でエラー報告
  ✅ JSON/Markdown 出力オプション
検収基準:
  □ エラーコード体系を定義
  □ Severity レベル（WARN/ERROR/FATAL）
  □ すべてのスクリプト 5 つ以上を refactor して移行
```

#### Task 3.2: Doc-Test-Impl 三点セット検証

```
所要時間: 3-4 時間
成果物:
  ✅ scripts/validation/triplet_validator.py
  ✅ README ↔ スクリプト ↔ テスト の一貫性チェック
  ✅ CI に統合
検収基準:
  □ Skill 6 で三点セット検証パス
  □ 実装漏れを自動検出
  □ リポートが見やすい形式
```

#### Task 3.3: RBAC 基本実装

```
所要時間: 5-6 時間
成果物:
  ✅ scripts/security/role_access_control.py
  ✅ ROLE_PERMISSIONS 定義
  ✅ 権限チェック関数
検収基準:
  □ Manager/Researcher/AuditorRole を定義
  □ read/write/execute 権限を分離
  □ audit log に権限チェック結果を記録
```

### Week 4-5: Skill 1 実装（static-check）

#### Task 4.1-4.3: Skill 1 スクリプト化

```
所要時間: 8-10 時間（分割実装）
成果物:
  ✅ .github/skills/01-static-check/run-check.py
  ✅ checkers/python_type_checker.py
  ✅ checkers/python_lint_checker.py
  ✅ checkers/docker_validator.py (既存 reuse)
  ✅ .github/skills/01-static-check/test-static-check.sh
検収基準:
  □ claude --skill static-check で動作確認
  □ すべてのチェッカーが独立して実行可能
  □ テストが全機能をカバー（80% 以上）
  □ Docker で実行可能
```

**Task 4.4: Skill 1 CI 統合**

```
所要時間: 2-3 時間
成果物:
  ✅ .github/workflows/ci.yml で Skill 1 呼び出し
  ✅ PR に自動コメント（Skill 1 結果）
検収基準:
  □ PR 作成時に自動実行
  □ 結果がコメント表示される
  □ エラー時は CI fail
```

### Week 6: Skill 2 開始 + Skill 7 準備

#### Task 6.1: Skill 2 (code-review) スケルトン

```
所要時間: 6-8 時間
成果物:
  ✅ .github/skills/02-code-review/run-review.py (stub)
  ✅ checkers/layer_a_checker.py (skeleton)
  ✅ checkers/layer_b_checker.py (skeleton)
検収基準:
  □ Layer A のみ完全実装
  □ B/C は stub（後続 week で実装）
```

#### Task 6.2: Skill 7 (Health Monitor) 準備

```
所要時間: 4-5 時間
成果物:
  ✅ .github/skills/07-health-monitor/run-monitor.py (design)
  ✅ health_score.py (metrics 定義)
  ✅ README ドラフト
検収基準:
  □ 5 つの主要メトリクスが定義
  □ ウェイト設定が妥当
  □ Dashboard mock が可視化
```

---

## Phase 3: 中期実装（Week 7-9）

### Week 7: Skill 2 完成 + Skill 7 リリース

#### Task 7.1: Skill 2 (code-review) Layer A/B 完成

```
所要時間: 8-10 時間
成果物:
  ✅ Layer A: 型・Docstring・テスト チェック
  ✅ Layer B: テストアーキテクチャ・スタイル チェック
  ✅ テストカバレッジ 80%+
検収基準:
  □ 既存サンプルコード 3-5 個でテスト
  □ 誤検知なし
  □ すべてのエラーパス有テスト
```

#### Task 7.2: Skill 7 (Health Monitor) リリース

```
所要時間: 6-8 時間
成果物:
  ✅ .github/skills/07-health-monitor/run-monitor.py (完全)
  ✅ .github/workflows/health-monitor.yml (毎日 02:00 UTC)
  ✅ Dashboard 用 JSON output
  ✅ Slack 通知
検収基準:
  □ 毎日実行で health score 計算
  □ 3 日分以上のデータ蓄積
  □ トレンド表示（矢印 ▲/▼）
  □ Slack 通知が送られる
```

### Week 8: Skill 3 開始 + Skill 8 準備

#### Task 8.1: Skill 3 (run-experiment) スケルトン

```
所要時間: 6-8 時間
成果物:
  ✅ .github/skills/03-run-experiment/run-experiment.py
  ✅ experiment orchestrator (準備段階)
  ✅ README リファイン
検収基準:
  □ 1-5 段階の基本フロー
  □ setup-template.md との一貫性
  □ experiment_runner 連携
```

#### Task 8.2: Skill 8 (Security Audit) 詳細設計

```
所要時間: 4-5 時間
成果物:
  ✅ .github/skills/08-security-audit/design.md
  ✅ checkers リスト定義
  ✅ 実装計画書
```

### Week 9: Skill 4/5 開始

#### Task 9.1: Skill 4 (critical-review) 基本実装

```
所要時間: 8-10 時間
成果物:
  ✅ .github/skills/04-critical-review/run-critical.py
  ✅ Change/Experiment/Math Reviewer 役割の基本実装
```

#### Task 9.2: Skill 5 (research-workflow) 基本実装

```
所要時間: 8-10 時間
成果物:
  ✅ .github/skills/05-research-workflow/run-workflow.py
  ✅ iteration manager 基本実装
```

---

## Phase 4: 完成・本運用（Week 10-12）

### Week 10: Skill 2 Layer C + Skill 8 リリース

### Week 11-12: Skill 9-10 + テスト・統合

---

## 実装チェックリスト（Weekly）

### 各 Week で確認すべき項目

```markdown
## Week 1 検収
- [ ] GitHub Secrets が使用されている
- [ ] audit.log に JSON エントリがある
- [ ] 脆弱性検出時に Issue が自動作成される
- [ ] セキュリティチェックが CI で実行される

## Week 2 検収
- [ ] reports/metrics/ に daily/*.json がある
- [ ] スケジュールワークフローが実行された
- [ ] DAG に循環依存なし
- [ ] Dashboard 用 CSV が生成される

## Week 3-6 検収
- [ ] エラーハンドリング統一化 (5+ scripts)
- [ ] 三点セット検証が Skill 6 でパス
- [ ] RBAC が定義・実装されている
- [ ] Skill 1 が PR で自動実行される

## Week 7-9 検収
- [ ] Skill 2 が3層（A/B/C）すべて実装
- [ ] Health Monitor が毎日実行
- [ ] Skill 3-5 が基本実装完了
- [ ] Skill 8 がセキュリティスキャン実行

## Week 10-12 検収
- [ ] すべての Skill が CI に統合
- [ ] 24/7 レビューメカニズムが稼働
- [ ] Dashboard にプロジェクト health が表示
- [ ] 自動改善フロー（auto-PR）が機能
```

---

## リスク管理

### 高リスク項目

| リスク | 対策 |
|--------|------|
| Skill 実装の遅延 | 週ごとのマイルストーン明確化、スケルトン駆動開発 |
| セキュリティ実装ミス | Security Review + Audit log からの検証 |
| 既存スクリプトとの衝突 | 互換性テスト before refactor |
| ドキュメント同期漏れ | Triple-check (code-test-doc validation) |

---

## 投資対効果

| 投資 | 期間 | 効果 |
|------|------|------|
| 1-2 週（秘密管理） | 10-20 時間 | セキュリティリスク 90% 削減 |
| 2-4 週（自動化） | 20-30 時間 | 手動作業 40-50% 削減 |
| 4-8 週（Skill）| 60-80 時間 | チーム効率 2x-3x 向上 |
| 8-12 週（完成）| 150-200 時間 | **完全自動化・監視システム確立** |

**総投資**: 約 150-200 時間 (3-4 人月相当)  
**ROI**: 3-6 ヶ月後に **月間 100+ 時間** の自動化効果

---

## 成功基準（4 ヶ月目）

```
✅ すべての Skill (1-8) が実装・本運用
✅ セキュリティスコア: 0 → 85 以上
✅ 自動化スコア: 12% → 70% 以上
✅ 24/7 プロジェクト health monitoring 稼働
✅ 月間手動作業 40-50% 削減
✅ セキュリティ監査完全自動化
✅ 開発速度・品質 大幅向上（メトリクスで証明）
```

---

**最終更新**: 2026-04-01  
**責務**: Infrastructure Steward / DevOps Lead  
**期間**: 12 週間（Q2 2026）
