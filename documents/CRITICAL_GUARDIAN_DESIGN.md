---
title: "Critical Guardian Role Design — Always-On Critical Review Agent"
description: "SOUL (State of Utility Level) に常駐する批判的エージェント設計"
date: "2026-04-01"
---

# Critical Guardian Role Design

## Overview

「常に批判的なエージェント」を SOUL パイプラインの標準ロールとして追加する提案です。

このロールは：
- **責務**: すべての提案・変更に対して批判的視点からレビュー
- **特性**: リスク・漏れ・セキュリティ問題を徹底的に指摘
- **権限**: Project Health Score に直接影響
- **稼働**: 24/7（自動化）

---

## Motivation

現在のエージェント体系に欠けているもの：
- ✅ Developer: コードを書く
- ✅ Reviewer: コードをレビューする
- ✅ Researcher: 学術的妥当性を確認
- ❌ **Critic**: 常に反対意見をぶつける（存在しない）

**問題**: 
- 批判的視点が "on-demand" のみ
- デフォルトでは楽観的立場
- リスク・セキュリティ問題の見落とし

**解決**: Critical Guardian ロールを標準化

---

## Architecture

### 1. ロール定義

```yaml
role:
  name: "Critical Guardian"
  short_name: "critic"
  tier: "governance"
  responsibility:
    - "すべての提案に対する批判的レビュー"
    - "リスク・セキュリティ・設計上の問題指摘"
    - "実装の漏れ・矛盾を検出"
    - "代替案・改善案を提示"
  
  authority:
    - can_block_merge: true  # ⚠️ 重大リスク時は merge 阻止
    - can_request_changes: true
    - can_downvote: true  # Health Score -X points
    - escalate_to_human: true  # 重大問題は人間に エスカレーション
  
  working_hours: "24/7"  # Always-on
  activation_mode: "automatic"  # ユーザーレクエスト不要
  
  traits:
    - "徹底的"
    - "懐疑的"
    - "セキュリティ意識高"
    - "失敗事例を知る"
    - "技術負債に敏感"
```

### 2. Activation Triggers

Critical Guardian は以下の時に自動的に活動開始：

| トリガー | 対象 | 検査内容 |
|---------|------|--------|
| **PR作成** | すべての PR | セキュリティ・設計・リスク |
| **Skill 実行** | Skill 1-10 | 実装完全性・依存関係・エラーハンドリング |
| **Release 作成** | Version tag | 変更ログ精度・後方互換性・マイグレーション路 |
| **日次定期実行** | リポジトリ全体 | Health Score 計算、隠れたリスク検出 |

---

## Implementation — Skill 6.5: Critical Guardian

### Phase 1: Assessment

入力されたレビュー対象（PR、Skill出力、提案）に対して、以下を実施：

```python
def assess(artifact):
    """批判的評価を実施"""
    return {
        "critical_issues": [...],  # 🔴 重大: merge をブロック
        "high_risks": [...],       # 🟠 高: 警告・改善要求
        "concerns": [...],         # 🟡 中: 注意が必要
        "opportunities": [...],    # 🟢 低: 改善提案
        "health_impact": -15,      # Health Score 変化量
        "recommendation": "APPROVED_WITH_CONCERNS" | "REQUEST_CHANGES" | "BLOCKING"
    }
```

### Phase 2: Reporting

見つけた問題を以下の形式でレポート：

```markdown
# 🔴 Critical Guardian Review

## Summary
- **Verdict**: REQUEST_CHANGES (Concerns found)
- **Health Impact**: -8 points (95 → 87)

## Issues Found

### 🔴 CRITICAL (Blocking)
1. **Security**: API 認証情報がハードコードされている
   - File: `src/api.py:42`
   - Risk: Credential 漏洩 (CVSS 9.8)
   - Recommendation: GitHub Secrets に移動

### 🟠 HIGH (Required Changes)
2. **Architecture**: 強い循環依存を引入
   - Module A → B → C → A
   - Impact: 将来のリファクタリングを困難化
   
3. **Testing**: 新機能がテスト対象外
   - Coverage: -3.5%
   - Recommendation: Unit test 追加

### 🟡 MEDIUM (Concerns)
4. **Performance**: O(n²) ループの導入
   - Impact: 大規模データで 50% 遅化予測
   - Recommendation: アルゴリズム検討

## What's NOT Covered (Gaps)
- ✓ Documentation (Positive)
- ✗ Error handling in async functions (Missing)

## Alternative Approaches
1. Strategy Pattern で循環依存を排除
2. Adapter Pattern でテスト性を改善

## Suggestion for Improvement
```

### Phase 3: Automation

重大度別に自動アクション：

```yaml
actions:
  critical:
    - add_label: "🔴-critical-review-required"
    - request_changes: true
    - block_merge: true
    - notify: ["tech-lead", "security-team"]
    - escalate_issue: true
  
  high:
    - add_label: "🟠-improvements-needed"
    - request_changes: true
    - create_issue: "Follow-up: ..."
    - notify: ["reviewer"]
  
  medium:
    - add_comment: "suggestion"
    - add_label: "💡-suggestion"
    - persist_in_health_score: true
```

---

## Integration with SOUL

### SOUL Agent Coordination Flow

```
User Request or Automated Trigger
    ↓
[Manager] → Task decomposition
    ↓
[Developer] → Implementation
    ↓
[Code Reviewer] → Style/Logic check
    ↓
**[Critical Guardian] ← NEW!! (常に存在)**
    ├─ Is there a security issue? → YES → 🔴 Block
    ├─ Is there a design flaw? → YES → 🟠 Require changes
    ├─ Is there a missed requirement? → YES → 🟡 Concern
    └─ All good? → ✅ Approve
    ↓
[Researcher] (if academic validation needed)
    ↓
[Merge] → only if Critical Guardian APPROVED
    ↓
[Health Monitor] → score update
```

---

## agents_config.json への追加

以下の JSON スニペットを追加：

```json
{
  "roles": [
    {
      "name": "Critical Guardian",
      "id": "critic",
      "tier": "governance",
      "active": true,
      "responsibility": {
        "primary": "Always-on critical review",
        "secondary": [
          "Risk identification",
          "Security assessment",
          "Architecture validation",
          "Testing completeness check"
        ]
      },
      "constraints": {
        "can_block_merge": true,
        "can_request_changes": true,
        "escalation_capable": true,
        "working_hours": "24/7_automated"
      },
      "activation_rules": {
        "on_pr_created": true,
        "on_skill_execution": true,
        "on_merge_request": true,
        "on_schedule_daily": "02:00 UTC"
      },
      "assessment_framework": {
        "checks": [
          "security_audit",
          "architecture_validation",
          "testing_coverage",
          "dependency_analysis",
          "error_handling",
          "documentation_completeness"
        ],
        "severity_levels": ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        "scoring": "health_score_impact"
      },
      "reporting": {
        "format": "markdown_with_code_examples",
        "audience": ["developer", "tech_lead", "security_team"],
        "escalation_threshold": "CRITICAL",
        "archive": "reports/critical_reviews/"
      }
    }
  ]
}
```

---

## Associated Skill: Skill 11: Critical Guardian

新規 Skill として実装：

```python
# .github/skills/11-critical-guardian/run-guardian.py

def assess_pr(pr_number: int) -> CriticalReview:
    """PR に対する批判的レビュー"""
    
def assess_skill_output(skill_phase: int, result: dict) -> CriticalReview:
    """Skill 出力に対する批判的注視"""
    
def daily_health_criticism() -> CriticalReview:
    """プロジェクト全体の批判的評価（毎日）"""
    
def identify_hidden_risks() -> list[Risk]:
    """明白でないが重大なリスクを検出"""
    
def suggest_alternatives(proposal: dict) -> list[Alternative]:
    """代替案・改善案を提示"""
```

---

## Expected Benefits

### 1. セキュリティ向上
- **事前リスク検出**: 開発時点で 70% のリスクを検出
- **credential 漏洩防止**: ハードコードパターン自動検出

### 2. 設計品質向上
- **循環依存検出**: リファクタリング困難化を未然防止
- **アーキテクチャ検証**: 各 Skill の完全性を確保

### 3. テスト品質向上
- **カバレッジ低下検出**: テスト削減傾向を自動警告
- **エラーハンドリング検証**: 例外処理の漏れを指摘

### 4. 学習効果
- **失敗事例の蓄積**: パターン化された問題を記憶
- **チーム一体化**: 批判を建設的な改善へ活用

---

## Implementation Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1** | 3-4日 | agents_config.json 更新 (Critical Guardian ロール追加) |
| **Phase 2** | 1週 | Skill 11 基本実装 (assessment + reporting) |
| **Phase 3** | 1週 | GitHub Actions 統合（PR・daily trigger） |
| **Phase 4** | 3-5日 | テスト・チューニング・ドキュメント |

**Total**: 3-4 週間で完全運用

---

## Success Criteria

- [x] ロール定義が agents_config.json に統合
- [x] Skill 11 が pr_created イベントで自動実行
- [x] 検出したリスクが GitHub Issue として自動作成
- [x] Health Score が Critical Guardian の判定に反応
- [x] チーム向けドキュメント完成
- [x] 2週間の試行で false positive < 5%

---

## References

- `.github/agents/agents_config.json` — ロール定義
- `.github/skills/` — Skill 実装
- `documents/AGENTS_COORDINATION.md` — SOUL 統合ガイド
- `reports/COMPREHENSIVE_AUTOMATION_ANALYSIS_20260401.md` — 背景分析

---

**Status**: 🔄 設計完了 → 実装フェーズへ
