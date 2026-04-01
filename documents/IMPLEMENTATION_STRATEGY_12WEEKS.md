---
title: "Complete Implementation Strategy: Skills 1-11 + Security + Dashboard"
description: "大規模自動化ロードマップの実装戦略（12週間）"
date: "2026-04-01"
version: "1.0"
---

# 大規模ロードマップ実装戦略 - Complete Guide

## 📋 目次

1. [最適な実装順序](#1-最適な実装順序)
2. [各Skillの実装方法・コード構造](#2-各skillの実装方法コード構造)
3. [段階的なマイルストーン設定](#3-段階的なマイルストーン設定)
4. [各フェーズで検証すべき項目](#4-各フェーズで検証すべき項目)
5. [実装時に注意すべき点](#5-実装時に注意すべき点)
6. [並列実装可能部分](#6-並列実装可能部分)
7. [依存関係マッピング](#7-依存関係マッピング)

---

## 1. 最適な実装順序

### 全体タイムライン（12週間）

リスク最小化と依存関係を考慮した実装順序：

```
Week 1-2   → Phase 1: Foundation (Security + Skill 1-2)
Week 3-6   → Phase 2: Core Skills (Skill 2-5)
Week 7-9   → Phase 3: Advanced & Monitoring (Skill 7-11 + Actions)
Week 10-12 → Phase 4: Integration & Dashboard
```

### Phase 1: Foundation (Week 1-2)

**Week 1: Security基盤 + Skill 1**

```
Day 1-2: セットアップ
├─ Dockerfile/requirements.txt 確認
├─ GitHub Secrets 初期化
└─ dev container 環境検証

Day 3-5: Security 基盤（共有インフラ）
├─ scripts/audit/
│  ├─ audit_logger.py              ← Who, What, When, Where, Why 記録
│  └─ audit_log_schema.py           ← JSON Schema 定義
├─ scripts/security/
│  ├─ rbac_manager.py              ← Role-based Access Control
│  ├─ secrets_vault.py             ← GitHub Secrets 暗号化管理
│  └─ permission_checker.py        ← Sync 検証
└─ テスト: audit log 出力確認

Day 6-7: Skill 1 実装
├─ .github/skills/01-static-check/
│  ├── checkers/
│  │   ├── pyright_checker.py      ← Python 型チェック
│  │   ├── mypy_checker.py         ← MyPy 統合
│  │   ├── pytest_runner.py        ← テスト実行
│  │   ├── docker_validator.py     ← Docker イメージ検証
│  │   └── coverage_analyzer.py    ← Coverage 分析
│  ├── reporters/
│  │   └── report_generator.py     ← JSON/MD 出力
│  └── run-check.py                ← CLI エントリー
└─ テスト: Python/C++/MD サンプルで実行確認
```

**チェックリスト**:
- [ ] GitHub Secrets 設定（PYTHONPATH, DB_CONNECTION等）
- [ ] audit_logger.py 実装 ＆ テスト
- [ ] Skill 1 実行可（`python3 run-check.py --help`）
- [ ] GitHub Actions runner 起動確認

---

**Week 2: Skill 2 A層 + CI/CD初版**

```
Day 1-3: Skill 2 - Layer A (基礎検証)
├─ .github/skills/02-code-review/
│  ├── phases/
│  │   └── layer_a.py
│  │       ├─ 型注釈チェック（Pyright）
│  │       ├─ Docstring 検証
│  │       ├─ テストカバレッジ
│  │       └─ import 形式統一
│  ├── checkers/
│  │   ├── python_checker.py
│  │   └── markdown_checker.py
│  └── run-review.py --phase A
└─ テスト: `.code-review-SKILL.md` セクション 1-3 検証

Day 4-5: Skill 2 - Layer B/C (概要実装)
├─ phases/layer_b.py (スケルトン)
├─ phases/layer_c.py (スケルトン)
└─ 実装は Week 3 で完成

Day 6-7: GitHub Actions 初版ワークフロー
├─ .github/workflows/
│  ├── skills-1to2-pr.yml          ← PR 時実行
│  └── skill1-daily.yml            ← 日次実行
└─ テスト: PR 作成 → Actions 実行確認
```

**マイルストーン M1-M2**:
- ✅ Security 基盤稼働
- ✅ Skill 1-2 A層 実行可
- ✅ CI/CD pipeline 起動確認

---

### Phase 2: Core Skills (Week 3-6)

#### Week 3: Skill 2 完成 + Skill 3 開始

**Skill 2 - Layer B・C 完成**：

```python
# .github/skills/02-code-review/phases/layer_b.py
def run_checks(files: list[Path]) -> dict:
    """
    B層: 深度検証
    - スタイルガイド遵守
    - アーキテクチャパターン
    - テスト戦略
    """
    results = {
        "architecture_issues": [],
        "style_violations": [],
        "design_patterns": [],
    }
    # 実装: circular dependency 検出など
    return results
```

**Skill 3 開始**:

```
.github/skills/03-run-experiment/
├── run-exp.py                     ← エントリー
├── stages/
│   ├── stage1_init.py             ← Config 読込・Data 準備
│   ├── stage2_train.py            ← 学習実行
│   ├── stage3_validate.py         ← 検証
│   ├── stage4_test.py             ← テスト
│   └── stage5_report.py           ← レポート生成
├── param_sweep/
│   └── grid_searcher.py           ← ハイパーパラメータ探索
└── loggers/
    ├── json_logger.py
    └── csv_logger.py
```

---

#### Week 4: Skill 3 完成 + Skill 4 開始

**Skill 3: Run Experiment**

実装ポイント：
- 5段階パイプライン（init → train → validate → test → report）
- 失敗時は段階ごと再開可能（checkpoint)
- パラメータ探索自動化

```python
# .github/skills/03-run-experiment/run-exp.py
def pipeline(config: dict) -> dict:
    """
    実験パイプライン（5段階）
    各段階で checkpoint 保存 → 失敗時再開可能
    """
    state = {"config": config, "stage": 1}
    
    state = stage1_init.run(state)           # Data 準備
    state = stage2_train.run(state)          # 学習
    state = stage3_validate.run(state)       # 検証
    state = stage4_test.run(state)           # テスト
    report = stage5_report.run(state)        # レポート
    
    return report
```

**Skill 4 実装開始**:

```
.github/skills/04-critical-review/
├── run-critical.py
├── experts/
│   ├── reviewer.py                ← コード・設計レビュー観点
│   ├── statistician.py            ← 統計的妥当性
│   └── domain_expert.py           ← 学術内容・数値安定性
└── scoring/
    └── score_calculator.py        ← 3視点から総合スコア
```

---

#### Week 5: Skill 4 完成 + Skill 5 開始

**Skill 4: Critical Review**

3視点から入力を評価：

```python
def assess(artifact: dict) -> dict:
    """
    Reviewer, Statistician, Domain Expert の 3視点評価
    各視点：独立して 0-100 スコア計算
    総合スコア = weight(0.3) * reviewer + weight(0.4) * stat + weight(0.3) * domain
    """
    scores = {
        "reviewer": reviewer.assess(artifact),        # セキュリティ・設計
        "statistician": statistician.assess(artifact), # 統計的方法論
        "domain_expert": domain_expert.assess(artifact), # 学術妥当性
    }
    overall = (
        0.3 * scores["reviewer"] +
        0.4 * scores["statistician"] +
        0.3 * scores["domain_expert"]
    )
    return {
        **scores,
        "overall": overall,
        "verdict": "APPROVED" if overall >= 80 else "REQUEST_CHANGES"
    }
```

**Skill 5 実装開始**:

```
.github/skills/05-research-workflow/
├── run-workflow.py
├── managers/
│   ├── branch_manager.py          ← Worktree・ブランチ管理
│   ├── experiment_tracker.py      ← 実験メタデータ
│   └── metadata_handler.py        ← Config・成果物管理
├── analyzers/
│   ├── meta_analyzer.py           ← クロス実験分析
│   └── trend_analyzer.py          ← 長期トレンド
└── outputs/
    └── workflow_report.py
```

---

#### Week 6: Skill 5 完成 + 統合テスト

**Skill 5 実装**:

```python
# .github/skills/05-research-workflow/run-workflow.py
def track_experiment(exp_id: str) -> dict:
    """
    長期研究管理
    - ブランチ管理（create, merge, cleanup）
    - メタ分析（パラメータ別成績、傾向）
    - レポート自動生成
    """
    branch_manager.setup_worktree(exp_id)
    results = meta_analyzer.analyze_across_exps()
    trends = trend_analyzer.detect_patterns()
    
    return {
        "experiment_id": exp_id,
        "branch_status": "active",
        "trends": trends,
        "recommendations": [],
    }
```

**統合テスト**:

```bash
# Skills 1-5 統合パイプライン
.github/workflows/full-integration-test.yml

1. Skill 1: Static Check → OK
2. Skill 2: Code Review (A/B/C層) → OK
3. Skill 3: Run Experiment (簡易版) → OK
4. Skill 4: Critical Review → OK
5. Skill 5: Research Workflow → OK
```

**マイルストーン M3-M4**:
- ✅ Skill 1-5 すべて実行可
- ✅ 統合テスト成功
- ✅ 使用例ドキュメント完成

---

### Phase 3: Advanced & Monitoring (Week 7-9)

#### Week 7: Skill 7 + Skill 11 並行実装

**Skill 7: Continuous Health Monitor**

```
.github/skills/07-health-monitor/
├── run-monitor.py                 ← 常時監視 CLI
├── collectors/
│   ├── code_quality_collector.py  ← Pyright/Mypy 通過率
│   ├── test_coverage_collector.py ← Coverage % トレンド
│   ├── security_collector.py      ← Vulnerability 数
│   ├── doc_coverage_collector.py  ← Doc % トレンド
│   ├── perf_collector.py          ← ビルド時間・テスト時間
│   └── dependency_collector.py    ← パッケージ freshness
├── scorers/
│   └── health_scorer.py           ← 0-100 スコア計算
└── logger/
    └── health_logger.py           ← Daily / Hourly logging
```

**Health Score 計算**:

```python
def calculate_health_score() -> dict:
    """
    プロジェクト健全性 0-100
    各コンポーネント 0-100 をスコア化
    """
    code_quality = 100 - (type_errors * 0.5)            # max 100
    testing = coverage_percent                           # 0-100
    security = max(100 - vulns * 2, 0)                  # 0-100
    documentation = doc_coverage                         # 0-100
    performance = calculate_perf_score()                 # 0-100
    dependencies = 100 - (outdated_pkgs * 1)            # 0-100
    
    # 重み付け平均
    overall = (
        0.25 * code_quality +
        0.25 * testing +
        0.20 * security +
        0.15 * documentation +
        0.10 * performance +
        0.05 * dependencies
    )
    
    return {
        "overall_score": round(overall, 1),
        "components": {
            "code_quality": code_quality,
            "testing": testing,
            "security": security,
            "documentation": documentation,
            "performance": performance,
            "dependencies": dependencies,
        },
        "trend": "📈 +3 from yesterday",
        "timestamp": datetime.now().isoformat(),
    }
```

**Skill 11: Critical Guardian**

```
.github/skills/11-critical-guardian/
├── run-guardian.py                ← 常時批判的レビュー
├── assessor/
│   ├── risk_detector.py           ← リスク検出
│   ├── security_checker.py        ← Sec issue 指摘
│   ├── design_analyzer.py         ← 設計問題
│   └── gap_finder.py              ← 実装漏れ検出
├── reporter/
│   └── finding_reporter.py        ← GitHub Comments・Issues
└── rules/
    └── assessment_rules.yaml      ← ルール定義（カスタマイズ可）
```

**Guardian Logic**:

```python
def assess_pr(pr: dict) -> dict:
    """
    PR に対して常に批判的立場からレビュー
    重大度別に判定
    """
    critical_issues = []
    high_risks = []
    concerns = []
    
    # リスク検出パイプライン
    for issue in risk_detector.detect(pr):
        if issue.severity == "CRITICAL":
            critical_issues.append(issue)
        elif issue.severity == "HIGH":
            high_risks.append(issue)
        else:
            concerns.append(issue)
    
    # ブロック判定
    verdict = (
        "BLOCKING" if critical_issues else
        "REQUEST_CHANGES" if high_risks else
        "APPROVED_WITH_CONCERNS" if concerns else
        "APPROVED"
    )
    
    return {
        "critical_issues": critical_issues,
        "high_risks": high_risks,
        "concerns": concerns,
        "verdict": verdict,
        "health_impact": -len(critical_issues) * 5 - len(high_risks) * 2,
    }
```

**並行実装理由**：
- Skill 7 と 11 は完全に独立
- 両者 Health Score を参考情報として使用（作成後活用）
- 並行で進めれば Week 7 末時点で両方稼働

---

#### Week 8: Skill 8 実装 + GitHub Actions ワークフロー

**Skill 8: Security & Compliance Audit**

```
.github/skills/08-security-audit/
├── run-audit.py                   ← セキュリティ監査 CLI
├── scanners/
│   ├── secret_scanner.py          ← gitleaks 統合
│   ├── sbom_generator.py          ← Software Bill of Materials
│   ├── license_checker.py         ← License compliance
│   ├── rbac_validator.py          ← RBAC 検証
│   └── dependency_scanner.py      ← Trivy/Safety 統合
├── reporters/
│   └── compliance_reporter.py     ← JSON + GitHub Issues
└── config/
    └── security_rules.yaml
```

**実行例**:

```bash
python3 .github/skills/08-security-audit/run-audit.py \
  --scan-secrets \
  --generate-sbom \
  --check-licenses \
  --trivy-severity HIGH,CRITICAL \
  --output json
```

**GitHub Actions ワークフロー（6種類）**:

1. **skills-1to5-daily.yml** - 日次（Skill 1-5）
2. **health-monitor-hourly.yml** - 毎時（Skill 7）
3. **security-audit-weekly.yml** - 週次（Skill 8）
4. **perf-metrics-daily.yml** - 日次（Skill 9）
5. **critical-guardian-pr.yml** - PR 作成時（Skill 11）
6. **critical-guardian-daily.yml** - 日次（Skill 11）

```yaml
# .github/workflows/health-monitor-hourly.yml
name: Hourly Health Monitor
on:
  schedule:
    - cron: '0 * * * *'  # Every hour

jobs:
  monitor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Run Health Monitor
        run: |
          python3 .github/skills/07-health-monitor/run-monitor.py \
            --output json \
            --save-report
      
      - name: Post to Dashboard
        # Dashboard への結果送信
```

---

#### Week 9: Skill 9 + Skill 10 実装

**Skill 9: Performance Metrics Collector**

```
.github/skills/09-perf-metrics/
├── run-metrics.py
├── collectors/
│   ├── ci_time_analyzer.py        ← CI build time
│   ├── test_runtime_analyzer.py   ← Test duration by type
│   └── docker_build_analyzer.py   ← Docker build time
├── storage/
│   └── metrics_db.py              ← SQLite or JSON storage
└── outputs/
    └── trend_visualizer.py        ← PNG/SVG チャート生成
```

**Skill 10: Dependency & Conflict Detector**

```
.github/skills/10-dependency-detector/
├── run-detector.py
├── analyzers/
│   ├── conflict_detector.py       ← Version conflict
│   ├── circular_dependency.py     ← Circular deps (pipdeptree)
│   ├── deprecated_scanner.py      ← Deprecated packages
│   └── license_incompatibility.py ← MIT vs GPL など
├── suggesters/
│   └── update_suggester.py        ← PR suggestion generator
└── reporters/
    └── dependency_reporter.py
```

**マイルストーン M5**:
- ✅ Skill 7-11 すべて実行可
- ✅ GitHub Actions 6ワークフロー稼働
- ✅ Dashboard 基盤準備

---

### Phase 4: Integration & Dashboard (Week 10-12)

#### Week 10: ダッシュボード実装

```
web/
├── app.py                         ← Flask/FastAPI エントリー
├── static/
│   ├── index.html
│   ├── dashboard.js               ← React/Vue.js
│   └── styles.css
├── templates/
│   ├── health_widget.html
│   ├── metrics_widget.html
│   └── alerts_widget.html
└── backend/
    ├── dashboard_core.py          ← データ集計
    ├── metrics_aggregator.py      ← Skill 結果統合
    ├── websocket_server.py        ← Real-time sync
    └── db/
        └── metrics_storage.py
```

**ダッシュボード機能**:

| Widget | Data Source | Update Frequency |
|--------|------------|-----------------|
| Health Score | Skill 7 | Hourly |
| Code Quality | Skill 1-2 | Daily |
| Test Results | Skill 1 | On PR |
| Security Status | Skill 8 | Weekly |
| Performance Trends | Skill 9 | Daily |
| Critical Issues | Skill 11 | Real-time |

```python
# web/backend/dashboard_core.py
@app.route("/api/health-score")
def get_health_score():
    """Health Score リアルタイム取得"""
    score = metrics_storage.get_latest_health_score()
    trend = metrics_storage.get_score_trend(days=7)
    return {
        "current": score,
        "trend": trend,
        "timestamp": datetime.now().isoformat(),
    }

@app.websocket("/ws/metrics")
async def websocket_metrics(websocket):
    """WebSocket: Real-time metrics stream"""
    while True:
        data = await metrics_aggregator.get_latest()
        await websocket.send_json(data)
        await asyncio.sleep(5)  # 5秒ごとに送信
```

---

#### Week 11: 統合テスト + ドキュメント

**E2E テスト**:

```python
# tests/e2e_test.py
def test_full_pipeline():
    """Skills 1-11 完全実行テスト"""
    
    # Setup
    test_config = load_config("tests/fixtures/test_config.yaml")
    
    # Run pipeline
    results = run_pipeline(test_config)
    
    # Assertions
    assert results["skill1"]["status"] == "PASS"
    assert results["skill2"]["pass_rate"] > 0.9
    assert results["skill3"]["duration"] < 600  # < 10 min
    assert results["dashboard"]["health_score"] > 0
    
    # Cleanup
    cleanup_artifacts()
```

**ドキュメント自動生成**:

```bash
# docs/ ディレクトリに自動生成
docs/
├── API_REFERENCE.md       ← 各 Skill の CLI リファレンス
├── USAGE_EXAMPLES.md      ← 使用例・よくある質問
├── TROUBLESHOOTING.md     ← トラブルシューティング
└── ARCHITECTURE.md        ← 全体アーキテクチャ図
```

---

#### Week 12: 本番導入準備

**本番環境チェックリスト**:

```yaml
Production Readiness Checklist:

□ Infrastructure
  □ GitHub Secrets 本番設定完了
  □ Audit ログ永続化（DB or Cloud Storage）
  □ Disaster recovery plan 策定
  □ Backup/Restore テスト成功

□ Security
  □ レート制限設定（API endpoints）
  □ SQL injection 対策完了
  □ RBAC 権限サニティ確認
  □ Secret scanning 有効化

□ Performance
  □ Load test (1000 concurrent) 成功
  □ Database indexing 最適化完了
  □ Caching strategy 実装

□ Monitoring
  □ Error tracking (Sentry等) 統合
  □ Alert rules 設定完了
  □ Dashboard alert 正常動作確認

□ Documentation
  □ 運用ガイド完成
  □ Onboarding 資料完成
  □ チーム トレーニング実施

□ Governance
  □ Audit log retention policy
  □ Data retention policy
  □ Incident response plan
```

**Go-Live**:

```bash
# 本番環境への切り替え
1. 全 Skill 最終チェック
2. GitHub Actions 本番設定
3. Dashboard 本番 URL 公開
4. チーム全員に通知
5. 運用開始
```

---

## 2. 各Skillの実装方法・コード構造

### Skill 1: Static Check Structure

```
.github/skills/01-static-check/
├── README.md
├── run-check.py                   ← Main CLI
├── checkers/
│   ├── __init__.py
│   ├── pyright_checker.py
│   │   ├─ def run() -> CheckResult
│   │   ├─ def parse_output() -> List[Issue]
│   │   └─ def format_report() -> str
│   ├── mypy_checker.py
│   ├── pytest_runner.py
│   │   ├─ def run_tests() -> TestResult
│   │   ├─ def analyze_failures() -> List[FailureAnalysis]
│   │   └─ def calculate_coverage() -> float
│   ├── docker_validator.py
│   │   ├─ def validate_image() -> ValidationResult
│   │   ├─ def check_vulnerabilities() -> List[Vulnerability]
│   │   └─ def analyze_layers() -> LayerAnalysis
│   └── coverage_analyzer.py
├── reporters/
│   ├── __init__.py
│   └── report_generator.py
│       ├─ class Report
│       ├─ def to_json() -> str
│       ├─ def to_markdown() -> str
│       └─ def merge_results() -> Report
├── config/
│   └── default_config.yaml
└── tests/
    ├── test_pyright_checker.py
    ├── test_pytest_runner.py
    └── fixtures/
        └── sample_project/
```

**Key Implementation Points**:

```python
# checkers/pyright_checker.py
from dataclasses import dataclass
from typing import List
import subprocess
import json

@dataclass
class TypeCheckResult:
    total_errors: int
    errors_by_file: dict[str, int]
    duration: float
    status: str  # "PASS" | "FAIL"

def run(workspace_root: str, verbose: bool = False) -> TypeCheckResult:
    """
    Python 型チェック (Pyright)
    
    Args:
        workspace_root: プロジェクトルート
        verbose: 詳細出力フラグ
    
    Returns:
        型チェック結果
    
    Raises:
        RuntimeError: Pyright 実行失敗時
    """
    try:
        result = subprocess.run(
            ["pyright", "--outputjson", workspace_root],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        output = json.loads(result.stdout)
        
        return TypeCheckResult(
            total_errors=output.get("summary", {}).get("errorCount", 0),
            errors_by_file=parse_errors_by_file(output),
            duration=output.get("general", {}).get("executionTime", 0),
            status="PASS" if output.get("summary", {}).get("errorCount", 0) == 0 else "FAIL",
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Pyright timeout exceeds 60 seconds")
    except json.JSONDecodeError:
        raise RuntimeError("Pyright output JSON decode error")

def parse_errors_by_file(output: dict) -> dict[str, int]:
    """Pyright JSON から ファイル別エラー数を抽出"""
    errors_by_file = {}
    for diagnostic in output.get("generalDiagnostics", []):
        file_path = diagnostic.get("file", "unknown")
        errors_by_file[file_path] = errors_by_file.get(file_path, 0) + 1
    return errors_by_file
```

---

### Skill 2: Code Review 3層構造

```
.github/skills/02-code-review/
├── README.md
├── run-review.py                  ← CLI エントリー
├── phases/
│   ├── __init__.py
│   ├── layer_a.py                 ← 自動型検証、形式チェック
│   │   ├─ def run_checks() -> ReviewResult
│   │   ├─ check_type_annotations()
│   │   ├─ check_docstrings()
│   │   └─ check_test_coverage()
│   ├── layer_b.py                 ← 深度検証（アーキテクチャ）
│   │   ├─ def run_checks() -> ReviewResult
│   │   ├─ detect_circular_deps()
│   │   ├─ check_design_patterns()
│   │   └─ analyze_coupling()
│   └── layer_c.py                 ← 統合検証
│       ├─ def run_checks() -> ReviewResult
│       ├─ verify_doc_test_triplet()
│       ├─ check_convention_consistency()
│       └─ detect_unmaintainable_code()
├── checkers/
│   ├── __init__.py
│   ├── python_checker.py
│   ├── cpp_checker.py
│   └── markdown_checker.py
└── reporters/
    └── review_report.py
```

**Design Pattern**: Strategy + Chain of Responsibility

```python
# run-review.py
class CodeReviewStrategy:
    def review(self, files: List[Path]) -> ReviewResult:
        raise NotImplementedError

class LayerAReview(CodeReviewStrategy):
    def review(self, files: List[Path]) -> ReviewResult:
        from phases.layer_a import run_checks
        return run_checks(files)

class LayerBReview(CodeReviewStrategy):
    def review(self, files: List[Path]) -> ReviewResult:
        from phases.layer_b import run_checks
        return run_checks(files)

class LayerCReview(CodeReviewStrategy):
    def review(self, files: List[Path]) -> ReviewResult:
        from phases.layer_c import run_checks
        return run_checks(files)

def main(phase: str):
    strategy = {
        "A": LayerAReview(),
        "B": LayerBReview(),
        "C": LayerCReview(),
    }[phase]
    
    result = strategy.review(get_modified_files())
    print(result.to_markdown())
```

---

### Skill 3: Run Experiment (5段階パイプライン)

**アーキテクチャ**: Pipeline + State Pattern

```python
# .github/skills/03-run-experiment/run-exp.py

@dataclass
class ExperimentState:
    """実験状態管理（各段階間で引き継ぎ）"""
    config: dict
    stage: int
    data: dict = None
    model: dict = None
    results: dict = None
    metadata: dict = None

def run_pipeline(config_path: str) -> dict:
    """
    5段階実験パイプライン
    各段階で checkpoint → 失敗時再開可能
    """
    config = load_config(config_path)
    state = ExperimentState(config=config, stage=1)
    
    try:
        # Stage 1: Initialize
        state = stage1_init.run(state)
        save_checkpoint(state, "stage1")
        
        # Stage 2: Train
        state = stage2_train.run(state)
        save_checkpoint(state, "stage2")
        
        # Stage 3: Validate
        state = stage3_validate.run(state)
        save_checkpoint(state, "stage3")
        
        # Stage 4: Test
        state = stage4_test.run(state)
        save_checkpoint(state, "stage4")
        
        # Stage 5: Report
        report = stage5_report.run(state)
        save_checkpoint(report, "final")
        
        return report
    
    except Exception as e:
        # エラーハンドリング + ログ記録
        audit_logger.log(
            action="experiment_failed",
            exp_id=config.get("experiment_id"),
            stage=state.stage,
            error=str(e),
        )
        raise
```

---

### Skill 4: Critical Review (3視点評価)

**設計**: Composite Pattern + Scoring

```python
# .github/skills/04-critical-review/run-critical.py

class ExpertEvaluator:
    """各視点による評価インターフェース"""
    def assess(self, artifact: dict) -> float:
        """0-100 スコア"""
        raise NotImplementedError

class CodeReviewer(ExpertEvaluator):
    def assess(self, artifact: dict) -> float:
        """
        コード・セキュリティ・設計レビュー観点
        チェック項目:
        - Security (API key exposed? ==> -50点)
        - Architecture (circular deps? ==> -30点)
        - Maintainability (type hints complete? ==> +10点)
        """
        score = 100
        # Detailed evaluation logic
        return max(0, min(100, score))

class Statistician(ExpertEvaluator):
    def assess(self, artifact: dict) -> float:
        """
        統計的方法論の妥当性
        チェック項目:
        - Sample size adequacy
        - Statistical test selection
        - Multiple testing correction
        """
        pass

class DomainExpert(ExpertEvaluator):
    def assess(self, artifact: dict) -> float:
        """
        学術・学域固有の妥当性
        チェック項目:
        - Algorithm correctness
        - Numerical stability
        - Citation completeness
        """
        pass

def final_verdict(scores: dict) -> str:
    """複数スコアから最終判定"""
    overall = (
        0.3 * scores["reviewer"] +
        0.4 * scores["statistician"] +
        0.3 * scores["domain_expert"]
    )
    
    if overall >= 85:
        return "APPROVED"
    elif overall >= 70:
        return "APPROVED_WITH_MINOR_CHANGES"
    elif overall >= 50:
        return "REQUEST_CHANGES"
    else:
        return "REJECTED"
```

---

### Skill 5-11: 簡略定義

**Skill 5: Research Workflow**

```python
def track_meta_experiments(exp_root: Path) -> MetaAnalysis:
    """
    クロス実験分析
    - パラメータ vs 性能 トレンド
    - 最適ハイパーパラメータ候補
    - 失敗パターン学習
    """
    pass
```

**Skill 7: Health Monitor**

```python
def calculate_health_score() -> HealthReport:
    """プロジェクト健全性 0-100（毎時更新）"""
    pass
```

**Skill 8: Security Audit**

```python
def run_security_audit() -> ComplianceReport:
    """Secret scanning + License + RBAC validation"""
    pass
```

**Skill 9-10, 11**: 類似の実装パターン

---

## 3. 段階的なマイルストーン設定

### M0: Pre-Phase Validation ✓

**必須項目**:
- GitHub Secrets 設定完了
- Dockerfile + requirements.txt 確認済み
- dev container 環境 OK
- チーム権限設定完了

---

### M1: Security Foundation (Week 1 end) ✅

**検証項目**:
- [ ] `audit_logger.py` 実装 & テスト完了
  - テスト: 5 件のアクション記録 → JSON 형식 검증
  - テスト: 秘密情報がログに含まれ**ない**
- [ ] `rbac_manager.py` 実装 & テスト完了
- [ ] GitHub Secrets 連携確認
- [ ] CI Runner 起動確認（1回以上実行）

**Go/No-Go Decision**: 合否判定
- ✅ 合格: audit + rbac + CI が稼働
- ❌ 不合格: Security 基盤やり直し（Week 1 内）

---

### M2: Skill 1-2 Readiness (Week 2 end) ✅

**実装完了条件**:
- [ ] Skill 1: `run-check.py --help` 実行可
  - 引数: `--include-docker`, `--coverage-threshold`, `--output`
  - テスト: Python サンプル 2 ファイルでチェック実行 ← OK
  - テスト: Docker image 検証 ← OK
- [ ] Skill 2 A層: `run-review.py --phase A` 実行可
  - Test output JSON schema 検証
  
**ドキュメント完成**:
- [ ] USAGE_EXAMPLES.md: Skill 1-2 使用例 5+ 件
- [ ] Troubleshooting セクション: 一般的なエラー対応

**Go/No-Go**:
- ✅ 合格: Skill 1-2 A層が動作
- ❌ 不合格: 実装やり直し

---

### M3: Core Skills Phase (Week 6 end) ✅

**実装完了条件**:
- [ ] Skill 2: B層・C層 実装完了
- [ ] Skill 3: Run Experiment 5段階動作確認
  - テスト: 簡易実験（config.yaml を用意）実行 ← OK
  - テスト: checkpoint 動作確認
- [ ] Skill 4: 3視点評価機能動作確認
  - テスト: サンプルアーティファクト → 3 スコア出力
- [ ] Skill 5: Meta-analysis & Trend tracking 実装

**統合テスト**:
- [ ] Skills 1-5 パイプライン実行 ← 0 errors
- [ ] Duration < 10 min (normal run)

**Go/No-Go**:
- ✅ 合格: Skill 1-5 パイプライン成功
- ❌ 不合格: 個別スキル再実装

---

### M4: Advanced Monitoring (Week 9 end) ✅

**実装完了条件**:
- [ ] Skill 7: Health Score 毎時計算 ← 稼働確認
  - テスト: 過去 24 時間のスコア履歴取得 ← OK
- [ ] Skill 11: PR 作成時 Critical Guardian 実行 ← 確認
- [ ] Skill 8: Security Audit 実行可
  - テスト: secret scanning 実行 ← OK
  - テスト: SBOM 生成 ← OK
- [ ] Skill 9-10: Performance & Dependency 実装完了

**GitHub Actions**:
- [ ] 6ワークフロー登録完了（cron・event トリガー確認）
- [ ] 最初の 1 週間のスケジュール実行結果確認

**Go/No-Go**:
- ✅ 合格: 全 Skill + Actions 稼働
- ❌ 不合格: CLI デバッグ + Actions 再設定

---

### M5: Dashboard Ready (Week 10 end) ✅

**ダッシュボード実装完了**:
- [ ] Web UI 起動可（localhost:5000）
- [ ] Health Score Widget リアルタイム表示
- [ ] Metrics aggregation 動作確認
- [ ] API endpoints テスト ← 200 OK

**パフォーマンステスト**:
- [ ] Load test: 100 concurrent users ← latency < 2sec

**Go/No-Go**:
- ✅ 合格: Dashboard + API 動作
- ❌ 不合格: バックエンド最適化

---

### M6: Go-Live (Week 12 end) 🚀

**本番環境チェック**:
- [ ] All GitHub Secrets set in production
- [ ] Audit logging to persistent storage
- [ ] Disaster recovery tested
- [ ] チーム全員 onboarding 完了
- [ ] 運用マニュアル署名（承認）完了

**出荷可能条件**:
- ✅ Core Skill すべて実行可
- ✅ Actions すべて本番トリガー設定
- ✅ Dashboard 本番 URL 公開
- ✅ Incident response plan 策定完了

**Go-Live Execution**:
1. 本番環境 final check (1日)
2. Team notification
3. Dashboard public URL 切り替え
4. Monitoring alert 有効化

---

## 4. 各フェーズで検証すべき項目

### Phase 1-2: Unit & Integration Tests

```python
# tests/test_skill1.py (Pytest)

def test_pyright_checker_pass():
    """型チェック成功パス"""
    result = pyright_checker.run("tests/fixtures/valid_code")
    assert result.total_errors == 0
    assert result.status == "PASS"

def test_pyright_checker_fail():
    """型エラー検出"""
    result = pyright_checker.run("tests/fixtures/invalid_code")
    assert result.total_errors > 0
    assert result.status == "FAIL"

def test_security_no_secrets_in_logs():
    """Audit ログに秘密情報がない"""
    logs = audit_logger.fetch_logs(limit=100)
    for log in logs:
        assert "password" not in log.lower()
        assert "api_key" not in log.lower()
        assert "token" not in log.lower()

def test_rbac_permission_denied():
    """権限なしユーザーはアクセス不可"""
    with pytest.raises(PermissionError):
        rbac_manager.check_permission(user="dev", resource="deploy", action="deploy")

def test_skill1_integration():
    """Skill 1 統合テスト"""
    result = subprocess.run(
        ["python3", ".github/skills/01-static-check/run-check.py", \
         "--output", "json", "tests/fixtures"],
        capture_output=True,
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert "summary" in output
    assert "details" in output
```

---

### Phase 3-6: Pipeline & State Tests

```python
def test_experiment_pipeline_end_to_end():
    """実験パイプライン完全実行"""
    config = load_config("tests/configs/simple_exp.yaml")
    report = run_pipeline(config_path)
    
    # 5段階すべて実行
    assert report["stage"] == 5
    assert "results" in report
    assert "metrics" in report
    
    # Checkpoint テスト
    state_file = Path(".checkpoints/stage4.pkl")
    assert state_file.exists()

def test_critical_review_scoring():
    """3視点スコア計算"""
    artifact = {
        "code": open("tests/fixtures/sample.py").read(),
        "tests": "...",
    }
    result = assess(artifact)
    
    assert 0 <= result["reviewer"] <= 100
    assert 0 <= result["statistician"] <= 100
    assert 0 <= result["domain_expert"] <= 100
    assert result["overall"] == \
        0.3 * result["reviewer"] + \
        0.4 * result["statistician"] + \
        0.3 * result["domain_expert"]
```

---

### Phase 7-9: Continuous Monitoring Tests

```python
def test_health_score_hourly_update():
    """Health Score 毎時更新"""
    score_t1 = get_health_score()
    time.sleep(3600)  # 1 hour
    score_t2 = get_health_score()
    
    assert score_t1["timestamp"] != score_t2["timestamp"]
    assert "overall_score" in score_t2
    assert 0 <= score_t2["overall_score"] <= 100

def test_critical_guardian_pr_trigger():
    """PR 作成時に Guardian 実行"""
    # PR 作成
    pr = create_test_pr()
    
    # GitHub Actions で Guardian が実行（待機）
    time.sleep(60)  # Wait for workflow
    
    # Comment を確認
    comments = get_pr_comments(pr.number)
    assert any("Critical Guardian" in c.body for c in comments)

def test_security_audit_output():
    """Security Audit 出力形式"""
    report = run_security_audit()
    
    assert "secret_scan" in report
    assert "sbom" in report
    assert "license_check" in report
    assert "rbac_validation" in report
```

---

### Phase 10-12: E2E & Production Readiness

```python
def test_dashboard_e2e():
    """ダッシュボード完全テスト"""
    # 1. Web UI 起動
    app.run_async(port=5000)
    
    # 2. Page load
    response = requests.get("http://localhost:5000/dashboard")
    assert response.status_code == 200
    
    # 3. API データ取得
    health = requests.get("http://localhost:5000/api/health").json()
    assert health["overall_score"] > 0
    
    # 4. WebSocket リアルタイム
    with websocket.create_connection("ws://localhost:5000/ws/metrics") as ws:
        data = json.loads(ws.recv())
        assert "metrics" in data

@pytest.mark.load
def test_load_1000_concurrent():
    """1000 concurrent users"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1000) as executor:
        futures = [
            executor.submit(requests.get, "http://localhost:5000/api/health")
            for _ in range(1000)
        ]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    success_rate = sum(1 for r in results if r.status_code == 200) / len(results)
    assert success_rate > 0.99  # 99%以上成功
```

---

## 5. 実装時に注意すべき点

### 🔒 セキュリティ

1. **秘密管理**
   ```python
   # ❌ 悪い例
   API_KEY = "sk-abc123xyz"
   
   # ✅ 良い例
   API_KEY = os.getenv("GITHUB_API_KEY")
   if not API_KEY:
       raise RuntimeError("GITHUB_API_KEY not set in secrets")
   ```

2. **入力検証**
   ```python
   # CVE-2025-XXXX: Path traversal 対策
   def read_config(config_path: str) -> dict:
       # ✅ Path を normalize + 範囲チェック
       safe_path = Path(config_path).resolve()
       workspace = Path("/workspace").resolve()
       
       if not safe_path.is_relative_to(workspace):
           raise ValueError("Path outside workspace")
       
       return json.load(open(safe_path))
   ```

3. **Audit Logging**
   ```python
   @audit_log("important_action")
   def deploy_skill(skill_id: str, version: str):
       # Who, What, When, Where, Why を記録
       # ✓ User ID
       # ✓ Action
       # ✓ Timestamp
       # ✓ IP address
       # ✓ Resource modify
       pass
   ```

---

### ⚡ パフォーマンス

1. **タイムアウト設定**
   ```python
   # Skill 1 < 30 秒（GitHub Actions timeout 前）
   result = subprocess.run(
       ["pyright", ...],
       timeout=25,  # 5 秒 margin
       ...
   )
   ```

2. **キャッシング**
   ```python
    # Skill 7 Health Score 計算 < 5 秒
   @functools.lru_cache(maxsize=1)
   def calculate_health_score():
       # キャッシュ期限: 1 時間
       pass
   
   # Cache invalidate
   calculate_health_score.cache_clear()
   ```

3. **並列化**
   ```python
   # Skill 1-2 複数ファイル parallel check
   from concurrent.futures import ThreadPoolExecutor
   
   with ThreadPoolExecutor(max_workers=4) as executor:
       results = executor.map(check_file, files)
   ```

---

### 🔄 信頼性

1. **エラーハンドリング**
   ```python
   try:
       result = pyright_checker.run()
   except subprocess.TimeoutExpired:
       # Fallback: cached result or "UNKNOWN"
       return {"status": "TIMEOUT", "cached": True}
   except FileNotFoundError:
       # Pyright not installed
       return {"status": "TOOL_NOT_FOUND"}
   except Exception as e:
       audit_logger.log_error(action="check_failed", error=str(e))
       raise
   ```

2. **Retry Logic**
   ```python
   @retry(max_attempts=3, backoff=2)
   def call_external_api():
       # API 一時的なエラー時に再試行
       pass
   ```

3. **Graceful Degradation**
   ```python
   # Skill が何か失敗 → Core pipeline は続行、warning のみ
   def run_skill_safely(skill_func):
       try:
           return skill_func()
       except Exception as e:
           logger.warning(f"Skill failed: {e}")
           return {"status": "FAILED", "error": str(e)}
   ```

---

### 📚 保守性

1. **Type Hints**
   ```python
   # すべての関数に型注釈
   def calculate_health_score(
       components: dict[str, float],
       weights: dict[str, float] | None = None,
   ) -> HealthReport:
       """
       健全性スコア計算
       
       Args:
           components: コンポーネント別スコア (name -> 0-100)
           weights: コンポーネント別重み (name -> 0-1)
       
       Returns:
           Health report
       
       Raises:
           ValueError: Invalid input
       """
       pass
   ```

2. **Docstrings**
   ```python
   def assess(artifact: dict) -> dict:
       """
       3視点からアーティファクトを評価
       
       処理フロー:
       1. Code reviewer perspective (コード品質・セキュリティ)
       2. Statistician perspective (統計的妥当性)
       3. Domain expert perspective (学術内容)
       
       返り値:
       {
           "reviewer": 0-100,
           "statistician": 0-100,
           "domain_expert": 0-100,
           "overall": 0-100,
           "verdict": "APPROVED" | "REQUEST_CHANGES" | "REJECTED"
       }
       """
       pass
   ```

3. **テスト可能性**
   ```python
   # Dependency Injection で外部依存を inject
   class SkillRunner:
       def __init__(self, logger=None, config_loader=None):
           self.logger = logger or default_logger
           self.config_loader = config_loader or load_config_from_file
   ```

---

## 6. 並列実装可能部分

### Wave 分析

```
Week 1-2: Foundation (順序依存あり)
├─ Team A: Security 基盤 (audit, rbac)
├─ Team B: Skill 1 (checkers)
├─ Team C: Skill 2 A層 (linter)
└─ Team D: CI/CD + Testing

並列度: 4 teams (独立可能)
```

```
Week 3-6: Core Skills
├─ Track 1: Skill 2 B層・C層最適化
├─ Track 2: Skill 3 (5段階パイプライン)
├─ Track 3: Skill 4 (3視点評価)
├─ Track 4: Skill 5 (メタ分析)
│   └─ 依存: Skill 3-4 完了後
└─ Track 5: ドキュメント・テスト

並列度: 4 tracks (制約あり)
Bottleneck: Skill 3 → Skill 4 → Skill 5
```

```
Week 7-9: Advanced
├─ Fork A: Skill 7 (Health Monitor)
├─ Fork B: Skill 11 (Critical Guardian)
├─ Fork C: Skill 8 (Security Audit) → 独立
├─ Fork D: Skill 9-10 (Perf + Dependency) → 独立
├─ Fork E: GitHub Actions (全ワーカー共有)
└─ Fork F: Dashboard 基盤

並列度: 6 forks（ほぼ完全に独立）
```

---

### 依存関係グラフ（Critical Path）

```
 Day 0                  Week 1              Week 2             Week 3-6
 
┌──────────┐
│  Setup   │          ┌──────────────────┐
└────┬─────┘          │ Security         │  ┌──────────────┐
     │                │ Foundation       │  │ Skill 1-5    │
     └───────────────►│ (audit, rbac)    │  │ Impl & Test  │
                      └────┬─────────────┘  │              │
                           │                │ + Actions    │
                           └───────────────►│ + Dashboard  │
                                            └──────────────┘


Critical Path Length: 12 weeks (順序依存あり)

Non-critical:
- Skill 7-11 は Skills 1-5 完了後 parallel 可能
- Dashboard は並列開発可（Skill 1-10 完了待ち）
```

---

### Recommended Parallelization Strategy

| Phase | Teams | Duration | Bottleneck |
|-------|-------|----------|-----------|
| **Phase 1** | 4 teams | 2 weeks | Security → dependent |
| **Phase 2** | 4 tracks | 4 weeks | Skill 3 → 4 → 5 chain |
| **Phase 3** | 6 forks | 3 weeks | Dashboard wait |
| **Phase 4** | 3 squads | 3 weeks | E2E test |

**最大並列効率**: 6 teams simultaneously (Week 7-9)

---

## 7. 依存関係マッピング

### Explicit Dependency Graph

```
Security Infrastructure
    │
    ├─ Skill 1 ────┬─ Static Check (Pyright + Pytest)
    │              │
    │              ├─ Skill 2 (A層) ┬──── Code Review Phase A
    │              │                │
    │              │                ├─ Skill 2 (B層) ──── Code Review Phase B
    │              │                │
    │              │                └─ Skill 2 (C層) ──── Code Review Phase C
    │              │
    │              ├─ CI/CD Pipeline ────► GitHub Actions
    │              │
    │              └─ Skill 3 ────┬───── Experiment Pipeline (5 stages)
    │                             │
    │                             └─ Skill 4 ───┬─── Critical Review (3 experts)
    │                                           │
    │                                           └─ Skill 5 ──── Research Workflow
    │
    ├─ Skill 7 ────┐
    │              ├─ Health Monitor (hourly)
    │              │
    ├─ Skill 8 ────┼─ Security & Compliance (weekly)
    │              │
    ├─ Skill 9 ────┼─ Performance Metrics (daily)
    │              │
    ├─ Skill 10 ───┼─ Dependency Detector (weekly)
    │              │
    ├─ Skill 11 ───┤─ Critical Guardian (PR + daily)
    │              │
    └─►Dashboard ◄─┴─ Real-time metrics aggregation
       (Web UI)
```

### Dependency Table

| From | To | Type | Criticality | Duration Impact |
|------|-----|------|-------------|-----------------|
| Security | All Skills | Blocking | Critical | 1 week delay |
| Skill 1 | Skill 2A | Hard (test case) | High | 0 days |
| Skill 1 | CI/CD | Required | Critical | 1 day delay |
| Skill 2A | Skill 2B | Soft | Medium | 0 days |
| Skill 2B | Skill 2C | Soft | Medium | 0 days |
| Skill 3 | Skill 4 | Hard (input) | High | 1 day delay |
| Skill 4 | Skill 5 | Soft | Low | 0 days |
| Skill 1-10 | Dashboard | Soft | Low | 2 days delay |
| Skill 7-11 | Dashboard | Soft | Low | 2 days delay |
| Dashboard | Go-Live | Hard | Critical |  final 日delay |

### Critical Path Analysis

```
最短必要期間: 12 週間

Critical Path:
1. Security Infrastructure      Week 1 (必須)
2. Skill 1 + Skill 2A           Week 1-2（Skill 3 の入力必要）
3. Skill 3 → Skill 4 → Skill 5   Week 3-6（チェーン依存）
4. Dashboard 実装                Week 10（全 Skill 待ち）
5. 統合テスト + Go-Live          Week 11-12

Non-Critical:
- Skill 7-11: Week 7-9 並行実装可（Skill 1-5 完了後）
- Skill 9-10: 完全に独立
```

---

## Summary & Recommendations

### Best Practices

✅ **実装順序**:
1. Security 基盤 → 全体の信頼性向上
2. Skill 1-2 → CI/CD 統合の基礎
3. Skill 3-5 → コア実験パイプライン
4. Skill 7-11 → 監視・ガーディアン機能
5. Dashboard → 最終統合

✅ **並列戦略**:
- Wave 1 (Week 1-2): 4 teams 並列
- Wave 2 (Week 3-6): 4 tracks（チェーン制約）
- Wave 3 (Week 7-9): 6 forks（ほぼ independent）
- Wave 4 (Week 10-12): 3 squads（統合テスト）

✅ **リスク軽減**:
- Skill 1-2 で CI/CD 早期有効化
- Checkpoint で失敗時の再開を容易化
- Monitor + Guardian で本番環境を保護

✅ **検証戦略**:
- Unit tests: Phase 1-2（基礎）
- Integration tests: Phase 3-6（パイプライン）
- E2E tests: Phase 10-12（本番準備）
- Load tests: Phase 10（ダッシュボード）

---

## References

関連ドキュメント:
- [EXTENDED_AUTOMATION_ROADMAP.md](./EXTENDED_AUTOMATION_ROADMAP.md)
- [CRITICAL_GUARDIAN_DESIGN.md](./CRITICAL_GUARDIAN_DESIGN.md)
- [SKILL_IMPLEMENTATION_GUIDE.md](./SKILL_IMPLEMENTATION_GUIDE.md)
- [coding-conventions-project.md](./coding-conventions-project.md)
