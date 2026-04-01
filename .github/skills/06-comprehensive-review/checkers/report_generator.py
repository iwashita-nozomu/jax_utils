"""
Phase 5: Report Generator

包括レビューレポートを生成：
- Markdown フォーマット
- 優先度付き Issue リスト
- 修正ロードマップ
- 実装チェックリスト
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def generate_summary_table(phases_results: List[dict]) -> str:
    """フェーズ別サマリーテーブル生成"""
    table = "| Phase | Name | Status | Issues |\n"
    table += "|-------|------|--------|--------|\n"
    
    for result in phases_results:
        phase = result.get("phase", "?")
        name = result.get("name", "Unknown")
        status = result.get("status", "unknown").upper()
        issues_count = len(result.get("issues", []))
        
        status_mark = {
            "PASS": "✅",
            "WARN": "⚠️",
            "ERROR": "❌",
        }.get(status, "❓")
        
        table += f"| {phase} | {name} | {status_mark} {status} | {issues_count} |\n"
    
    return table


def generate_issues_by_severity(all_results: List[dict]) -> str:
    """重大度別 Issue リスト生成"""
    issues_by_severity = {"error": [], "warn": [], "info": []}
    
    # Phase ごとに Issue 集約
    phase_names = {
        1: "Documentation",
        2: "Skills",
        3: "Tools",
        4: "Integration",
        5: "Report",
    }
    
    for result in all_results:
        phase = result.get("phase", 0)
        phase_name = phase_names.get(phase, "Unknown")
        
        for issue in result.get("issues", []):
            # Issue パース (format: "SEVERITY: message")
            if ":" in issue:
                severity_str, message = issue.split(":", 1)
                severity = severity_str.strip().lower()
                if severity in issues_by_severity:
                    issues_by_severity[severity].append({
                        "phase": phase,
                        "phase_name": phase_name,
                        "message": message.strip(),
                    })
    
    output = "## Issues by Severity\n\n"
    
    # Errors
    if issues_by_severity["error"]:
        output += "### 🔴 Errors (" + str(len(issues_by_severity["error"])) + ")\n\n"
        for issue in issues_by_severity["error"][:10]:
            output += f"- **[Phase {issue['phase']}: {issue['phase_name']}]** {issue['message']}\n"
        output += "\n"
    
    # Warnings
    if issues_by_severity["warn"]:
        output += "### 🟡 Warnings (" + str(len(issues_by_severity["warn"])) + ")\n\n"
        for issue in issues_by_severity["warn"][:10]:
            output += f"- **[Phase {issue['phase']}: {issue['phase_name']}]** {issue['message']}\n"
        output += "\n"
    
    # Info
    if issues_by_severity["info"]:
        output += "### 🔵 Info (" + str(len(issues_by_severity["info"])) + ")\n\n"
        for issue in issues_by_severity["info"][:5]:
            output += f"- **[Phase {issue['phase']}: {issue['phase_name']}]** {issue['message']}\n"
        output += "\n"
    
    return output


def generate_implementation_roadmap(all_results: List[dict]) -> str:
    """実装ロードマップ生成"""
    roadmap = "## Implementation Roadmap\n\n"
    
    roadmap += """### Week 1: Documentation Fixes
- [ ] Fix broken links (Phase 1)
- [ ] Standardize terminology (Phase 1)
- [ ] Resolve circular references (Phase 1)
- [ ] Verify metadata completeness (Phase 1)

### Week 2: Skills Coherence
- [ ] Fix circular Skill dependencies (Phase 2)
- [ ] Remove duplicate functionality (Phase 2)
- [ ] Implement missing Skill components (Phase 2)
- [ ] Validate Skill sequencing (Phase 2)

### Week 3: Tools & Testing
- [ ] Implement empty scripts (Phase 3)
- [ ] Add missing unit tests (Phase 3)
- [ ] Document all tools (Phase 3)
- [ ] Add external dependency specs (Phase 3)

### Week 4: Integration & Deployment
- [ ] Update CLI instructions (Phase 4)
- [ ] Configure GitHub Actions (Phase 4)
- [ ] Update Docker setup (Phase 4)
- [ ] Verify complete workflow (Phase 4)

"""
    
    return roadmap


def generate_health_metrics(all_results: List[dict]) -> str:
    """健全性指標生成"""
    metrics = "## Health Metrics\n\n"
    
    total_issues = 0
    error_count = 0
    warn_count = 0
    
    for result in all_results:
        for issue in result.get("issues", []):
            total_issues += 1
            if issue.startswith("ERROR"):
                error_count += 1
            elif issue.startswith("WARN"):
                warn_count += 1
    
    # 健全性スコア計算（0-100）
    health_score = max(0, 100 - (error_count * 10 + warn_count * 3))
    
    health_status = "🔴 Critical" if health_score < 30 else "🟡 Warning" if health_score < 70 else "🟢 Healthy"
    
    metrics += f"**Overall Health Score**: {health_score}/100 ({health_status})\n\n"
    metrics += f"- Total Issues: {total_issues}\n"
    metrics += f"- Errors: {error_count}\n"
    metrics += f"- Warnings: {warn_count}\n"
    
    metrics += "\n### Component Status\n\n"
    
    for result in all_results:
        phase = result.get("phase", 0)
        name = result.get("name", "Unknown")
        status = result.get("status", "unknown").upper()
        issue_count = len(result.get("issues", []))
        
        status_icon = {
            "PASS": "✅",
            "WARN": "⚠️",
            "ERROR": "❌",
        }.get(status, "❓")
        
        metrics += f"- {status_icon} **Phase {phase}: {name}** — {issue_count} issues\n"
    
    metrics += "\n"
    
    return metrics


def generate_next_steps(all_results: List[dict]) -> str:
    """次のステップ推奨生成"""
    steps = "## Recommended Next Steps\n\n"
    
    # Errors がある場合
    error_phases = [r for r in all_results if r.get("status") == "error"]
    if error_phases:
        steps += "### Priority 1: Fix Critical Errors\n\n"
        for result in error_phases:
            phase = result.get("phase", 0)
            name = result.get("name", "Unknown")
            steps += f"1. Phase {phase}: {name}\n"
        steps += "\n"
    
    # Warnings がある場合
    warn_phases = [r for r in all_results if r.get("status") == "warn"]
    if warn_phases:
        steps += "### Priority 2: Address Warnings\n\n"
        for result in warn_phases:
            phase = result.get("phase", 0)
            name = result.get("name", "Unknown")
            steps += f"1. Phase {phase}: {name}\n"
        steps += "\n"
    
    steps += "### Priority 3: Continuous Improvement\n\n"
    steps += "- Set up automated continuous review runs\n"
    steps += "- Integrate comprehensive review into CI/CD pipeline\n"
    steps += "- Schedule weekly review cycles\n"
    steps += "- Monitor health metrics over time\n\n"
    
    return steps


def generate_checklist(all_results: List[dict]) -> str:
    """実装チェックリスト生成"""
    checklist = "## Implementation Checklist\n\n"
    
    checklist += """### Documentation
- [ ] All markdown files are valid
- [ ] No broken internal links
- [ ] Terminology is consistent
- [ ] No circular references
- [ ] Metadata is complete

### Skills
- [ ] No circular dependencies
- [ ] No duplicate functionality
- [ ] All required files present
- [ ] Skill sequencing is valid
- [ ] All Skills documented

### Tools & Scripts
- [ ] All scripts implemented
- [ ] Test coverage > 70%
- [ ] All tools documented
- [ ] Dependencies listed
- [ ] No empty scripts

### Integration
- [ ] CLI instructions updated
- [ ] GitHub Actions configured
- [ ] Docker setup complete
- [ ] Workflow compatibility verified
- [ ] Environment fully configured

### Final Review
- [ ] End-to-end test passed
- [ ] All Issues resolved
- [ ] Health score > 90
- [ ] Documentation updated
- [ ] Release ready

"""
    
    return checklist


def run(workspace_root: Path = None, verbose: bool = False, phases_results: list = None, **kwargs) -> dict:
    """Phase 5 を実行"""
    if workspace_root is None:
        workspace_root = Path.cwd()
    
    workspace_root = Path(workspace_root)
    
    if verbose:
        print("📄 Generating comprehensive review report...")
    
    # テスト用ダミーデータ（本来は前のフェーズから受け取る）
    if phases_results is None:
        phases_results = []
    
    # レポート生成
    report = "# Comprehensive Review Report\n\n"
    report += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"**Workspace**: {workspace_root}\n\n"
    
    # サマリーテーブル
    report += "## Summary\n\n"
    report += generate_summary_table(phases_results)
    report += "\n"
    
    # 健全性指標
    report += generate_health_metrics(phases_results)
    
    # Issue リスト
    report += generate_issues_by_severity(phases_results)
    
    # チェックリスト
    report += generate_checklist(phases_results)
    
    # ロードマップ
    report += generate_implementation_roadmap(phases_results)
    
    # 次のステップ
    report += generate_next_steps(phases_results)
    
    # 報告書保存
    report_dir = workspace_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_file = report_dir / f"comprehensive-review-{timestamp}.md"
    
    try:
        report_file.write_text(report, encoding="utf-8")
        if verbose:
            print(f"   Report saved: {report_file}")
    except Exception as e:
        if verbose:
            print(f"   ❌ Error saving report: {e}")
    
    if verbose:
        print("\n   📊 Report Summary:")
        print(f"   - Total sections: 6")
        print(f"   - Report file: {report_file.name}")
    
    return {
        "status": "pass",
        "issues": [],
        "details": {
            "report_file": str(report_file),
            "report_size": len(report),
            "sections": 6,
        }
    }


if __name__ == "__main__":
    import sys
    
    workspace = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    result = run(workspace, verbose=True)
    
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Report generated: {result['details'].get('report_file', 'unknown')}")
