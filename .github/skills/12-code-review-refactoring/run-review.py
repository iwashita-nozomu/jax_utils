#!/usr/bin/env python3
"""
Skill 12: Code Review & Refactoring

AI 支援コードレビューと自動リファクタリング。
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime
import difflib

WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent


def get_changed_files():
    """変更ファイルを取得"""
    print("📁 Detecting changed files...")
    
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD~1"],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
        )
        files = result.stdout.strip().split("\n") if result.stdout.strip() else []
        print(f"   Found {len(files)} changed files")
        return files
    except Exception as e:
        print(f"⚠️  git diff failed: {e}")
        return []


def run_multi_layer_review(changed_files):
    """多層レビュー実行"""
    print("🔍 Running multi-layer review...")
    
    issues = []
    
    for file_path in changed_files:
        if not file_path.endswith(".py"):
            continue
        
        full_path = WORKSPACE_ROOT / file_path
        if not full_path.exists():
            continue
        
        content = full_path.read_text(encoding="utf-8")
        
        # Linter チェック（簡易版）
        if "import *" in content:
            issues.append({
                "file": file_path,
                "layer": "linter",
                "issue": "wildcard_import",
                "severity": "warning",
            })
        
        # アーキテクチャチェック
        if "global " in content:
            issues.append({
                "file": file_path,
                "layer": "architecture",
                "issue": "global_variable",
                "severity": "warning",
            })
        
        # 型アノテーションチェック
        if "def " in content and ") ->" not in content:
            issues.append({
                "file": file_path,
                "layer": "quality",
                "issue": "missing_type_annotation",
                "severity": "info",
            })
    
    print(f"   Found {len(issues)} issues")
    return issues


def generate_refactoring_proposals(issues):
    """リファクタリング提案生成"""
    print("💡 Generating refactoring proposals...")
    
    proposals = []
    
    for issue in issues:
        if issue["issue"] == "wildcard_import":
            proposals.append({
                "issue_id": issue,
                "proposal": "Replace wildcard import with explicit imports",
                "priority": "high",
            })
        elif issue["issue"] == "global_variable":
            proposals.append({
                "issue_id": issue,
                "proposal": "Refactor to use class attribute or function parameter",
                "priority": "medium",
            })
    
    return proposals


def generate_report(issues, proposals):
    """レポート生成"""
    print("\n📊 Generating report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "issues": issues,
        "proposals": proposals,
        "summary": {
            "total_issues": len(issues),
            "high_priority": len([x for x in proposals if x.get("priority") == "high"]),
            "medium_priority": len([x for x in proposals if x.get("priority") == "medium"]),
        }
    }
    
    return report


def main():
    """メイン実行"""
    print("=" * 60)
    print("Skill 12: Code Review & Refactoring")
    print("=" * 60)
    
    changed_files = get_changed_files()
    issues = run_multi_layer_review(changed_files)
    proposals = generate_refactoring_proposals(issues)
    report = generate_report(issues, proposals)
    
    # レポート保存
    report_dir = WORKSPACE_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    report_file = report_dir / f"code-review-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Report saved: {report_file}")
    print(f"✅ Code review complete: {report['summary']['total_issues']} issues found")
    
    return 1 if report["summary"]["high_priority"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
