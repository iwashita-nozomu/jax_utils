#!/usr/bin/env python3
"""
Skill 16: CI/CD Pipeline Integration

GitHub Actions ワークフロー・Makefile・スクリプトの統合管理。
"""

import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
import subprocess

WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent


def validate_workflow_yaml():
    """GitHub Actions ワークフロー定義検証"""
    print("🔍 Validating GitHub Actions workflows...")
    
    issues = []
    workflow_dir = WORKSPACE_ROOT / ".github" / "workflows"
    
    if workflow_dir.exists():
        for yaml_file in workflow_dir.glob("*.yml"):
            try:
                with open(yaml_file) as f:
                    workflow = yaml.safe_load(f)
                
                # 基本的なバリデーション
                if "name" not in workflow:
                    issues.append({
                        "file": yaml_file.name,
                        "type": "missing_name",
                    })
                
                if "on" not in workflow:
                    issues.append({
                        "file": yaml_file.name,
                        "type": "missing_trigger",
                    })
                
                if "jobs" not in workflow or not workflow["jobs"]:
                    issues.append({
                        "file": yaml_file.name,
                        "type": "missing_jobs",
                    })
            except Exception as e:
                issues.append({
                    "file": yaml_file.name,
                    "type": "parse_error",
                    "error": str(e),
                })
    
    print(f"   Found {len(issues)} validation issues")
    return issues


def check_makefile():
    """Makefile の確認"""
    print("📄 Checking Makefile...")
    
    issues = []
    makefile = WORKSPACE_ROOT / "Makefile"
    
    required_targets = ["test", "build", "check"]
    
    if makefile.exists():
        content = makefile.read_text()
        
        for target in required_targets:
            if f".PHONY: {target}" not in content and f"{target}:" not in content:
                issues.append({
                    "type": "missing_target",
                    "target": target,
                })
    else:
        issues.append({
            "type": "makefile_not_found",
        })
    
    print(f"   Found {len(issues)} issues")
    return issues


def validate_dependencies():
    """ワークフロー依存関係検証"""
    print("🔗 Validating dependencies...")
    
    issues = []
    
    # GitHub Actions ワークフロー定義から依存関係を抽出
    workflow_dir = WORKSPACE_ROOT / ".github" / "workflows"
    
    if workflow_dir.exists():
        for yaml_file in workflow_dir.glob("*.yml"):
            try:
                with open(yaml_file) as f:
                    workflow = yaml.safe_load(f)
                
                for job_name, job_def in workflow.get("jobs", {}).items():
                    if "needs" in job_def:
                        needs = job_def["needs"]
                        if isinstance(needs, str):
                            needs = [needs]
                        
                        for need in needs:
                            if need not in workflow.get("jobs", {}):
                                issues.append({
                                    "type": "missing_job",
                                    "job": job_name,
                                    "needs": need,
                                })
            except Exception as e:
                pass
    
    print(f"   Found {len(issues)} dependency issues")
    return issues


def generate_report(workflow_issues, makefile_issues, dep_issues):
    """レポート生成"""
    print("\n📊 Generating report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "issues": {
            "workflows": workflow_issues,
            "makefile": makefile_issues,
            "dependencies": dep_issues,
        },
        "summary": {
            "total_issues": len(workflow_issues) + len(makefile_issues) + len(dep_issues),
            "workflow_issues": len(workflow_issues),
            "makefile_issues": len(makefile_issues),
            "dependency_issues": len(dep_issues),
        }
    }
    
    return report


def main():
    """メイン実行"""
    print("=" * 60)
    print("Skill 16: CI/CD Pipeline Integration")
    print("=" * 60)
    
    workflow_issues = validate_workflow_yaml()
    makefile_issues = check_makefile()
    dep_issues = validate_dependencies()
    
    report = generate_report(workflow_issues, makefile_issues, dep_issues)
    
    # レポート保存
    report_dir = WORKSPACE_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    report_file = report_dir / f"ci-cd-validate-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ CI/CD validation complete")
    print(f"   Total issues: {report['summary']['total_issues']}")
    print(f"\n📄 Report saved: {report_file}")
    
    return 1 if report["summary"]["total_issues"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
