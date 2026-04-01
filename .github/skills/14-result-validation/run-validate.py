#!/usr/bin/env python3
"""
Skill 14: Result Validation & Critical Review

実験結果の批判的検証。
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np

WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent


def load_results(result_file):
    """結果ファイルを読み込み"""
    print(f"📂 Loading results from {result_file}...")
    
    with open(result_file) as f:
        return json.load(f)


def validate_statistics(results):
    """統計検証"""
    print("📊 Performing statistical validation...")
    
    issues = []
    
    # サンプルサイズ確認
    if isinstance(results, dict):
        for key, values in results.items():
            if isinstance(values, list) and len(values) < 5:
                issues.append({
                    "type": "low_sample_size",
                    "key": key,
                    "n": len(values),
                    "recommendation": "Increase number of runs",
                })
    
    return issues


def check_reproducibility(results):
    """再現可能性確認"""
    print("🔄 Checking reproducibility...")
    
    issues = []
    
    if "seed" not in results:
        issues.append({
            "type": "missing_seed",
            "recommendation": "Set random seed for reproducibility",
        })
    
    if "dependencies" not in results:
        issues.append({
            "type": "missing_dependencies",
            "recommendation": "Record dependency versions",
        })
    
    return issues


def validate_paper_standard(results):
    """論文基準検証"""
    print("📝 Validating against paper standards...")
    
    issues = []
    
    required_fields = ["abstract", "method", "results", "discussion"]
    
    for field in required_fields:
        if field not in results:
            issues.append({
                "type": "missing_section",
                "field": field,
            })
    
    return issues


def generate_report(stat_issues, repro_issues, paper_issues):
    """レポート生成"""
    print("\n📊 Generating report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "issues": {
            "statistical": stat_issues,
            "reproducibility": repro_issues,
            "paper_standard": paper_issues,
        },
        "summary": {
            "total_issues": len(stat_issues) + len(repro_issues) + len(paper_issues),
            "critical": len([x for x in stat_issues if x.get("type") == "low_sample_size"]),
        }
    }
    
    return report


def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description="Validate experiment results")
    parser.add_argument("--results", required=True, help="Results JSON file")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Skill 14: Result Validation & Critical Review")
    print("=" * 60)
    
    result_file = Path(args.results)
    if not result_file.exists():
        print(f"❌ Results file not found: {result_file}")
        return 1
    
    results = load_results(result_file)
    
    stat_issues = validate_statistics(results)
    repro_issues = check_reproducibility(results)
    paper_issues = validate_paper_standard(results)
    
    report = generate_report(stat_issues, repro_issues, paper_issues)
    
    # レポート保存
    report_dir = WORKSPACE_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    report_file = report_dir / f"result-validate-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Validation complete")
    print(f"   Total issues: {report['summary']['total_issues']}")
    print(f"   Critical issues: {report['summary']['critical']}")
    print(f"\n📄 Report saved: {report_file}")
    
    return 1 if report["summary"]["critical"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
