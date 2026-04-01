#!/usr/bin/env python3
"""
Skill 11: Test Execution & Coverage

pytest による包括的なテスト実行とカバレッジ測定。
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent


def run_tests():
    """pytest でテスト実行"""
    print("🧪 Running tests with pytest...")
    
    test_dir = WORKSPACE_ROOT / "python" / "tests"
    
    try:
        result = subprocess.run(
            [
                "pytest",
                str(test_dir),
                "-v",
                "--cov=python",
                "--cov-report=json",
                "--cov-report=html",
            ],
            cwd=str(WORKSPACE_ROOT),
            capture_output=True,
            text=True,
        )
        
        print(result.stdout)
        return result.returncode, result.stdout
    except Exception as e:
        print(f"⚠️  pytest failed: {e}")
        return 1, ""


def generate_report(test_output):
    """レポート生成"""
    print("\n📊 Generating report...")
    
    # Coverage JSON レポートを読み込み
    cov_report_file = WORKSPACE_ROOT / ".coverage"
    cov_data = {}
    
    if (WORKSPACE_ROOT / "coverage.json").exists():
        with open(WORKSPACE_ROOT / "coverage.json") as f:
            cov_data = json.load(f)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_output": test_output,
        "coverage": cov_data,
    }
    
    return report


def main():
    """メイン実行"""
    print("=" * 60)
    print("Skill 11: Test Execution & Coverage")
    print("=" * 60)
    
    returncode, test_output = run_tests()
    report = generate_report(test_output)
    
    # レポート保存
    report_dir = WORKSPACE_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    report_file = report_dir / f"test-execution-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Report saved: {report_file}")
    
    return returncode


if __name__ == "__main__":
    sys.exit(main())
