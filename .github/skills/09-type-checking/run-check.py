#!/usr/bin/env python3
"""
Skill 9: Type Checking & Static Analysis

Pyright による厳密な型チェックと複数の静的解析ツール実行。
"""

import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent


def run_pyright():
    """Pyright strict mode で型チェック"""
    print("🔍 Running Pyright strict mode...")
    try:
        result = subprocess.run(
            ["pyright", "--outputjson", str(WORKSPACE_ROOT / "python")],
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout) if result.stdout else {}
    except Exception as e:
        print(f"⚠️  Pyright failed: {e}")
        return {}


def run_ruff():
    """ruff による lint チェック"""
    print("🔍 Running ruff...")
    try:
        result = subprocess.run(
            ["ruff", "check", str(WORKSPACE_ROOT / "python"), "--output-format=json"],
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout) if result.stdout else []
    except Exception as e:
        print(f"⚠️  ruff failed: {e}")
        return []


def run_mypy():
    """mypy による型チェック"""
    print("🔍 Running mypy...")
    try:
        result = subprocess.run(
            ["mypy", str(WORKSPACE_ROOT / "python"), "--json"],
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout) if result.stdout else []
    except Exception as e:
        print(f"⚠️  mypy failed: {e}")
        return []


def generate_report(pyright_results, ruff_results, mypy_results):
    """レポート生成"""
    print("\n📊 Generating report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "tools": {
            "pyright": pyright_results,
            "ruff": ruff_results,
            "mypy": mypy_results,
        },
        "summary": {
            "pyright_errors": len([x for x in pyright_results.get("generalDiagnostics", []) if x.get("severity") == "error"]),
            "ruff_issues": len(ruff_results),
            "mypy_errors": len([x for x in mypy_results if x.get("severity") == "error"]),
        }
    }
    
    print(f"✅ Type checking complete")
    print(f"   Pyright errors: {report['summary']['pyright_errors']}")
    print(f"   Ruff issues: {report['summary']['ruff_issues']}")
    print(f"   Mypy errors: {report['summary']['mypy_errors']}")
    
    return report


def main():
    """メイン実行"""
    print("=" * 60)
    print("Skill 9: Type Checking & Static Analysis")
    print("=" * 60)
    
    pyright_results = run_pyright()
    ruff_results = run_ruff()
    mypy_results = run_mypy()
    
    report = generate_report(pyright_results, ruff_results, mypy_results)
    
    # レポート保存
    report_dir = WORKSPACE_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    report_file = report_dir / f"type-check-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved: {report_file}")
    
    return 0 if report["summary"]["pyright_errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
