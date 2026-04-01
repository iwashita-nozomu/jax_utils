#!/usr/bin/env python3
"""
Comprehensive Review Skill - Main Execution Script

ドキュメント・Skill・ツール全体の包括的なレビューを実施。
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import importlib.util

# スクリプットディレクトリ
SCRIPT_DIR = Path(__file__).parent
CHECKERS_DIR = SCRIPT_DIR / "checkers"
CONFIG_DIR = SCRIPT_DIR / "config"
WORKSPACE_ROOT = SCRIPT_DIR.parent.parent.parent  # /workspace

# チェッカーモジュール一覧
PHASES = {
    1: ("doc_checker.py", "Documentation Review"),
    2: ("skills_checker.py", "Skills Coherence"),
    3: ("tools_checker.py", "Tools & Scripts"),
    4: ("integration_checker.py", "Integration Test"),
    5: ("report_generator.py", "Report Generation"),
}


def load_checker(phase: int):
    """チェッカースクリプトを動的に読み込み"""
    module_name, _ = PHASES.get(phase, (None, None))
    if not module_name:
        raise ValueError(f"Unknown phase: {phase}")
    
    module_path = CHECKERS_DIR / module_name
    if not module_path.exists():
        raise FileNotFoundError(f"Checker not found: {module_path}")
    
    spec = importlib.util.spec_from_file_location(
        f"checker_{phase}", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_phase(phase: int, verbose: bool = False, **kwargs):
    """フェーズを実行"""
    if phase not in PHASES:
        raise ValueError(f"Invalid phase: {phase}")
    
    _, phase_name = PHASES[phase]
    
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Phase {phase}: {phase_name}")
        print(f"{'=' * 60}")
    
    try:
        checker = load_checker(phase)
        result = checker.run(workspace_root=WORKSPACE_ROOT, verbose=verbose, **kwargs)
        return {
            "phase": phase,
            "name": phase_name,
            "status": result.get("status", "unknown"),
            "issues": result.get("issues", []),
            "details": result.get("details", {}),
        }
    except Exception as e:
        if verbose:
            print(f"❌ Phase {phase} failed: {e}")
        return {
            "phase": phase,
            "name": phase_name,
            "status": "error",
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Review Skill - Review docs/skills/tools holistically"
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="1,2,3,4,5",
        help="Phase numbers to run (comma-separated, default: all)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--output",
        choices=["markdown", "json", "text"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save report to file",
    )
    parser.add_argument(
        "--generate-roadmap",
        action="store_true",
        help="Generate implementation roadmap",
    )
    
    args = parser.parse_args()
    
    # フェーズパース
    try:
        phases = [int(p.strip()) for p in args.phases.split(",")]
    except ValueError:
        print("❌ Invalid phases format. Use comma-separated numbers (e.g., 1,2,3)")
        sys.exit(1)
    
    # レビュー実行
    results = []
    for phase in phases:
        result = run_phase(phase, verbose=args.verbose)
        results.append(result)
    
    # 結果出力
    if args.output == "json":
        output = {
            "timestamp": datetime.now().isoformat(),
            "workspace": str(WORKSPACE_ROOT),
            "phases": results,
            "summary": {
                "total_phases": len(results),
                "passed": sum(1 for r in results if r.get("status") == "pass"),
                "warnings": sum(1 for r in results if r.get("status") == "warn"),
                "errors": sum(1 for r in results if r.get("status") == "error"),
            }
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        # Text/Markdown 形式
        print("\n" + "=" * 70)
        print("COMPREHENSIVE REVIEW REPORT")
        print("=" * 70)
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Workspace: {WORKSPACE_ROOT}")
        print()
        
        for result in results:
            phase = result["phase"]
            name = result["name"]
            status = result.get("status", "unknown").upper()
            
            status_mark = {
                "PASS": "✅",
                "WARN": "⚠️",
                "ERROR": "❌",
            }.get(status, "❓")
            
            print(f"\n{status_mark} Phase {phase}: {name} [{status}]")
            
            if result.get("error"):
                print(f"   Error: {result['error']}")
            elif result.get("issues"):
                for issue in result["issues"][:5]:  # 最初の5件のみ表示
                    print(f"   - {issue}")
    
    # レポート保存
    if args.save_report:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_path = WORKSPACE_ROOT / "reports" / f"comprehensive-review-{timestamp}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Comprehensive Review Report\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n\n")
            
            for result in results:
                phase = result["phase"]
                name = result["name"]
                status = result.get("status", "unknown").upper()
                
                f.write(f"## Phase {phase}: {name}\n\n")
                f.write(f"**Status**: {status}\n\n")
                
                if result.get("issues"):
                    f.write("**Issues**:\n\n")
                    for issue in result["issues"]:
                        f.write(f"- {issue}\n")
                    f.write("\n")
        
        print(f"\n📄 Report saved: {report_path}")
    
    # ロードマップ生成
    if args.generate_roadmap:
        print("\n📋 Generating roadmap...")
        # TODO: ロードマップ生成ロジック
        print("(Feature not yet implemented)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
