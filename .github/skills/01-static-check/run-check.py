#!/usr/bin/env python3
"""
Skill 1: Static Check — Unified CLI

複数のスタティックチェッカーを統合実行するメインスクリプト

使用方法:
  python3 run-check.py --verbose
  python3 run-check.py --checks type,test,docker,coverage
  python3 run-check.py --output json
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# パス設定
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "checkers"))

# グローバルインポート
try:
    from type_checker import PyrightChecker, MypyChecker
    from test_runner import TestRunner
    from docker_validator import DockerValidator
    from coverage_analyzer import CoverageAnalyzer
except ImportError as e:
    print(f"Error importing checkers: {e}")
    sys.exit(1)


def import_checkers():
    """チェッカーモジュールをインポート"""
    return {
        "type": (PyrightChecker, MypyChecker),
        "test": TestRunner,
        "docker": DockerValidator,
        "coverage": CoverageAnalyzer,
    }


def run_checks(
    checks: List[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """複数チェックを実行
    
    Args:
        checks: 実行するチェック名リスト
        verbose: 詳細出力するか
    
    Returns:
        チェック結果
    """
    if checks is None:
        checks = ["type", "test", "docker", "coverage"]
    
    checkers = import_checkers()
    results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "environment": {
            "python_version": sys.version.split()[0],
            "platform": sys.platform,
        },
        "checks": {},
        "summary": {
            "total": 0,
            "passed": 0,
            "failed": 0,
        },
    }
    
    for check_name in checks:
        if check_name not in checkers:
            print(f"⚠️  Unknown check: {check_name}")
            continue
        
        print(f"\n▶️  Running: {check_name}")
        print("-" * 60)
        
        try:
            checker_class = checkers[check_name]
            
            # チェッカー実行
            if check_name == "type":
                # Pyright と Mypy の両方を実行
                pyright = PyrightChecker()
                mypy = MypyChecker()
                
                pyright_result = pyright.check()
                mypy_result = mypy.check()
                
                results["checks"]["pyright"] = {
                    "success": pyright_result.success,
                    "errors": pyright_result.error_count,
                    "warnings": pyright_result.warning_count,
                    "duration_ms": pyright_result.duration_ms,
                }
                
                results["checks"]["mypy"] = {
                    "success": mypy_result.success,
                    "errors": mypy_result.error_count,
                    "warnings": mypy_result.warning_count,
                    "duration_ms": mypy_result.duration_ms,
                }
                
                check_passed = pyright_result.success and mypy_result.success
                
                if verbose:
                    print(f"  Pyright: {pyright_result.error_count} errors")
                    print(f"  Mypy: {mypy_result.error_count} errors")
            
            elif check_name == "test":
                runner = TestRunner()
                result = runner.run(with_coverage=False, verbose=verbose)
                
                results["checks"]["test"] = {
                    "total": result.total,
                    "passed": result.passed,
                    "failed": result.failed,
                    "skipped": result.skipped,
                    "duration_ms": result.duration_ms,
                }
                
                check_passed = result.failed == 0
                
                if verbose:
                    print(f"  Total: {result.total}")
                    print(f"  Passed: {result.passed}")
                    print(f"  Failed: {result.failed}")
            
            elif check_name == "docker":
                validator = DockerValidator()
                result = validator.build_image(dry_run=False)
                
                results["checks"]["docker"] = {
                    "success": result.success,
                    "dockerfile_valid": result.dockerfile_valid,
                    "build_successful": result.build_successful,
                    "image_size_mb": result.image_size_mb,
                    "layers": result.layers,
                    "duration_ms": result.duration_ms,
                }
                
                if result.warnings and verbose:
                    for w in result.warnings[:3]:
                        print(f"  Warning: {w}")
                
                if result.errors and verbose:
                    for e in result.errors[:3]:
                        print(f"  Error: {e}")
                
                check_passed = result.success
            
            elif check_name == "coverage":
                analyzer = CoverageAnalyzer()
                result = analyzer.analyze()
                
                results["checks"]["coverage"] = {
                    "overall_percent": result.overall.percent_covered,
                    "covered_lines": result.overall.covered_lines,
                    "total_lines": result.overall.total_lines,
                    "success": result.success,
                    "files_analyzed": len(result.by_file),
                    "low_coverage_files": len(result.low_coverage_files),
                    "duration_ms": result.duration_ms,
                }
                
                if verbose and result.low_coverage_files:
                    print(f"  Low coverage files: {len(result.low_coverage_files)}")
                    for file_path, cov in result.low_coverage_files[:3]:
                        print(f"    {file_path}: {cov:.1f}%")
                
                check_passed = result.success
            
            else:
                check_passed = False
            
            # 結果集計
            results["summary"]["total"] += 1
            if check_passed:
                results["summary"]["passed"] += 1
                print(f"✅ {check_name} PASSED")
            else:
                results["summary"]["failed"] += 1
                print(f"❌ {check_name} FAILED")
        
        except Exception as e:
            print(f"❌ Error running {check_name}: {e}")
            results["checks"][check_name] = {"error": str(e)}
            results["summary"]["failed"] += 1
    
    return results


def main():
    """メイン エントリーポイント"""
    parser = argparse.ArgumentParser(
        description="Skill 1: Static Check CLI",
    )
    parser.add_argument(
        "--checks",
        type=str,
        default="type,test,docker,coverage",
        help="Comma-separated list of checks to run",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json", "html"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save report to file",
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("Skill 1: Static Check")
    print("=" * 60)
    
    # チェック実行
    checks = args.checks.split(",") if args.checks else []
    results = run_checks(checks=checks, verbose=args.verbose)
    
    # 結果出力
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    summary = results["summary"]
    print(f"Total:  {summary['total']}")
    print(f"✅ Passed: {summary['passed']}")
    print(f"❌ Failed: {summary['failed']}")
    
    if args.output == "json":
        print("\n" + json.dumps(results, indent=2, ensure_ascii=False))
    
    if args.save:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            if args.output == "json":
                json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                f.write(json.dumps(results, indent=2, ensure_ascii=False))
        
        print(f"\n💾 Report saved to: {output_path}")
    
    # 終了コード
    if summary["failed"] == 0:
        print("\n✨ All checks passed!")
        return 0
    else:
        print(f"\n⚠️  {summary['failed']} check(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
