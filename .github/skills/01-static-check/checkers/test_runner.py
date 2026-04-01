"""
Skill 1: Static Check — Test Runner (Pytest)

pytest によるユニット・統合テスト実行
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class TestResult:
    """テスト実行結果"""
    total: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_ms: float
    failures: List[Dict[str, Any]]
    coverage: Optional[float] = None


class TestRunner:
    """Pytest テストランナー"""
    
    def __init__(self, test_dir: Path = Path("python/tests")):
        """
        Args:
            test_dir: テストディレクトリ
        """
        self.test_dir = test_dir
    
    def run(
        self,
        with_coverage: bool = True,
        verbose: bool = False,
    ) -> TestResult:
        """テスト実行
        
        Args:
            with_coverage: カバレッジ測定を含むか
            verbose: 詳細出力するか
        
        Returns:
            テスト結果
        """
        import time
        start = time.time()
        
        cmd = [
            "pytest",
            str(self.test_dir),
            "--json-report",
            "--json-report-file=/tmp/pytest-report.json",
        ]
        
        if with_coverage:
            cmd.extend([
                "--cov=python",
                "--cov-report=json",
            ])
        
        if verbose:
            cmd.append("-v")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5分
            )
            
            # JSON レポート解析
            failures = []
            total = 0
            passed = 0
            failed = 0
            skipped = 0
            errors = 0
            coverage = None
            
            try:
                import json as json_module
                
                # pytest-json-report
                report_file = Path("/tmp/pytest-report.json")
                if report_file.exists():
                    with open(report_file, "r") as f:
                        report = json_module.load(f)
                        total = report.get("summary", {}).get("total", 0)
                        passed = report.get("summary", {}).get("passed", 0)
                        failed = report.get("summary", {}).get("failed", 0)
                        skipped = report.get("summary", {}).get("skipped", 0)
                        errors = report.get("summary", {}).get("error", 0)
                
                # Coverage
                coverage_file = Path(".coverage.json")
                if coverage_file.exists() and with_coverage:
                    with open(coverage_file, "r") as f:
                        cov_data = json_module.load(f)
                        coverage = cov_data.get("totals", {}).get("percent_covered", 0)
            
            except Exception as e:
                print(f"Warning: Failed to parse test results: {e}")
            
            duration_ms = (time.time() - start) * 1000
            
            return TestResult(
                total=total,
                passed=passed,
                failed=failed,
                skipped=skipped,
                errors=errors,
                duration_ms=duration_ms,
                failures=failures,
                coverage=coverage,
            )
        
        except subprocess.TimeoutExpired:
            return TestResult(
                total=0,
                passed=0,
                failed=1,
                skipped=0,
                errors=1,
                duration_ms=(time.time() - start) * 1000,
                failures=[{"message": "Tests timed out"}],
            )
        except Exception as e:
            return TestResult(
                total=0,
                passed=0,
                failed=1,
                skipped=0,
                errors=1,
                duration_ms=(time.time() - start) * 1000,
                failures=[{"message": str(e)}],
            )
    
    def run_specific(
        self,
        test_pattern: str,
        verbose: bool = False,
    ) -> TestResult:
        """特定のテストを実行
        
        Args:
            test_pattern: テストパターン (e.g. "test_week1*")
            verbose: 詳細出力するか
        
        Returns:
            テスト結果
        """
        import time
        start = time.time()
        
        cmd = ["pytest", str(self.test_dir / test_pattern)]
        if verbose:
            cmd.append("-v")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            success = result.returncode == 0
            duration_ms = (time.time() - start) * 1000
            
            return TestResult(
                total=1,
                passed=1 if success else 0,
                failed=0 if success else 1,
                skipped=0,
                errors=0,
                duration_ms=duration_ms,
                failures=[],
            )
        
        except Exception as e:
            return TestResult(
                total=0,
                passed=0,
                failed=1,
                skipped=0,
                errors=1,
                duration_ms=(time.time() - start) * 1000,
                failures=[{"message": str(e)}],
            )


if __name__ == "__main__":
    print("Test Runner")
    
    runner = TestRunner()
    result = runner.run()
    
    print(f"Total: {result.total}")
    print(f"Passed: {result.passed}")
    print(f"Failed: {result.failed}")
    print(f"Duration: {result.duration_ms:.2f}ms")
    if result.coverage:
        print(f"Coverage: {result.coverage:.1f}%")
