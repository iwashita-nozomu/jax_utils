#!/usr/bin/env python3
"""
Week 4-5: Skill 1 統合テストスイート

Skill 1 の全チェッカーが正常に動作することを確認
- Type Checker (pyright/mypy)
- Linter (ruff/black)
- Test Runner (pytest)
- Docker Validator
- Coverage Analyzer
- 統合 CLI (run-check.py)
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Tuple, Dict
import time

# パス設定
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts" / "shared"))
from error_handler import ExecutionResult, ErrorCode


class SkillWeek45Tester:
    """Week 4-5 Skill 1 統合テスト。"""

    def __init__(self):
        """初期化。"""
        self.skill_dir = Path(".github/skills/01-static-check")
        self.test_results = {}

    def run_command(self, cmd: list, timeout: int = 60) -> Tuple[int, str]:
        """コマンド実行。"""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return -1, "TIMEOUT"
        except Exception as e:
            return -1, str(e)

    def test_run_check_cli(self) -> bool:
        """run-check.py CLI テスト。"""
        print("⏳ テスト 1: run-check.py CLI 基本動作")

        code, output = self.run_command([
            "python3",
            str(self.skill_dir / "run-check.py"),
            "--help",
        ])

        if code == 0 and "usage" in output.lower():
            print("  ✅ PASS - CLI ヘルプ表示成功")
            self.test_results["cli_help"] = True
            return True
        else:
            print(f"  ❌ FAIL - CLI ヘルプ表示失敗")
            self.test_results["cli_help"] = False
            return False

    def test_type_checker(self) -> bool:
        """Type Checker テスト。"""
        print("⏳ テスト 2: Type Checker")

        code, output = self.run_command([
            "python3",
            str(self.skill_dir / "run-check.py"),
            "--checks", "type",
            "--verbose",
        ], timeout=120)

        # Type エラーが検出されるのは正常（エラー 0 も問題なし）
        if code in [0, 1]:  # 0=成功 or エラー検出
            print(f"  ✅ PASS - Type Checker 実行完了")
            self.test_results["type_checker"] = True
            return True
        else:
            print(f"  ❌ FAIL - Type Checker 実行失敗 (code={code})")
            self.test_results["type_checker"] = False
            return False

    def test_test_runner(self) -> bool:
        """Test Runner テスト。"""
        print("⏳ テスト 3: Test Runner (pytest)")

        code, output = self.run_command([
            "python3",
            str(self.skill_dir / "run-check.py"),
            "--checks", "test",
            "--verbose",
        ], timeout=120)

        # テスト実行成功（失敗があっても OK）
        if code in [0, 1]:
            print(f"  ✅ PASS - Test Runner 実行完了")
            self.test_results["test_runner"] = True
            return True
        else:
            print(f"  ❌ FAIL - Test Runner 実行失敗 (code={code})")
            self.test_results["test_runner"] = False
            return False

    def test_docker_validator(self) -> bool:
        """Docker Validator テスト。"""
        print("⏳ テスト 4: Docker Validator")

        # Docker ファイルの存在確認
        docker_file = Path("/workspace/docker/Dockerfile")
        if not docker_file.exists():
            print(f"  ⚠️ SKIP - Dockerfile なし")
            self.test_results["docker_validator"] = None
            return True

        code, output = self.run_command([
            "python3",
            str(self.skill_dir / "run-check.py"),
            "--checks", "docker",
            "--verbose",
        ], timeout=60)

        if code in [0, 1]:
            print(f"  ✅ PASS - Docker Validator 実行完了")
            self.test_results["docker_validator"] = True
            return True
        else:
            print(f"  ❌ FAIL - Docker Validator 実行失敗 (code={code})")
            self.test_results["docker_validator"] = False
            return False

    def test_coverage_analyzer(self) -> bool:
        """Coverage Analyzer テスト。"""
        print("⏳ テスト 5: Coverage Analyzer")

        code, output = self.run_command([
            "python3",
            str(self.skill_dir / "run-check.py"),
            "--checks", "coverage",
            "--verbose",
        ], timeout=120)

        # Coverage 実行完了（0 または 1）
        if code in [0, 1]:
            print(f"  ✅ PASS - Coverage Analyzer 実行完了")
            self.test_results["coverage_analyzer"] = True
            return True
        else:
            print(f"  ❌ FAIL - Coverage Analyzer 実行失敗 (code={code})")
            self.test_results["coverage_analyzer"] = False
            return False

    def test_json_output(self) -> bool:
        """JSON 出力テスト。"""
        print("⏳ テスト 6: JSON 出力形式")

        code, output = self.run_command([
            "python3",
            str(self.skill_dir / "run-check.py"),
            "--output", "json",
            "--verbose",
        ], timeout=120)

        # JSON パース可能か確認
        try:
            data = json.loads(output)
            if isinstance(data, dict) and "checks" in data:
                print(f"  ✅ PASS - JSON 出力形式正常")
                self.test_results["json_output"] = True
                return True
        except json.JSONDecodeError:
            pass

        print(f"  ❌ FAIL - JSON 出力形式不正")
        self.test_results["json_output"] = False
        return False

    def run_all_tests(self) -> ExecutionResult:
        """全テスト実行。"""
        print("\n" + "=" * 70)
        print("Week 4-5: Skill 1 (Static Check) 統合テストスイート")
        print("=" * 70 + "\n")

        start_time = time.time()

        results = [
            self.test_run_check_cli(),
            self.test_type_checker(),
            self.test_test_runner(),
            self.test_docker_validator(),
            self.test_coverage_analyzer(),
            self.test_json_output(),
        ]

        execution_time = time.time() - start_time

        # ExecutionResult にまとめる
        result = ExecutionResult(
            success=all(r for r in results if r is not None),
            script_name="skill1_integration_test",
            execution_time=execution_time,
        )

        passed = sum(1 for r, v in self.test_results.items() if v is True)
        failed = sum(1 for r, v in self.test_results.items() if v is False)
        skipped = sum(1 for r, v in self.test_results.items() if v is None)

        result.output = {
            "tests": self.test_results,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "total": len(self.test_results),
        }

        if failed > 0:
            result.add_error(
                code=ErrorCode.TEST_FAILED,
                message=f"{failed} 個のテストが失敗",
                context={"failed_tests": [k for k, v in self.test_results.items() if v is False]},
            )

        print("\n" + "=" * 70)
        print(f"テスト結果: PASS={passed}, FAIL={failed}, SKIP={skipped}")
        print("=" * 70 + "\n")

        return result


def main():
    """メイン処理。"""
    tester = SkillWeek45Tester()
    result = tester.run_all_tests()

    if "--json" in sys.argv:
        print(result.to_json())
    else:
        print(result.to_markdown())

    result.exit_with_status()


if __name__ == "__main__":
    main()
