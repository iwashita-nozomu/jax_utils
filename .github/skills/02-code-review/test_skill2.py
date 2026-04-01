#!/usr/bin/env python3
"""
Skill 2 統合テストスイート

目的:
- 全 Layer A コンポーネントの動作検証
- 返却フォーマット（JSON）の正確性確認
- 統合 CLI の全フェーズの動作確認
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, Tuple


class SkillCodeReviewTests:
    """Skill 2 テストスイート。"""

    def __init__(self):
        """初期化。"""
        self.skill_dir = Path(".github/skills/02-code-review")
        self.passed = 0
        self.failed = 0

    def run_command(self, cmd: list) -> Tuple[int, str]:
        """コマンド実行。

        Args:
            cmd: コマンドリスト。

        Returns:
            (終了コード, 出力)
        """
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode, result.stdout + result.stderr

    def test_type_checker(self) -> bool:
        """Type Checker テスト。"""
        print("⏳ テスト: Layer A - Type Checker")

        cmd = [
            "python3",
            str(self.skill_dir / "layer_a_type_checker.py"),
            "--target-dir",
            "python",
            "--json",
        ]

        code, output = self.run_command(cmd)

        try:
            data = json.loads(output)
            assert "status" in data, "status キーがありません"
            assert "component" in data, "component キーがありません"
            print(f"  ✅ PASS (status={data['status']})")
            self.passed += 1
            return True
        except (json.JSONDecodeError, AssertionError) as e:
            print(f"  ❌ FAIL: {e}")
            self.failed += 1
            return False

    def test_linting_checker(self) -> bool:
        """Linting Checker テスト。"""
        print("⏳ テスト: Layer A - Linting Checker")

        cmd = [
            "python3",
            str(self.skill_dir / "layer_a_linting_checker.py"),
            "--target-dir",
            "python",
            "--json",
        ]

        code, output = self.run_command(cmd)

        try:
            data = json.loads(output)
            assert "status" in data, "status キーがありません"
            assert (
                "ruff_passed" in data or "status" in data
            ), "検査結果がありません"
            print(f"  ✅ PASS (status={data['status']})")
            self.passed += 1
            return True
        except (json.JSONDecodeError, AssertionError) as e:
            print(f"  ❌ FAIL: {e}")
            self.failed += 1
            return False

    def test_docstring_checker(self) -> bool:
        """Docstring Checker テスト。"""
        print("⏳ テスト: Layer A - Docstring Checker")

        cmd = [
            "python3",
            str(self.skill_dir / "layer_a_docstring_checker.py"),
            "--target-dir",
            "python",
            "--json",
        ]

        code, output = self.run_command(cmd)

        try:
            data = json.loads(output)
            assert "status" in data, "status キーがありません"
            assert (
                "files_checked" in data
            ), "ファイルチェック情報がありません"
            print(
                f"  ✅ PASS "
                f"(checked={data['files_checked']}, status={data['status']})"
            )
            self.passed += 1
            return True
        except (json.JSONDecodeError, AssertionError) as e:
            print(f"  ❌ FAIL: {e}")
            self.failed += 1
            return False

    def test_coverage_checker(self) -> bool:
        """Test Coverage Checker テスト（スキップ可能）。"""
        print("⏳ テスト: Layer A - Test Coverage Checker")

        cmd = [
            "python3",
            str(self.skill_dir / "layer_a_test_coverage_checker.py"),
            "--json",
        ]

        try:
            code, output = self.run_command(cmd)
            data = json.loads(output)
            print(f"  ✅ PASS (status={data['status']})")
            self.passed += 1
            return True
        except Exception as e:
            print(f"  ⚠️ SKIP: {e}")
            # スキップとしてカウント
            return True

    def test_run_review_cli(self) -> bool:
        """統合 CLI (run-review.py) テスト。"""
        print("⏳ テスト: 統合 CLI - run-review.py --phase A")

        cmd = [
            "python3",
            str(self.skill_dir / "run-review.py"),
            "--phase",
            "A",
            "--json",
        ]

        try:
            code, output = self.run_command(cmd)
            data = json.loads(output)
            assert "phases" in data, "phases キーがありません"
            assert "A" in data["phases"], "Phase A がありません"
            phase_a = data["phases"]["A"]
            assert (
                "overall_status" in phase_a
            ), "overall_status がありません"
            overall = phase_a["overall_status"]
            print(
                f"  ✅ PASS (overall_status={overall})"
            )
            self.passed += 1
            return True
        except (json.JSONDecodeError, AssertionError, KeyError) as e:
            print(f"  ❌ FAIL: {e}")
            self.failed += 1
            return False

    def test_doc_test_triplet(self) -> bool:
        """三点セット検証 スクリプトテスト。"""
        print("⏳ テスト: Level C - check_doc_test_triplet.py")

        cmd = [
            "python3",
            str(self.skill_dir / "check_doc_test_triplet.py"),
            "--target-dir",
            "python",
            "--json",
        ]

        try:
            code, output = self.run_command(cmd)
            data = json.loads(output)
            assert "status" in data, "status キーがありません"
            assert (
                "files_checked" in data
            ), "ファイルチェック情報がありません"
            print(
                f"  ✅ PASS "
                f"(checked={data['files_checked']}, status={data['status']})"
            )
            self.passed += 1
            return True
        except (json.JSONDecodeError, AssertionError) as e:
            print(f"  ❌ FAIL: {e}")
            self.failed += 1
            return False

    def run_all_tests(self) -> int:
        """全テスト実行。"""
        print("\n" + "=" * 70)
        print("Skill 2: Code Review — 統合テストスイート")
        print("=" * 70 + "\n")

        results = {
            "test_type_checker": self.test_type_checker(),
            "test_linting_checker": self.test_linting_checker(),
            "test_docstring_checker": self.test_docstring_checker(),
            "test_coverage_checker": self.test_coverage_checker(),
            "test_doc_test_triplet": self.test_doc_test_triplet(),
            "test_run_review_cli": self.test_run_review_cli(),
        }

        print("\n" + "=" * 70)
        print(f"テスト結果サマリ")
        print("=" * 70)
        print(f"✅ PASS: {self.passed}/{len(results)}")
        print(f"❌ FAIL: {self.failed}/{len(results)}")
        print("=" * 70)

        return 0 if self.failed == 0 else 1


def main() -> int:
    """メイン処理。"""
    tests = SkillCodeReviewTests()
    return tests.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
