#!/usr/bin/env python3
"""
Layer A: Test Coverage Check（テストカバレッジ チェッカー）

目的:
- pytest によるテスト実行
- coverage によるカバレッジ測定
- 最小カバレッジ基準（70%）の確認

レイヤー:
- A層: 基礎検証（開発者向け、常に実行）

出力:
- PASS: カバレッジ >= 70% かつ全テスト成功
- WARN: カバレッジ 50-70% または テスト一部失敗
- FAIL: カバレッジ < 50% または テスト全体失敗
"""

import subprocess
import json
import sys
import re
from pathlib import Path
from typing import Dict, Tuple, Optional


class TestCoverageChecker:
    """テストカバレッジ チェッカー。"""

    def __init__(self, verbose: bool = False):
        """初期化。

        Args:
            verbose: 詳細出力フラグ。
        """
        self.verbose = verbose
        self.min_coverage = 70

    def run_pytest(
        self, test_dir: str = "python/tests"
    ) -> Tuple[bool, str, int]:
        """Pytest でテスト実行。

        Args:
            test_dir: テストディレクトリ。

        Returns:
            (成功フラグ, 出力テキスト, テスト数)
        """
        cmd = [
            "pytest",
            test_dir,
            "-v",
            "--tb=short",
            "--no-header",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            if self.verbose:
                print(f"[Pytest] stdout:\n{result.stdout[-500:]}")

            # テスト数を抽出
            match = re.search(
                r"(\d+) passed|(\d+) failed",
                result.stdout,
            )
            test_count = 0
            if match:
                test_count = int(match.group(1) or match.group(2) or 0)

            passed = result.returncode == 0

            if passed:
                # 成功した場合、テスト数を抽出
                summary = re.search(
                    r"(\d+) passed",
                    result.stdout,
                )
                if summary:
                    test_count = int(summary.group(1))
                    return (
                        True,
                        f"すべてのテストに成功 ({test_count} テスト)",
                        test_count,
                    )
                else:
                    return True, "テスト成功", test_count
            else:
                # 失敗した場合
                failed_match = re.search(
                    r"(\d+) failed",
                    result.stdout,
                )
                passed_match = re.search(
                    r"(\d+) passed",
                    result.stdout,
                )
                failed = int(failed_match.group(1) or 0)
                passed_count = int(passed_match.group(1) or 0)
                return (
                    False,
                    f"{failed} テスト失敗, "
                    f"{passed_count} テスト成功",
                    test_count,
                )

        except subprocess.TimeoutExpired:
            return False, "テスト実行タイムアウト（120秒）", 0
        except FileNotFoundError:
            return None, "pytest: インストールされていません", 0

    def run_coverage(
        self, test_dir: str = "python/tests"
    ) -> Tuple[bool, str, Optional[float]]:
        """Coverage でカバレッジ測定。

        Args:
            test_dir: テストディレクトリ。

        Returns:
            (成功フラグ, 出力テキスト, カバレッジ%)
        """
        cmd = [
            "coverage",
            "run",
            "-m",
            "pytest",
            test_dir,
            "-q",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )

            # カバレッジレポート取得
            report_cmd = ["coverage", "report", "--skip-covered"]

            report_result = subprocess.run(
                report_cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if self.verbose:
                print(f"[Coverage] report:\n{report_result.stdout}")

            # カバレッジ % を抽出
            match = re.search(
                r"TOTAL.*?(\d+)%",
                report_result.stdout,
            )
            if match:
                coverage_pct = float(match.group(1))
                passed = coverage_pct >= self.min_coverage

                status = (
                    f"カバレッジ: {coverage_pct}%"
                    if passed
                    else f"カバレッジ不足: {coverage_pct}% "
                    f"(最小: {self.min_coverage}%)"
                )

                return passed, status, coverage_pct
            else:
                return (
                    False,
                    "カバレッジの解析に失敗",
                    None,
                )

        except subprocess.TimeoutExpired:
            return False, "カバレッジ測定タイムアウト", None
        except FileNotFoundError:
            return None, "coverage: インストールされていません", None

    def check(
        self, test_dir: str = "python/tests"
    ) -> Dict:
        """テストカバレッジチェック実行。

        Args:
            test_dir: テストディレクトリ。

        Returns:
            チェック結果辞書。
        """
        results = {
            "component": "layer_a_test_coverage_checker",
            "pytest_passed": None,
            "pytest_message": "",
            "pytest_count": 0,
            "coverage_passed": None,
            "coverage_message": "",
            "coverage_pct": None,
        }

        # テスト実行
        test_passed, test_msg, test_count = self.run_pytest(test_dir)
        results["pytest_passed"] = test_passed
        results["pytest_message"] = test_msg
        results["pytest_count"] = test_count

        # カバレッジ測定
        cov_passed, cov_msg, cov_pct = self.run_coverage(test_dir)
        results["coverage_passed"] = cov_passed
        results["coverage_message"] = cov_msg
        results["coverage_pct"] = cov_pct

        # 判定
        num_failures = sum(
            1
            for v in [test_passed, cov_passed]
            if v is False
        )

        if num_failures == 2 or (
            cov_pct is not None and cov_pct < 50
        ):
            results["status"] = "FAIL"
        elif num_failures == 1 or (
            cov_pct is not None and (50 <= cov_pct < self.min_coverage)
        ):
            results["status"] = "WARN"
        elif test_passed is None or cov_passed is None:
            results["status"] = "N/A"
        else:
            results["status"] = "PASS"

        return results


def main() -> int:
    """メイン処理。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Layer A: Test Coverage Check（テストカバレッジ チェッカー）"
    )
    parser.add_argument(
        "--test-dir",
        default="python/tests",
        help="テストディレクトリ（デフォルト: python/tests）",
    )
    parser.add_argument(
        "--min-coverage",
        type=int,
        default=70,
        help="最小カバレッジ % （デフォルト: 70）",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="詳細出力"
    )
    parser.add_argument(
        "--json", action="store_true", help="JSON 形式で出力"
    )

    args = parser.parse_args()

    checker = TestCoverageChecker(verbose=args.verbose)
    checker.min_coverage = args.min_coverage
    results = checker.check(args.test_dir)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"[Test Coverage Check] Status: {results['status']}")
        print(f"  Pytest: {results['pytest_message']}")
        print(f"  Coverage: {results['coverage_message']}")

    # 終了コード
    if results["status"] == "FAIL":
        return 1
    elif results["status"] == "WARN":
        return 2
    elif results["status"] == "N/A":
        return 3
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
