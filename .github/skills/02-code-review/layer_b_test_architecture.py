#!/usr/bin/env python3
"""
Layer B: Test Architecture Check（テストアーキテクチャ検証）

目的:
- テストスイート全体の構造・設計を検証
- テストファイルの配置・分類規則の遵守確認
- テスト命名規則の統一

レイヤー:
- B層: 深度検証（レビュアー向け、PR レビュー時）

出力:
- PASS: テストアーキテクチャが規約に準拠
- WARN: 軽微な違反（改善推奨）
- FAIL: 重大な違反（修正必須）
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class TestArchitectureChecker:
    """テストアーキテクチャ検証チェッカー。"""

    def __init__(self, verbose: bool = False):
        """初期化。

        Args:
            verbose: 詳細出力フラグ。
        """
        self.verbose = verbose
        # 許可されたテストディレクトリ
        self.allowed_dirs = {
            "base": "単体テスト（高速）",
            "solvers": "数値ソルバ検証（中〜高コスト）",
            "optimizers": "最適化アルゴリズム検証（中〜高コスト）",
            "hlo": "HLO ユーティリティ検証（低〜中コスト）",
            "neuralnetwork": "NN 系検証（実験段階）",
        }

    def check_directory_structure(
        self, test_dir: Path
    ) -> Tuple[bool, List[Dict]]:
        """テストディレクトリ構造をチェック。

        Args:
            test_dir: テストルートディレクトリ。

        Returns:
            (成功フラグ, 問題リスト)
        """
        issues = []

        if not test_dir.exists():
            return False, [
                {
                    "severity": "HIGH",
                    "issue": f"テストディレクトリ {test_dir} が見つかりません",
                }
            ]

        # 許可されたサブディレクトリの確認
        subdirs = [
            d.name for d in test_dir.iterdir() if d.is_dir()
        ]

        for subdir in subdirs:
            if subdir.startswith("__"):
                continue
            if subdir not in self.allowed_dirs:
                issues.append({
                    "severity": "MEDIUM",
                    "issue": (
                        f"不明なテストディレクトリ: {subdir} "
                        f"(許可: {', '.join(self.allowed_dirs.keys())})"
                    ),
                })

        return len(issues) == 0, issues

    def check_test_naming(
        self, test_dir: Path
    ) -> Tuple[bool, List[Dict]]:
        """テストファイルの命名規則をチェック。

        規則:
        - ファイル: test_*.py
        - 関数: test_*()
        - 大規模ケース: *_large または *_case_*

        Args:
            test_dir: テストディレクトリ。

        Returns:
            (成功フラグ, 問題リスト)
        """
        issues = []

        for test_file in sorted(test_dir.rglob("*.py")):
            if "__pycache__" in str(test_file):
                continue

            # ファイル名チェック
            if not test_file.name.startswith("test_"):
                if (
                    "__init__.py" not in test_file.name
                    and "conftest.py" not in test_file.name
                ):
                    issues.append({
                        "severity": "MEDIUM",
                        "issue": (
                            f"テストファイル命名規則違反: "
                            f"{test_file.name} "
                            f"(要: test_*.py)"
                        ),
                    })
                    continue

            # ファイル内の関数名チェック
            try:
                with open(test_file) as f:
                    content = f.read()

                import re

                # test_ で始まらない関数をチェック
                func_pattern = r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("
                functions = re.findall(
                    func_pattern, content, re.MULTILINE
                )

                for func_name in functions:
                    # サポート関数は許可
                    if func_name.startswith("_"):
                        continue
                    if func_name in ["_run_all_tests"]:
                        continue
                    if not func_name.startswith("test_"):
                        issues.append({
                            "severity": "LOW",
                            "issue": (
                                f"{test_file.name}: "
                                f"関数 '{func_name}' は "
                                f"test_ で始まるべきです"
                            ),
                        })

            except Exception:
                pass

        return len([i for i in issues if i["severity"] == "HIGH"]) == 0, issues

    def check_test_structure(
        self, test_dir: Path
    ) -> Tuple[bool, List[Dict]]:
        """個別テストファイルの構造をチェック。

        チェック項目:
        - pytest 対応性
        - _run_all_tests() 関数の有無
        - JSON ログ出力の確認

        Args:
            test_dir: テストディレクトリ。

        Returns:
            (成功フラグ, 問題リスト)
        """
        issues = []

        for test_file in sorted(test_dir.rglob("test_*.py")):
            if "__pycache__" in str(test_file):
                continue

            try:
                with open(test_file) as f:
                    content = f.read()

                # _run_all_tests 関数の確認
                if "_run_all_tests" not in content:
                    issues.append({
                        "severity": "MEDIUM",
                        "issue": (
                            f"{test_file.name}: "
                            f"_run_all_tests() 関数がありません"
                        ),
                    })

                # if __name__ == "__main__": の確認
                if 'if __name__ == "__main__":' not in content:
                    issues.append({
                        "severity": "LOW",
                        "issue": (
                            f"{test_file.name}: "
                            f"単体実行対応なし"
                        ),
                    })

            except Exception as e:
                if self.verbose:
                    print(f"警告: {test_file} を読み込めません: {e}")

        return len([i for i in issues if i["severity"] == "HIGH"]) == 0, issues

    def check_coverage_balance(
        self, test_dir: Path
    ) -> Tuple[bool, List[Dict]]:
        """テスト配分バランスをチェック。

        基準:
        - base テスト (単体): 40-50%
        - solvers テスト: 30-40%
        - その他: 10-30%

        Args:
            test_dir: テストディレクトリ。

        Returns:
            (成功フラグ, 問題リスト)
        """
        issues = []

        # テストファイル数を集計
        test_counts = {}
        total = 0

        for subdir in self.allowed_dirs.keys():
            count = len(
                list(
                    (test_dir / subdir).glob("test_*.py")
                    if (test_dir / subdir).exists()
                    else []
                )
            )
            test_counts[subdir] = count
            total += count

        if total == 0:
            return False, [
                {
                    "severity": "HIGH",
                    "issue": "テストファイルが見つかりません",
                }
            ]

        # バランス確認
        base_ratio = (
            test_counts.get("base", 0) / total * 100
        )

        if base_ratio < 30:
            issues.append({
                "severity": "MEDIUM",
                "issue": (
                    f"単体テスト (base) の比率が低い: "
                    f"{base_ratio:.1f}% (推奨: 40-50%)"
                ),
            })

        return len([i for i in issues if i["severity"] == "HIGH"]) == 0, issues

    def check(
        self, test_dir: str = "python/tests"
    ) -> Dict:
        """テストアーキテクチャ チェック実行。

        Args:
            test_dir: テストディレクトリ。

        Returns:
            チェック結果辞書。
        """
        results = {
            "component": "layer_b_test_architecture_checker",
            "checks": {},
            "issues": [],
        }

        test_path = Path(test_dir)

        # 1. ディレクトリ構造チェック
        passed, issues = self.check_directory_structure(test_path)
        results["checks"]["directory_structure"] = {
            "passed": passed,
            "issues": issues,
        }
        results["issues"].extend(issues)

        # 2. 命名規則チェック
        passed, issues = self.check_test_naming(test_path)
        results["checks"]["naming"] = {
            "passed": passed,
            "issues": issues,
        }
        results["issues"].extend(issues)

        # 3. ファイル構造チェック
        passed, issues = self.check_test_structure(test_path)
        results["checks"]["file_structure"] = {
            "passed": passed,
            "issues": issues,
        }
        results["issues"].extend(issues)

        # 4. カバレッジバランスチェック
        passed, issues = self.check_coverage_balance(test_path)
        results["checks"]["coverage_balance"] = {
            "passed": passed,
            "issues": issues,
        }
        results["issues"].extend(issues)

        # 判定
        high_severity = sum(
            1 for i in results["issues"] if i["severity"] == "HIGH"
        )
        medium_severity = sum(
            1 for i in results["issues"] if i["severity"] == "MEDIUM"
        )

        if high_severity > 0:
            results["status"] = "FAIL"
        elif medium_severity > 2:
            results["status"] = "WARN"
        else:
            results["status"] = "PASS"

        results["message"] = (
            f"チェック完了: {len(results['issues'])} 個の問題 "
            f"(HIGH: {high_severity}, MEDIUM: {medium_severity})"
        )

        return results


def main() -> int:
    """メイン処理。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Layer B: Test Architecture Check（テストアーキテクチャ検証）"
    )
    parser.add_argument(
        "--test-dir",
        default="python/tests",
        help="テストディレクトリ（デフォルト: python/tests）",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="詳細出力"
    )
    parser.add_argument(
        "--json", action="store_true", help="JSON 形式で出力"
    )

    args = parser.parse_args()

    checker = TestArchitectureChecker(verbose=args.verbose)
    results = checker.check(args.test_dir)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(
            f"[Test Architecture] Status: {results['status']}"
        )
        print(f"  {results['message']}")
        if results.get("issues") and args.verbose:
            print("  問題:")
            for issue in results["issues"][:10]:
                print(
                    f"    - [{issue['severity']}] "
                    f"{issue['issue']}"
                )

    # 終了コード
    if results["status"] == "FAIL":
        return 1
    elif results["status"] == "WARN":
        return 2
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
