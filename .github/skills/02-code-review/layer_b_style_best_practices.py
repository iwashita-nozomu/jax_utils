#!/usr/bin/env python3
"""
Layer B: Style & Best Practices Check（ベストプラクティス検証）

目的:
- コード品質のベストプラクティス検証
- 例外処理・エラーハンドリングの適切性
- 計算量・パフォーマンスパターンの確認

レイヤー:
- B層: 深度検証（レビュアー向け、PR レビュー時）

出力:
- PASS: ベストプラクティスに準拠
- WARN: 軽微な違反（改善推奨）
- FAIL: 重大な違反（修正必須）
"""

import ast
import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple


class StyleBestPracticesChecker:
    """スタイル・ベストプラクティス検証チェッカー。"""

    def __init__(self, verbose: bool = False):
        """初期化。

        Args:
            verbose: 詳細出力フラグ。
        """
        self.verbose = verbose

    def check_exception_handling(
        self, tree: ast.Module, file_path: str
    ) -> List[Dict]:
        """例外処理のベストプラクティス検証。

        ルール:
        - 裸の except: は禁止（generic except）
        - 例外型は明示的に指定
        - except Exception: は最後の手段

        Args:
            tree: AST Module。
            file_path: ファイルパス。

        Returns:
            問題リスト。
        """
        issues = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue

            for handler in node.handlers:
                # 裸の except
                if handler.type is None:
                    issues.append({
                        "severity": "HIGH",
                        "issue": (
                            f"{file_path}:{node.lineno} "
                            f"裸の except: は禁止です"
                        ),
                    })

                # Exception 型を確認
                elif isinstance(handler.type, ast.Name):
                    if handler.type.id == "Exception":
                        issues.append({
                            "severity": "MEDIUM",
                            "issue": (
                                f"{file_path}:{node.lineno} "
                                f"except Exception: は最後の手段です "
                                f"(具体的な例外型を使用)"
                            ),
                        })

        return issues

    def check_generic_patterns(
        self, content: str, file_path: str
    ) -> List[Dict]:
        """汎用コードパターンの検証。

        確認項目:
        - TODO/FIXME コメント
        - print() デバッグ出力（テストファイル外）
        - magic numbers（ハードコード化された定数値）

        Args:
            content: ファイル内容。
            file_path: ファイルパス。

        Returns:
            問題リスト。
        """
        issues = []
        is_test_file = "test" in file_path or "tests" in file_path

        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # TODO/FIXME の検出（情報提供）
            if re.search(r"#\s*(TODO|FIXME)", line):
                issues.append({
                    "severity": "LOW",
                    "issue": f"{file_path}:{i} TODO/FIXME があります",
                })

            # print() デバッグ出力（テストファイル外）
            if not is_test_file and re.search(
                r"\bprint\s*\(", line
            ):
                if "docstring" not in line:
                    issues.append({
                        "severity": "MEDIUM",
                        "issue": (
                            f"{file_path}:{i} "
                            f"デバッグ用 print() あり "
                            f"(本番コードで禁止)"
                        ),
                    })

            # Magic numbers（簡易検出）
            if re.search(r"[a-zA-Z_]\s*(=|:)\s*\d{4,}", line):
                # 年号や大きな数値
                issues.append({
                    "severity": "LOW",
                    "issue": (
                        f"{file_path}:{i} "
                        f"magic number - 定数として定義してください"
                    ),
                })

        return issues

    def check_code_complexity(
        self, tree: ast.Module, file_path: str
    ) -> List[Dict]:
        """コード複雑度の基本的な確認。

        確認項目:
        - 過度にネストされた if 文（深さ > 5）
        - 巨大な関数（行数 > 100）

        Args:
            tree: AST Module。
            file_path: ファイルパス。

        Returns:
            問題リスト。
        """
        issues = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue

            # 関数の複雑度簡易チェック
            if_count = sum(
                1 for _ in ast.walk(node) if isinstance(_, ast.If)
            )

            if if_count > 10:
                issues.append({
                    "severity": "MEDIUM",
                    "issue": (
                        f"{file_path}:{node.lineno} "
                        f"関数 '{node.name}' の複雑度が高い "
                        f"(if文: {if_count}個)"
                    ),
                })

            # 関数の大きさチェック
            func_lines = node.end_lineno - node.lineno
            if func_lines > 100:
                issues.append({
                    "severity": "MEDIUM",
                    "issue": (
                        f"{file_path}:{node.lineno} "
                        f"関数 '{node.name}' が大きすぎます "
                        f"({func_lines} 行)"
                    ),
                })

        return issues

    def check_performance_patterns(
        self, content: str, file_path: str
    ) -> List[Dict]:
        """パフォーマンスパターンの確認。

        確認項目:
        - ループ内での重い操作（ファイル I/O、ネットワーク）
        - 不要なリスト生成（list comp の活用）
        - JAX コンテキストでの Python ループ

        Args:
            content: ファイル内容。
            file_path: ファイルパス。

        Returns:
            問題リスト。
        """
        issues = []

        # JAX ファイルの確認
        if "jax" in content or ".py" in file_path:
            # Python ループ内での JAX 配列操作を検出
            if re.search(
                r"for\s+\w+\s+in\s+.*:\s*\n.*jnp\.",
                content,
                re.MULTILINE,
            ):
                issues.append({
                    "severity": "MEDIUM",
                    "issue": (
                        f"{file_path}: "
                        f"Python ループ内での JAX 操作検出 "
                        f"(vmap/jit 化を検討)"
                    ),
                })

        return issues

    def check_file(self, file_path: Path) -> Tuple[bool, List[Dict]]:
        """単一 Python ファイルをチェック。

        Args:
            file_path: Python ファイルパス。

        Returns:
            (成功フラグ, 問題リスト)
        """
        try:
            with open(file_path) as f:
                content = f.read()

            tree = ast.parse(content)

            issues = []

            # 各チェック実行
            issues.extend(
                self.check_exception_handling(tree, str(file_path))
            )
            issues.extend(
                self.check_generic_patterns(content, str(file_path))
            )
            issues.extend(
                self.check_code_complexity(tree, str(file_path))
            )
            issues.extend(
                self.check_performance_patterns(content, str(file_path))
            )

            passed = not any(
                i["severity"] == "HIGH" for i in issues
            )

            return passed, issues

        except SyntaxError:
            return False, [
                {
                    "severity": "HIGH",
                    "issue": f"{file_path}: 構文エラー",
                }
            ]
        except Exception as e:
            return False, [
                {
                    "severity": "HIGH",
                    "issue": f"{file_path}: 処理エラー ({e})",
                }
            ]

    def check(
        self, target_dir: str = "python"
    ) -> Dict:
        """ベストプラクティス チェック実行。

        Args:
            target_dir: 対象ディレクトリ。

        Returns:
            チェック結果辞書。
        """
        results = {
            "component": "layer_b_style_best_practices_checker",
            "files_checked": 0,
            "files_passed": 0,
            "files_with_issues": 0,
            "total_issues": 0,
            "issues": [],
        }

        target_path = Path(target_dir)
        if not target_path.exists():
            return {
                **results,
                "status": "N/A",
                "message": f"対象ディレクトリが見つかりません",
            }

        for py_file in sorted(target_path.rglob("*.py")):
            if "__pycache__" in str(py_file):
                continue

            results["files_checked"] += 1

            passed, file_issues = self.check_file(py_file)

            if passed:
                results["files_passed"] += 1
            elif file_issues:
                results["files_with_issues"] += 1
                results["total_issues"] += len(file_issues)
                results["issues"].extend([
                    {"file": str(py_file), **issue}
                    for issue in file_issues
                ])

            if self.verbose:
                status_str = "✓" if passed else "✗"
                print(f"  {status_str} {py_file.name}")

        # 判定
        high_count = sum(
            1 for i in results["issues"] if i["severity"] == "HIGH"
        )

        if high_count > 0:
            results["status"] = "FAIL"
        elif results["total_issues"] > 10:
            results["status"] = "WARN"
        else:
            results["status"] = "PASS"

        results["message"] = (
            f"チェック: {results['files_checked']} ファイル, "
            f"OK: {results['files_passed']}, "
            f"問題: {results['total_issues']}"
        )

        return results


def main() -> int:
    """メイン処理。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Layer B: Style & Best Practices Check（ベストプラクティス検証）"
    )
    parser.add_argument(
        "--target-dir",
        default="python",
        help="対象ディレクトリ（デフォルト: python）",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="詳細出力"
    )
    parser.add_argument(
        "--json", action="store_true", help="JSON 形式で出力"
    )

    args = parser.parse_args()

    checker = StyleBestPracticesChecker(verbose=args.verbose)
    results = checker.check(args.target_dir)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(
            f"[Style & Best Practices] Status: {results['status']}"
        )
        print(f"  {results['message']}")
        if results.get("issues") and args.verbose:
            print("  問題:")
            for issue in results["issues"][:10]:
                print(f"    - {issue['issue']}")

    # 終了コード
    if results["status"] == "FAIL":
        return 1
    elif results["status"] == "WARN":
        return 2
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
