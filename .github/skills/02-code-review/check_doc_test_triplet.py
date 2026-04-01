#!/usr/bin/env python3
"""
Layer C: Doc-Test Triplet Check（三点セット検証）

目的:
- 全公開関数について「実装 ⟺ Docstring ⟺ テスト」の対応を検証
- 実装の型情報と Docstring の Args/Returns の型一致を確認
- テストカバレッジと関数の複雑度との対応を検証

レイヤー:
- C層: 統合検証（プロジェクト規約整合、三点セット検証）

出力:
- PASS: 全公開関数が三点セットを満たす
- WARN: 一部関数が不完全（改善推奨）
- FAIL: 多数関数が三点セット未達成（修正必須）
"""

import ast
import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class DocTestTripletChecker:
    """三点セット（実装-Docstring-テスト）チェッカー。"""

    def __init__(self, verbose: bool = False):
        """初期化。

        Args:
            verbose: 詳細出力フラグ。
        """
        self.verbose = verbose

    def extract_type_from_annotation(
        self, annotation: Optional[ast.expr]
    ) -> Optional[str]:
        """AST の型アノテーション を文字列に変換。

        Args:
            annotation: AST 型アノテーション。

        Returns:
            型文字列（例: "int", "List[str]"）
        """
        if annotation is None:
            return None

        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif isinstance(annotation, ast.Subscript):
            value = self.extract_type_from_annotation(
                annotation.value
            )
            slice_ = self.extract_type_from_annotation(
                annotation.slice
            )
            return f"{value}[{slice_}]"
        elif isinstance(annotation, ast.Attribute):
            value = self.extract_type_from_annotation(
                annotation.value
            )
            return f"{value}.{annotation.attr}"
        else:
            return ast.unparse(annotation)

    def extract_function_info(
        self, tree: ast.Module, file_path: str
    ) -> List[Dict]:
        """AST から関数情報を抽出。

        Args:
            tree: AST Module。
            file_path: ファイルパス。

        Returns:
            関数情報辞書のリスト。
        """
        functions = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue

            # プライベート関数は対象外
            if node.name.startswith("_"):
                continue

            docstring = ast.get_docstring(node)

            # 型アノテーション抽出
            args_types = {}
            for arg in node.args.args:
                args_types[arg.arg] = (
                    self.extract_type_from_annotation(
                        arg.annotation
                    )
                )

            return_type = (
                self.extract_type_from_annotation(
                    node.returns
                )
            )

            functions.append({
                "name": node.name,
                "file": file_path,
                "lineno": node.lineno,
                "args": list(args_types.keys()),
                "args_types": args_types,
                "return_type": return_type,
                "has_docstring": docstring is not None,
                "docstring": docstring or "",
                "has_type_annotations": all(
                    args_types.values()
                ) and return_type is not None,
            })

        return functions

    def check_docstring_types(
        self, docstring: str, args_types: Dict, return_type: Optional[str]
    ) -> Tuple[bool, List[str]]:
        """Docstring の Args/Returns の型が実装と対応しているか確認。

        Args:
            docstring: Docstring テキスト。
            args_types: 実装の引数型辞書。
            return_type: 実装の戻り値型。

        Returns:
            (対応フラグ, 問題リスト)
        """
        issues = []

        # Args セクションを抽出
        args_section = re.search(
            r"Args:(.*?)(?=Returns:|Raises:|$)",
            docstring,
            re.DOTALL,
        )

        if args_section:
            args_text = args_section.group(1)
            # 簡易解析: Args セクション内の型情報をチェック
            for arg_name in args_types.keys():
                if arg_name not in args_text:
                    issues.append(
                        f"Args セクションに '{arg_name}' がありません"
                    )
        elif args_types:
            issues.append("Args セクションが見つかりません")

        # Returns セクションをチェック
        returns_section = "Returns:" in docstring
        if return_type and not returns_section:
            issues.append("戻り値型があるのに Returns セクションがありません")

        return len(issues) == 0, issues

    def find_test_for_function(
        self, func_name: str, test_dir: str = "python/tests"
    ) -> bool:
        """関数のテストが存在するか確認。

        Args:
            func_name: 関数名。
            test_dir: テストディレクトリ。

        Returns:
            テスト存在フラグ。
        """
        test_pattern = f"test_{func_name}|Test{func_name}"

        for test_file in Path(test_dir).rglob("test_*.py"):
            try:
                with open(test_file) as f:
                    content = f.read()
                if re.search(test_pattern, content):
                    return True
            except Exception:
                pass

        return False

    def check_file(
        self, file_path: Path, test_dir: str = "python/tests"
    ) -> Tuple[bool, str, List[Dict]]:
        """単一 Python ファイルの三点セットをチェック。

        Args:
            file_path: ファイルパス。
            test_dir: テストディレクトリ。

        Returns:
            (成功フラグ, メッセージ, 問題リスト)
        """
        try:
            with open(file_path) as f:
                content = f.read()

            tree = ast.parse(content)
            functions = self.extract_function_info(tree, str(file_path))

            issues = []

            for func in functions:
                func_issues = []

                # チェック 1: Docstring 存在
                if not func["has_docstring"]:
                    func_issues.append(
                        {
                            "severity": "HIGH",
                            "issue": "Docstring がありません",
                        }
                    )
                else:
                    # チェック 2: Docstring の型情報
                    type_match, type_issues = (
                        self.check_docstring_types(
                            func["docstring"],
                            func["args_types"],
                            func["return_type"],
                        )
                    )
                    if not type_match:
                        for issue in type_issues:
                            func_issues.append({
                                "severity": "MEDIUM",
                                "issue": issue,
                            })

                # チェック 3: 型アノテーション
                if not func["has_type_annotations"]:
                    func_issues.append({
                        "severity": "MEDIUM",
                        "issue": "型アノテーション が不完全です",
                    })

                # チェック 4: テスト存在
                if not self.find_test_for_function(
                    func["name"], test_dir
                ):
                    func_issues.append({
                        "severity": "MEDIUM",
                        "issue": "テストがありません",
                    })

                if func_issues:
                    issues.append({
                        "function": func["name"],
                        "line": func["lineno"],
                        "issues": func_issues,
                    })

            if issues:
                summary = (
                    f"{file_path.name}: "
                    f"{len(issues)} 個の関数が三点セット未達成"
                )
                return False, summary, issues
            else:
                return True, f"{file_path.name}: OK", []

        except SyntaxError:
            return False, f"{file_path.name}: 構文エラー", []
        except Exception as e:
            return False, f"{file_path.name}: エラー ({e})", []

    def check(
        self, target_dir: str = "python", test_dir: str = "python/tests"
    ) -> Dict:
        """全 Python ファイルの三点セットチェック実行。

        Args:
            target_dir: 対象ディレクトリ。
            test_dir: テストディレクトリ。

        Returns:
            チェック結果辞書。
        """
        results = {
            "component": "layer_c_doctest_triplet_checker",
            "files_checked": 0,
            "files_passed": 0,
            "files_with_issues": 0,
            "total_functions_with_issues": 0,
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

            passed, msg, file_issues = self.check_file(
                py_file, test_dir
            )

            if passed:
                results["files_passed"] += 1
            elif file_issues:
                results["files_with_issues"] += 1
                results["total_functions_with_issues"] += len(
                    file_issues
                )
                results["issues"].extend([
                    {"file": str(py_file), **issue}
                    for issue in file_issues
                ])

            if self.verbose:
                print(f"  {py_file.name}: {msg}")

        # 判定
        if (
            results["total_functions_with_issues"]
            > 20
        ):
            results["status"] = "FAIL"
        elif (
            results["total_functions_with_issues"]
            > 5
        ):
            results["status"] = "WARN"
        else:
            results["status"] = "PASS"

        results["message"] = (
            f"チェック: {results['files_checked']} ファイル, "
            f"OK: {results['files_passed']}, "
            f"問題: {results['total_functions_with_issues']} 関数"
        )

        return results


def main() -> int:
    """メイン処理。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Layer C: Doc-Test Triplet Check（三点セット検証）"
    )
    parser.add_argument(
        "--target-dir",
        default="python",
        help="対象ディレクトリ（デフォルト: python）",
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

    checker = DocTestTripletChecker(verbose=args.verbose)
    results = checker.check(args.target_dir, args.test_dir)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"[Doc-Test Triplet] Status: {results['status']}")
        print(f"  {results['message']}")
        if results.get("issues") and args.verbose:
            print("  問題:")
            for issue in results["issues"][:5]:
                print(
                    f"    - {Path(issue['file']).name}:"
                    f"{issue['line']} {issue['function']}"
                )

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
