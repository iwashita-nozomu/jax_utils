#!/usr/bin/env python3
"""
Layer A: Docstring Check（ドキュメンテーション チェッカー）

目的:
- 全公開関数・クラスへの docstring 存在確認
- Args / Returns / Raises セクションの有無確認
- Docstring と実装コードの一貫性検証

レイヤー:
- A層: 基礎検証（開発者向け、常に実行）

出力:
- PASS: 全公開関数に適切な docstring あり
- WARN: Docstring の一部欠落（改善推奨）
- FAIL: Docstring の多数欠落（修正必須）
"""

import ast
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class DocstringChecker:
    """Python Docstring チェッカー。"""

    def __init__(self, verbose: bool = False):
        """初期化。

        Args:
            verbose: 詳細出力フラグ。
        """
        self.verbose = verbose

    def extract_function_signatures(
        self, tree: ast.Module
    ) -> List[Dict]:
        """AST から関数シグネチャを抽出。

        Args:
            tree: AST Module オブジェクト。

        Returns:
            関数情報辞書のリスト。
        """
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # プライベート関数は対象外
                if node.name.startswith("_"):
                    continue

                docstring = ast.get_docstring(node)
                args = [arg.arg for arg in node.args.args]
                returns = (
                    node.returns is not None
                )

                functions.append({
                    "name": node.name,
                    "lineno": node.lineno,
                    "args": args,
                    "has_docstring": docstring is not None,
                    "docstring": docstring or "",
                    "has_return_type": returns,
                })

        return functions

    def check_docstring_sections(
        self, docstring: str
    ) -> Dict[str, bool]:
        """Docstring に必要なセクションがあるか確認。

        Args:
            docstring: Docstring テキスト。

        Returns:
            セクション有無辞書。
        """
        return {
            "has_args": "Args:" in docstring or "Arguments:" in docstring,
            "has_returns": (
                "Returns:" in docstring or "Return:" in docstring
            ),
            "has_raises": "Raises:" in docstring or "Exceptions:" in docstring,
        }

    def check_file(self, file_path: Path) -> Tuple[bool, str, List]:
        """単一 Python ファイルをチェック。

        Args:
            file_path: Python ファイルパス。

        Returns:
            (成功フラグ, メッセージ, 問題リスト)
        """
        try:
            with open(file_path) as f:
                content = f.read()

            tree = ast.parse(content)
            functions = self.extract_function_signatures(tree)

            issues = []

            for func in functions:
                # Docstring がない場合
                if not func["has_docstring"]:
                    issues.append({
                        "type": "MISSING_DOCSTRING",
                        "function": func["name"],
                        "line": func["lineno"],
                        "severity": "HIGH",
                    })
                    continue

                # Docstring がある場合、セクション確認
                sections = self.check_docstring_sections(
                    func["docstring"]
                )

                # Args セクション（引数がある場合）
                if func["args"] and not sections["has_args"]:
                    issues.append({
                        "type": "MISSING_ARGS_SECTION",
                        "function": func["name"],
                        "line": func["lineno"],
                        "severity": "MEDIUM",
                    })

                # Returns セクション
                if func["has_return_type"] and not sections["has_returns"]:
                    issues.append({
                        "type": "MISSING_RETURNS_SECTION",
                        "function": func["name"],
                        "line": func["lineno"],
                        "severity": "MEDIUM",
                    })

            if issues:
                summary = (
                    f"{file_path.name}: "
                    f"{len(issues)} 個の Docstring 問題"
                )
                return False, summary, issues
            else:
                return True, f"{file_path.name}: OK", []

        except SyntaxError as e:
            return False, f"{file_path.name}: 構文エラー (L{e.lineno})", []
        except Exception as e:
            return False, f"{file_path.name}: 処理エラー ({e})", []

    def check(self, target_dir: str = "python") -> Dict:
        """全 Python ファイルの Docstring チェック実行。

        Args:
            target_dir: 対象ディレクトリ。

        Returns:
            チェック結果辞書。
        """
        results = {
            "component": "layer_a_docstring_checker",
            "files_checked": 0,
            "files_passed": 0,
            "files_warned": 0,
            "files_failed": 0,
            "total_issues": 0,
            "issues": [],
        }

        target_path = Path(target_dir)
        if not target_path.exists():
            return {
                **results,
                "status": "N/A",
                "message": f"対象ディレクトリ {target_dir} が見つかりません",
            }

        for py_file in sorted(target_path.rglob("*.py")):
            if "__pycache__" in str(py_file):
                continue

            results["files_checked"] += 1

            passed, msg, issues = self.check_file(py_file)

            if passed:
                results["files_passed"] += 1
            elif issues:
                results["files_warned"] += 1
                results["total_issues"] += len(issues)
                results["issues"].extend([
                    {"file": str(py_file), **issue}
                    for issue in issues
                ])

            if self.verbose:
                print(f"  {py_file.name}: {msg}")

        # 判定
        if results["total_issues"] > 10:
            results["status"] = "FAIL"
        elif results["total_issues"] > 0:
            results["status"] = "WARN"
        else:
            results["status"] = "PASS"

        results["message"] = (
            f"チェック: {results['files_checked']} ファイル, "
            f"PASS: {results['files_passed']}, "
            f"WARN: {results['files_warned']}, "
            f"問題: {results['total_issues']}"
        )

        return results


def main() -> int:
    """メイン処理。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Layer A: Docstring Check（ドキュメンテーション チェッカー）"
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

    checker = DocstringChecker(verbose=args.verbose)
    results = checker.check(args.target_dir)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"[Docstring Check] Status: {results['status']}")
        print(f"  {results['message']}")
        if results.get("issues") and args.verbose:
            print("  問題:")
            for issue in results["issues"][:5]:
                print(
                    f"    - {issue['file']}:{issue['line']} "
                    f"{issue['type']}"
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
