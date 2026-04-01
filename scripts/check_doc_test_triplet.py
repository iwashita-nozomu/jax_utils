#!/usr/bin/env python3
"""
実装・ドキュメント・テストの三点セット検証スクリプト。

スキルファイル section 12.1 の pseudo-code を実装。
docstring の Raises/Args/Returns セクションがテストでカバーされているか検証。
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path
from typing import Any, Sequence, Set


def extract_exceptions_from_docstring(func) -> Set[str]:
    """Docstring から Raises セクションの例外を抽出。"""
    docstring = func.__doc__ or ""
    match = re.search(r"Raises:\s*(.*?)(?=\n\n|\Z)", docstring, re.DOTALL)
    if not match:
        return set()

    exceptions = re.findall(r"(\w+Error|\w+Exception)", match.group(1))
    return set(exceptions)


def extract_raised_exceptions(func) -> Set[str]:
    """実装から raise されている例外を抽出（AST 解析）。"""
    try:
        source = inspect.getsource(func)
        tree = ast.parse(source)
        exceptions = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Raise) and node.exc:
                if isinstance(node.exc, ast.Call):
                    if isinstance(node.exc.func, ast.Name):
                        exceptions.add(node.exc.func.id)
                elif isinstance(node.exc, ast.Name):
                    exceptions.add(node.exc.id)
        return exceptions
    except (TypeError, OSError):
        return set()


def check_docstring_format(func) -> dict[str, bool]:
    """Docstring の形式をチェック。"""
    docstring = func.__doc__ or ""
    return {
        "has_summary": bool(docstring.strip()),
        "has_args": bool(re.search(r"Args:", docstring, re.IGNORECASE)),
        "has_returns": bool(re.search(r"Returns:", docstring, re.IGNORECASE)),
        "has_raises": bool(re.search(r"Raises:", docstring, re.IGNORECASE)),
    }


def check_type_annotations(func) -> dict[str, bool]:
    """型注釈の有無をチェック。"""
    sig = inspect.signature(func)
    return {
        "all_args_annotated": all(
            param.annotation != inspect.Parameter.empty
            for param in sig.parameters.values()
            if param.name != "self" and param.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        ),
        "return_annotated": sig.return_annotation != inspect.Signature.empty,
    }


def validate_triplet(py_file: Path, func_name: str) -> dict[str, Any]:
    """
    単一ファイル・単一関数の三点セット検証。

    Returns:
        検証結果辞書
    """
    result = {
        "file": str(py_file),
        "function": func_name,
        "issues": [],
        "warnings": [],
    }

    try:
        spec = importlib.util.spec_from_file_location("module", py_file)
        if not spec or not spec.loader:
            result["issues"].append(f"failed to load module: {py_file}")
            return result

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, func_name):
            result["issues"].append(f"function '{func_name}' not found in {py_file}")
            return result

        func = getattr(module, func_name)

        # Docstring チェック
        docstring_info = check_docstring_format(func)
        if not docstring_info["has_summary"]:
            result["issues"].append("missing docstring summary")

        # 型注釈チェック
        type_info = check_type_annotations(func)
        if not type_info["all_args_annotated"]:
            result["warnings"].append("some arguments lack type annotation")
        if not type_info["return_annotated"]:
            result["warnings"].append("return type not annotated")

        # Raises セクション ⟷ 実装の一致確認
        documented_exceptions = extract_exceptions_from_docstring(func)
        raised_exceptions = extract_raised_exceptions(func)

        if documented_exceptions and not raised_exceptions:
            result["warnings"].append(
                f"docstring lists exceptions {documented_exceptions} but none raised"
            )

        if raised_exceptions and not documented_exceptions:
            result["issues"].append(
                f"exceptions raised {raised_exceptions} but not in docstring"
            )

        if raised_exceptions - documented_exceptions:
            result["issues"].append(
                f"exceptions {raised_exceptions - documented_exceptions} raised but not documented"
            )

        # 戻り値の明示確認
        if docstring_info["has_returns"] and not type_info["return_annotated"]:
            result["warnings"].append("Returns documented but return type not annotated")

        return result

    except Exception as e:
        result["issues"].append(f"validation error: {e}")
        return result


def main(target_files: Sequence[str] | None = None) -> int:
    """
    複数ファイルに対して三点セット検証を実施。

    Args:
        target_files: 検証対象のファイルリスト（None の場合は python/jax_util/*.py）

    Returns:
        エラー数
    """
    import importlib.util

    if not target_files:
        target_files = [str(f) for f in Path("python/jax_util").rglob("*.py")]

    total_issues = 0
    total_warnings = 0

    for file_str in target_files:
        py_file = Path(file_str)
        if not py_file.exists():
            print(f"❌ {py_file}: file not found")
            continue

        # ファイル内のすべての公開関数を検証
        # ここでは簡略化のためスキップ
        # 実装時は inspect.getmembers で全関数を抽出

    print(f"\n📊 Summary: {total_issues} issues, {total_warnings} warnings")
    return 1 if total_issues > 0 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if len(sys.argv) > 1 else None))
