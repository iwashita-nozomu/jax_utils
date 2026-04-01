"""
Phase 3: Tools & Scripts Checker

ツール・スクリプトの実装状況をチェック：
- スクリプト実装状況
- テストカバレッジ
- ドキュメント参照
- 依存関係
"""

import re
import ast
from pathlib import Path
from typing import Dict, List, Set


def find_scripts(workspace_root: Path) -> Dict[str, dict]:
    """スクリプトファイルを検出"""
    scripts = {}
    scripts_dir = workspace_root / "scripts"
    
    if scripts_dir.exists():
        for script_file in scripts_dir.glob("*.py"):
            scripts[script_file.name] = {"path": script_file, "type": "script"}
        for script_file in scripts_dir.glob("*.sh"):
            scripts[script_file.name] = {"path": script_file, "type": "shell"}
    
    # Skill 内の checker スクリプト
    skills_root = workspace_root / ".github" / "skills"
    if skills_root.exists():
        for checker_file in skills_root.glob("*/checkers/*.py"):
            scripts[checker_file.name] = {"path": checker_file, "type": "checker"}
    
    return scripts


def check_script_implementation(script_path: Path) -> dict:
    """スクリプトの実装状況をチェック"""
    result = {
        "path": str(script_path),
        "exists": script_path.exists(),
        "has_content": False,
        "has_docstring": False,
        "has_main": False,
        "lines": 0,
        "functions": [],
        "issues": [],
    }
    
    if not script_path.exists():
        result["issues"].append("File does not exist")
        return result
    
    try:
        content = script_path.read_text(encoding="utf-8")
        result["lines"] = len(content.split("\n"))
        result["has_content"] = len(content.strip()) > 0
        
        if not result["has_content"]:
            result["issues"].append("File is empty")
            return result
        
        # Python ファイルの詳細チェック
        if script_path.suffix == ".py":
            try:
                tree = ast.parse(content)
                
                # Module docstring チェック
                if tree.body and isinstance(tree.body[0], ast.Expr):
                    if isinstance(tree.body[0].value, ast.Constant):
                        result["has_docstring"] = True
                
                # 関数抽出
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        result["functions"].append(node.name)
                        if node.name == "main":
                            result["has_main"] = True
            except SyntaxError as e:
                result["issues"].append(f"Syntax error: {e}")
        
        # Shell スクリプトの main チェック
        elif script_path.suffix == ".sh":
            if "#!/bin/bash" in content or "#!/bin/sh" in content:
                result["has_main"] = True
    
    except Exception as e:
        result["issues"].append(f"Error reading file: {e}")
    
    return result


def check_script_documentation(script_path: Path) -> dict:
    """スクリプトのドキュメント参照をチェック"""
    result = {
        "docs_referenced": [],
        "documented": False,
        "docstring": None,
    }
    
    try:
        content = script_path.read_text(encoding="utf-8")
        
        # ドキュメントとの参照確認
        doc_references = re.findall(r"documents/([a-z_\-]+\.md)", content)
        result["docs_referenced"] = list(set(doc_references))
        
        # Docstring 抽出
        if script_path.suffix == ".py":
            try:
                tree = ast.parse(content)
                if tree.body and isinstance(tree.body[0], ast.Expr):
                    if isinstance(tree.body[0].value, ast.Constant):
                        result["docstring"] = tree.body[0].value.value
                        result["documented"] = True
            except:
                pass
    
    except Exception:
        pass
    
    return result


def check_script_tests(workspace_root: Path, script_path: Path) -> dict:
    """スクリプトのテストカバレッジをチェック"""
    result = {
        "has_test": False,
        "test_file": None,
        "test_referenced": False,
    }
    
    # テストファイル確認
    test_dir = workspace_root / "python" / "tests"
    if test_dir.exists():
        # スクリプト名からテストファイル名推測
        script_name = script_path.stem
        possible_test_files = [
            test_dir / f"test_{script_name}.py",
            test_dir / f"{script_name}_test.py",
        ]
        
        for test_file in possible_test_files:
            if test_file.exists():
                result["has_test"] = True
                result["test_file"] = str(test_file)
                result["test_referenced"] = True
                break
    
    return result


def check_script_dependencies(script_path: Path) -> dict:
    """スクリプトの依存関係をチェック"""
    result = {
        "imports": [],
        "external_deps": [],
        "issues": [],
    }
    
    if script_path.suffix != ".py":
        return result
    
    try:
        content = script_path.read_text(encoding="utf-8")
        tree = ast.parse(content)
        
        # import 抽出
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    result["imports"].append(node.module)
        
        # 外部依存推定（標準ライブラリ以外）
        stdlib = {
            "sys", "os", "re", "json", "pathlib", "argparse",
            "datetime", "collections", "itertools", "functools",
            "typing", "ast", "importlib", "subprocess", "shutil",
        }
        
        for imp in result["imports"]:
            top_level = imp.split(".")[0]
            if top_level not in stdlib and not top_level.startswith("_"):
                result["external_deps"].append(top_level)
    
    except Exception as e:
        result["issues"].append(f"Error parsing dependencies: {e}")
    
    return result


def run(workspace_root: Path = None, verbose: bool = False, **kwargs) -> dict:
    """Phase 3 を実行"""
    if workspace_root is None:
        workspace_root = Path.cwd()
    
    workspace_root = Path(workspace_root)
    
    if verbose:
        print("🔍 Scanning tools and scripts...")
    
    # スクリプト検出
    scripts = find_scripts(workspace_root)
    
    if not scripts:
        if verbose:
            print("⚠️  No scripts found")
        return {"status": "pass", "issues": []}
    
    if verbose:
        print(f"   Found {len(scripts)} scripts")
    
    # チェック実行
    all_issues = []
    script_stats = {
        "total": len(scripts),
        "implemented": 0,
        "documented": 0,
        "tested": 0,
        "with_external_deps": 0,
    }
    
    for script_name, script_info in scripts.items():
        script_path = script_info["path"]
        
        if verbose:
            print(f"   Checking {script_name}...")
        
        # 実装状況
        impl = check_script_implementation(script_path)
        if impl["has_content"]:
            script_stats["implemented"] += 1
        else:
            all_issues.append({
                "severity": "warn",
                "type": "empty_script",
                "script": script_name,
                "message": f"Script '{script_name}' is empty",
            })
        
        # ドキュメント
        docs = check_script_documentation(script_path)
        if docs["documented"]:
            script_stats["documented"] += 1
        
        # テスト
        tests = check_script_tests(workspace_root, script_path)
        if tests["has_test"]:
            script_stats["tested"] += 1
        else:
            if impl["has_content"]:
                all_issues.append({
                    "severity": "warn",
                    "type": "no_test",
                    "script": script_name,
                    "message": f"Script '{script_name}' has no test",
                })
        
        # 依存関係
        deps = check_script_dependencies(script_path)
        if deps["external_deps"]:
            script_stats["with_external_deps"] += 1
            for dep in deps["external_deps"]:
                all_issues.append({
                    "severity": "info",
                    "type": "external_dependency",
                    "script": script_name,
                    "dependency": dep,
                    "message": f"Script '{script_name}' depends on external package: {dep}",
                })
        
        # 実装エラー
        for issue in impl["issues"]:
            all_issues.append({
                "severity": "error",
                "type": "implementation_error",
                "script": script_name,
                "message": f"Implementation error in '{script_name}': {issue}",
            })
    
    # 覆率計算
    coverage = {
        "implementation": int(100 * script_stats["implemented"] / len(scripts)),
        "documentation": int(100 * script_stats["documented"] / len(scripts)),
        "testing": int(100 * script_stats["tested"] / len(scripts)),
    }
    
    error_count = sum(1 for i in all_issues if i["severity"] == "error")
    warn_count = sum(1 for i in all_issues if i["severity"] == "warn")
    
    status = "pass" if error_count == 0 else "error"
    if error_count == 0 and warn_count > 0:
        status = "warn"
    
    if verbose:
        print(f"\n   📊 Coverage: {coverage['implementation']}% impl, {coverage['documentation']}% doc, {coverage['testing']}% test")
    
    return {
        "status": status,
        "issues": [f"{i['severity'].upper()}: {i['message']}" for i in all_issues[:10]],
        "details": {
            "total_scripts": len(scripts),
            "stats": script_stats,
            "coverage": coverage,
            "total_issues": len(all_issues),
            "errors": error_count,
            "warnings": warn_count,
        }
    }


if __name__ == "__main__":
    import sys
    
    workspace = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    result = run(workspace, verbose=True)
    
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Status: {result['status']}")
    print(f"Issues found: {len(result['issues'])}")
    for issue in result['issues']:
        print(f"  - {issue}")
