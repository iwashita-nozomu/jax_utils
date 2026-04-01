"""
Phase 4: Integration Test Checker

統合テスト：
- CLI 実行可能性
- GitHub Actions パイプライン
- Docker 統合
- ワークフロー互換性
"""

import re
import json
from pathlib import Path
from typing import Dict, List


def check_cli_references(workspace_root: Path) -> List[dict]:
    """CLI ドキュメント参照をチェック"""
    issues = []
    
    cli_doc_path = workspace_root / ".github" / "copilot-instructions.md"
    if not cli_doc_path.exists():
        issues.append({
            "severity": "warn",
            "type": "missing_cli_doc",
            "file": "copilot-instructions.md",
            "message": "CLI instructions file not found",
        })
        return issues
    
    try:
        content = cli_doc_path.read_text(encoding="utf-8")
        
        # コマンド参照確認
        if "comprehensive-review" not in content and "comprehensive_review" not in content:
            issues.append({
                "severity": "warn",
                "type": "missing_skill_reference",
                "file": "copilot-instructions.md",
                "message": "Comprehensive Review Skill not referenced in CLI instructions",
            })
        
        # Python パス確認
        if "${PYTHONPATH}" not in content and "$PYTHONPATH" not in content:
            issues.append({
                "severity": "info",
                "type": "no_pythonpath_reference",
                "file": "copilot-instructions.md",
                "message": "PYTHONPATH not configured in CLI instructions",
            })
    
    except Exception as e:
        issues.append({
            "severity": "error",
            "type": "cli_doc_error",
            "message": f"Error reading CLI instructions: {e}",
        })
    
    return issues


def check_github_actions(workspace_root: Path) -> List[dict]:
    """GitHub Actions ワークフロー検証"""
    issues = []
    
    workflows_dir = workspace_root / ".github" / "workflows"
    if not workflows_dir.exists():
        issues.append({
            "severity": "warn",
            "type": "no_workflows",
            "message": "GitHub Actions workflows directory not found",
        })
        return issues
    
    yaml_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
    
    if not yaml_files:
        issues.append({
            "severity": "warn",
            "type": "no_workflow_files",
            "message": "No GitHub Actions workflow files found",
        })
        return issues
    
    for workflow_file in yaml_files:
        try:
            content = workflow_file.read_text(encoding="utf-8")
            
            # Comprehensive Review ワークフロー確認
            if workflow_file.stem == "comprehensive-review" or workflow_file.stem == "comprehensive_review":
                # テスト対象スクリプトの参照確認
                if "run-review.py" not in content:
                    issues.append({
                        "severity": "warn",
                        "type": "missing_script_reference",
                        "workflow": workflow_file.name,
                        "message": f"Workflow '{workflow_file.name}' doesn't reference run-review.py",
                    })
            
            # Docker reference 確認
            if "docker" not in content.lower() and "container" not in content.lower():
                if workflow_file.stem not in ["test", "lint"]:  # 一部のワークフローは Docker 不要
                    pass  # Docker 要件は Skill によって異なる
        
        except Exception as e:
            issues.append({
                "severity": "error",
                "type": "workflow_parse_error",
                "workflow": workflow_file.name,
                "message": f"Error parsing workflow: {e}",
            })
    
    return issues


def check_docker_configuration(workspace_root: Path) -> List[dict]:
    """Docker 設定検証"""
    issues = []
    
    dockerfile_path = workspace_root / "docker" / "Dockerfile"
    requirements_path = workspace_root / "docker" / "requirements.txt"
    
    if not dockerfile_path.exists():
        issues.append({
            "severity": "warn",
            "type": "missing_dockerfile",
            "message": "Dockerfile not found",
        })
        return issues
    
    try:
        dockerfile_content = dockerfile_path.read_text(encoding="utf-8")
        
        # Python 環境確認
        if "python" not in dockerfile_content.lower():
            issues.append({
                "severity": "warn",
                "type": "no_python_in_docker",
                "message": "Python not configured in Dockerfile",
            })
        
        # requirements.txt 参照確認
        if "requirements.txt" not in dockerfile_content:
            if requirements_path.exists():
                issues.append({
                    "severity": "warn",
                    "type": "requirements_not_referenced",
                    "message": "requirements.txt exists but not referenced in Dockerfile",
                })
        
        # PYTHONPATH 設定確認
        if "PYTHONPATH" not in dockerfile_content:
            issues.append({
                "severity": "info",
                "type": "no_pythonpath_in_docker",
                "message": "PYTHONPATH not set in Dockerfile",
            })
    
    except Exception as e:
        issues.append({
            "severity": "error",
            "type": "dockerfile_error",
            "message": f"Error reading Dockerfile: {e}",
        })
    
    # requirements.txt チェック
    if requirements_path.exists():
        try:
            reqs = requirements_path.read_text(encoding="utf-8")
            if len(reqs.strip()) == 0:
                issues.append({
                    "severity": "warn",
                    "type": "empty_requirements",
                    "message": "requirements.txt is empty",
                })
        except Exception as e:
            issues.append({
                "severity": "error",
                "type": "requirements_error",
                "message": f"Error reading requirements.txt: {e}",
            })
    
    return issues


def check_workflow_compatibility(workspace_root: Path) -> List[dict]:
    """ワークフロー互換性検証"""
    issues = []
    
    # pyproject.toml 確認
    pyproject_path = workspace_root / "pyproject.toml"
    if not pyproject_path.exists():
        issues.append({
            "severity": "warn",
            "type": "missing_pyproject",
            "message": "pyproject.toml not found",
        })
    
    # Makefile 確認
    makefile_path = workspace_root / "Makefile"
    if not makefile_path.exists():
        issues.append({
            "severity": "warn",
            "type": "missing_makefile",
            "message": "Makefile not found",
        })
    else:
        try:
            makefile_content = makefile_path.read_text(encoding="utf-8")
            
            # review ターゲット確認
            if "review" not in makefile_content:
                issues.append({
                    "severity": "info",
                    "type": "no_review_target",
                    "message": "Makefile doesn't have 'review' target",
                })
        except Exception as e:
            issues.append({
                "severity": "warn",
                "type": "makefile_error",
                "message": f"Error reading Makefile: {e}",
            })
    
    # skill root README 確認
    skills_readme = workspace_root / ".github" / "skills" / "README.md"
    if not skills_readme.exists():
        issues.append({
            "severity": "warn",
            "type": "missing_skills_hub",
            "message": "Skills hub README not found",
        })
    
    return issues


def check_environment_setup(workspace_root: Path) -> List[dict]:
    """環境設定検証"""
    issues = []
    
    # pyrightconfig.json 確認
    pyrightconfig_path = workspace_root / "pyrightconfig.json"
    if not pyrightconfig_path.exists():
        issues.append({
            "severity": "info",
            "type": "missing_pyrightconfig",
            "message": "pyrightconfig.json not found",
        })
    
    # .python-version 確認
    python_version_path = workspace_root / ".python-version"
    if not python_version_path.exists():
        issues.append({
            "severity": "info",
            "type": "missing_python_version",
            "message": ".python-version not found",
        })
    
    return issues


def run(workspace_root: Path = None, verbose: bool = False, **kwargs) -> dict:
    """Phase 4 を実行"""
    if workspace_root is None:
        workspace_root = Path.cwd()
    
    workspace_root = Path(workspace_root)
    
    if verbose:
        print("🔍 Checking integration...")
    
    all_issues = []
    
    if verbose:
        print("   Checking CLI integration...")
    all_issues.extend(check_cli_references(workspace_root))
    
    if verbose:
        print("   Checking GitHub Actions workflows...")
    all_issues.extend(check_github_actions(workspace_root))
    
    if verbose:
        print("   Checking Docker configuration...")
    all_issues.extend(check_docker_configuration(workspace_root))
    
    if verbose:
        print("   Checking workflow compatibility...")
    all_issues.extend(check_workflow_compatibility(workspace_root))
    
    if verbose:
        print("   Checking environment setup...")
    all_issues.extend(check_environment_setup(workspace_root))
    
    # 結果集約
    error_count = sum(1 for i in all_issues if i["severity"] == "error")
    warn_count = sum(1 for i in all_issues if i["severity"] == "warn")
    
    status = "pass" if error_count == 0 else "error"
    if error_count == 0 and warn_count > 0:
        status = "warn"
    
    if verbose:
        print(f"\n   📊 Results: {error_count} errors, {warn_count} warnings")
    
    return {
        "status": status,
        "issues": [f"{i['severity'].upper()}: {i['message']}" for i in all_issues[:10]],
        "details": {
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
