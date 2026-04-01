"""
Phase 2: Skills Coherence Checker

Skill の構成・依存関係・重複・漏れをチェック：
- Skill メタデータ検証
- 依存関係の循環性
- 機能重複検出
- 実装漏れ検出
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Set


def find_skill_dirs(workspace_root: Path) -> Dict[int, Path]:
    """Skill ディレクトリを検出"""
    skills_root = workspace_root / ".github" / "skills"
    skills = {}
    
    if not skills_root.exists():
        return skills
    
    for skill_dir in skills_root.iterdir():
        if not skill_dir.is_dir():
            continue
        
        # ディレクトリ名から Skill 番号抽出（01-*, 02-*, etc）
        match = re.match(r"(\d+)-", skill_dir.name)
        if match:
            num = int(match.group(1))
            skills[num] = skill_dir
    
    return skills


def extract_skill_metadata(skill_dir: Path) -> dict:
    """Skill の README から メタデータを抽出"""
    readme_path = skill_dir / "README.md"
    
    if not readme_path.exists():
        return {}
    
    try:
        content = readme_path.read_text(encoding="utf-8")
        
        # 基本情報抽出
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        description_match = re.search(r"^##\s+概要\s*\n(.+?)(?:\n##|\Z)", content, re.MULTILINE | re.DOTALL)
        
        # 依存関係抽出
        depends_on = set()
        for match in re.finditer(r"Skill\s+(\d+)", content):
            num = int(match.group(1))
            if num != int(skill_dir.name.split("-")[0]):
                depends_on.add(num)
        
        # ツール参照抽出
        tools = set()
        for match in re.finditer(r"`([a-z_]+\.py)`", content):
            tools.add(match.group(1))
        
        return {
            "number": int(skill_dir.name.split("-")[0]),
            "name": skill_dir.name,
            "title": title_match.group(1) if title_match else "Unknown",
            "description": description_match.group(1).strip()[:100] if description_match else "",
            "depends_on": sorted(list(depends_on)),
            "tools": sorted(list(tools)),
        }
    except Exception as e:
        return {
            "number": int(skill_dir.name.split("-")[0]),
            "error": str(e),
        }


def check_circular_dependencies(skills_metadata: Dict[int, dict]) -> List[dict]:
    """Skill 間の循環依存を検出"""
    issues = []
    
    def has_cycle(current: int, visited: Set[int], path: Set[int], graph: Dict[int, List[int]]) -> bool:
        if current in path:
            return True
        if current in visited:
            return False
        
        visited.add(current)
        path.add(current)
        
        for dep in graph.get(current, []):
            if has_cycle(dep, visited, path, graph):
                return True
        
        path.remove(current)
        return False
    
    # 依存グラフ構築
    graph = {num: meta.get("depends_on", []) for num, meta in skills_metadata.items()}
    
    visited = set()
    for skill_num in skills_metadata:
        if has_cycle(skill_num, visited, set(), graph):
            issues.append({
                "severity": "error",
                "type": "circular_dependency",
                "skill": skill_num,
                "message": f"Circular dependency detected in Skill {skill_num}",
            })
    
    return issues


def check_duplicate_functionality(skills_metadata: Dict[int, dict]) -> List[dict]:
    """機能重複を検出"""
    issues = []
    
    # キーワード集約
    tool_groups = {}
    for skill_num, meta in skills_metadata.items():
        for tool in meta.get("tools", []):
            if tool not in tool_groups:
                tool_groups[tool] = []
            tool_groups[tool].append(skill_num)
    
    # 同じツールを参照する Skill を検出
    for tool, skill_nums in tool_groups.items():
        if len(skill_nums) > 1:
            issues.append({
                "severity": "warn",
                "type": "duplicate_tool_reference",
                "tool": tool,
                "skills": skill_nums,
                "message": f"Tool '{tool}' is referenced by multiple skills: {skill_nums}",
            })
    
    return issues


def check_missing_implementations(workspace_root: Path, skills_metadata: Dict[int, dict]) -> List[dict]:
    """実装漏れを検出"""
    issues = []
    
    # 各 Skill の実装ページ確認
    for skill_num, meta in skills_metadata.items():
        skill_dir = workspace_root / ".github" / "skills" / meta["name"]
        
        # 必須ファイル確認
        required_files = ["README.md"]
        for req_file in required_files:
            if not (skill_dir / req_file).exists():
                issues.append({
                    "severity": "error",
                    "type": "missing_file",
                    "skill": skill_num,
                    "file": req_file,
                    "message": f"Required file missing in Skill {skill_num}: {req_file}",
                })
        
        # checkers/ 実装確認
        checkers_dir = skill_dir / "checkers"
        if checkers_dir.exists():
            py_files = list(checkers_dir.glob("*.py"))
            if not py_files:
                issues.append({
                    "severity": "warn",
                    "type": "empty_checkers",
                    "skill": skill_num,
                    "message": f"Skill {skill_num} has empty checkers/ directory",
                })
    
    return issues


def check_metadata_completeness(skills_metadata: Dict[int, dict]) -> List[dict]:
    """メタデータの完全性をチェック"""
    issues = []
    
    for skill_num, meta in skills_metadata.items():
        if "error" in meta:
            issues.append({
                "severity": "warn",
                "type": "metadata_error",
                "skill": skill_num,
                "message": f"Error reading Skill {skill_num} metadata: {meta['error']}",
            })
            continue
        
        # 必須フィールド確認
        if not meta.get("title") or meta["title"] == "Unknown":
            issues.append({
                "severity": "warn",
                "type": "missing_title",
                "skill": skill_num,
                "message": f"Skill {skill_num} missing or unclear title",
            })
        
        if not meta.get("description"):
            issues.append({
                "severity": "warn",
                "type": "missing_description",
                "skill": skill_num,
                "message": f"Skill {skill_num} missing description",
            })
    
    return issues


def check_skill_sequencing(skills_metadata: Dict[int, dict]) -> List[dict]:
    """Skill の依存順序が適切か検証"""
    issues = []
    
    # 前提: Skill 番号順に依存関係が構築されるべき
    for skill_num, meta in skills_metadata.items():
        for dep in meta.get("depends_on", []):
            if dep > skill_num:
                issues.append({
                    "severity": "warn",
                    "type": "invalid_dependency_order",
                    "skill": skill_num,
                    "depends_on": dep,
                    "message": f"Skill {skill_num} depends on later Skill {dep} (should depend on earlier skills)",
                })
    
    return issues


def run(workspace_root: Path = None, verbose: bool = False, **kwargs) -> dict:
    """Phase 2 を実行"""
    if workspace_root is None:
        workspace_root = Path.cwd()
    
    workspace_root = Path(workspace_root)
    
    if verbose:
        print("🔍 Scanning Skill definitions...")
    
    # Skill ディレクトリ検出
    skills = find_skill_dirs(workspace_root)
    
    if not skills:
        if verbose:
            print("⚠️  No Skill directories found")
        return {"status": "pass", "issues": []}
    
    if verbose:
        print(f"   Found {len(skills)} skills: {sorted(skills.keys())}")
    
    # メタデータ抽出
    skills_metadata = {}
    for skill_num, skill_dir in skills.items():
        meta = extract_skill_metadata(skill_dir)
        skills_metadata[skill_num] = meta
        if verbose:
            print(f"   Skill {skill_num}: {meta.get('title', 'Unknown')}")
    
    # チェック実行
    all_issues = []
    
    if verbose:
        print("   Checking for circular dependencies...")
    all_issues.extend(check_circular_dependencies(skills_metadata))
    
    if verbose:
        print("   Checking for duplicate functionality...")
    all_issues.extend(check_duplicate_functionality(skills_metadata))
    
    if verbose:
        print("   Checking for missing implementations...")
    all_issues.extend(check_missing_implementations(workspace_root, skills_metadata))
    
    if verbose:
        print("   Checking metadata completeness...")
    all_issues.extend(check_metadata_completeness(skills_metadata))
    
    if verbose:
        print("   Checking Skill sequencing...")
    all_issues.extend(check_skill_sequencing(skills_metadata))
    
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
            "total_skills": len(skills),
            "total_issues": len(all_issues),
            "errors": error_count,
            "warnings": warn_count,
            "skills": sorted(skills.keys()),
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
