"""
Phase 1: Documentation Review Checker

ドキュメント品質をチェック：
- broken link 検出
- 用語一貫性
- 循環参照
- メタデータ検証
"""

import re
from pathlib import Path
from functools import lru_cache


def find_all_md_files(workspace_root: Path) -> list[Path]:
    """Markdown ファイル一覧を取得"""
    return list(workspace_root.glob("**/*.md"))


def extract_links(content: str) -> list[tuple[str, int]]:
    """Markdown 内のリンク [text](link) を抽出 (行番号付き)"""
    links = []
    for i, line in enumerate(content.split("\n"), 1):
        # [text](link) 形式のリンク抽出
        for match in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', line):
            link = match.group(2)
            links.append((link, i))
    return links


def check_broken_links(workspace_root: Path, md_files: list[Path], verbose: bool = False) -> list[dict]:
    """broken link を検出"""
    issues = []
    
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
            links = extract_links(content)
            
            for link, line_no in links:
                # ローカルリンクの場合のみチェック
                if link.startswith("http"):
                    continue
                
                # アンカーを分離
                if "#" in link:
                    file_path, anchor = link.split("#", 1)
                else:
                    file_path = link
                    anchor = None
                
                # ファイル存在確認
                if file_path:
                    target = (md_file.parent / file_path).resolve()
                    if not target.exists():
                        issues.append({
                            "severity": "error",
                            "type": "broken_link",
                            "file": str(md_file.relative_to(workspace_root)),
                            "line": line_no,
                            "link": link,
                            "message": f"Link target not found: {link}",
                        })
        except Exception as e:
            if verbose:
                print(f"  Error reading {md_file}: {e}")
    
    return issues


def check_terminology(workspace_root: Path, md_files: list[Path]) -> list[dict]:
    """用語の一貫性をチェック"""
    issues = []
    
    # 用語チェック定義
    term_patterns = [
        (r"\bSkill\b", r"\bskill\b", "Skill"),  # Skill 表記ゆれ
        (r"\bGitHub Copilot\b", r"\bGithub Copilot\b|copilot\b", "GitHub Copilot"),
        (r"\bVirtual Environment\b", r"\bvirtual environment\b", "Virtual Environment"),
    ]
    
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
            lines = content.split("\n")
            
            for pattern_correct, pattern_incorrect, term_name in term_patterns:
                for i, line in enumerate(lines, 1):
                    if re.search(pattern_incorrect, line):
                        issues.append({
                            "severity": "warn",
                            "type": "terminology",
                            "file": str(md_file.relative_to(workspace_root)),
                            "line": i,
                            "term": term_name,
                            "message": f"Inconsistent terminology: should be '{term_name}'",
                        })
        except Exception:
            pass
    
    return issues


def check_circular_refs(workspace_root: Path, md_files: list[Path]) -> list[dict]:
    """循環参照をチェック"""
    issues = []
    
    # グラフ構築: file -> [referenced files]
    graph = {}
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
            links = extract_links(content)
            
            referenced = set()
            for link, _ in links:
                if link.startswith("http"):
                    continue
                if "#" in link:
                    file_path, _ = link.split("#", 1)
                else:
                    file_path = link
                
                if file_path:
                    target = (md_file.parent / file_path).resolve()
                    if target in [f.resolve() for f in md_files]:
                        referenced.add(target)
            
            graph[md_file.resolve()] = referenced
        except Exception:
            pass
    
    # 簡易的な循環参照検出（DFS）
    def has_cycle(start, current, visited, path):
        if current in path:
            return True
        if current in visited:
            return False
        
        visited.add(current)
        path.add(current)
        
        for neighbor in graph.get(current, []):
            if has_cycle(start, neighbor, visited, path):
                return True
        
        path.remove(current)
        return False
    
    for start_file in graph:
        visited = set()
        if has_cycle(start_file, start_file, visited, set()):
            issues.append({
                "severity": "warn",
                "type": "circular_reference",
                "file": str(start_file.relative_to(workspace_root)),
                "message": f"Possible circular reference detected",
            })
    
    return issues


def check_metadata(workspace_root: Path, md_files: list[Path]) -> list[dict]:
    """メタデータ（YAML frontmatter）を検証"""
    issues = []
    
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
            lines = content.split("\n")
            
            # YAML frontmatter チェック
            if lines and lines[0].strip() == "---":
                # 開始タグあり - 終了タグ確認
                if not any(line.strip() == "---" for line in lines[1:]):
                    issues.append({
                        "severity": "warn",
                        "type": "metadata",
                        "file": str(md_file.relative_to(workspace_root)),
                        "line": 1,
                        "message": "YAML frontmatter not properly closed",
                    })
        except Exception:
            pass
    
    return issues


def run(workspace_root: Path = None, verbose: bool = False, **kwargs) -> dict:
    """Phase 1 を実行"""
    if workspace_root is None:
        workspace_root = Path.cwd()
    
    workspace_root = Path(workspace_root)
    
    if verbose:
        print("🔍 Scanning documentation files...")
    
    # ドキュメント対象ディレクトリ
    doc_dirs = [workspace_root / "documents", workspace_root / "notes"]
    md_files = []
    for doc_dir in doc_dirs:
        if doc_dir.exists():
            md_files.extend(doc_dir.glob("**/*.md"))
    
    if not md_files:
        if verbose:
            print("⚠️  No markdown files found")
        return {"status": "pass", "issues": []}
    
    if verbose:
        print(f"   Found {len(md_files)} markdown files")
    
    # チェック実行
    all_issues = []
    
    if verbose:
        print("   Checking for broken links...")
    all_issues.extend(check_broken_links(workspace_root, md_files, verbose))
    
    if verbose:
        print("   Checking terminology consistency...")
    all_issues.extend(check_terminology(workspace_root, md_files))
    
    if verbose:
        print("   Checking for circular references...")
    all_issues.extend(check_circular_refs(workspace_root, md_files))
    
    if verbose:
        print("   Checking metadata...")
    all_issues.extend(check_metadata(workspace_root, md_files))
    
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
        "issues": [f"{i['severity'].upper()}: {i['message']} ({i['file']}:{i.get('line', '?')})" for i in all_issues[:10]],
        "details": {
            "total_files": len(md_files),
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
