#!/usr/bin/env python3
"""
Skill 15: Docker Environment Validation

Docker 環境の整合性・依存関係・セキュリティを検証。
"""

import sys
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime

WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent


def parse_requirements():
    """requirements.txt を解析"""
    print("📄 Parsing requirements.txt...")
    
    req_file = WORKSPACE_ROOT / "docker" / "requirements.txt"
    requirements = {}
    
    if req_file.exists():
        with open(req_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    match = re.match(r"^([a-zA-Z0-9-]+).*==(.+)$", line)
                    if match:
                        requirements[match.group(1)] = match.group(2)
    
    print(f"   Found {len(requirements)} packages")
    return requirements


def parse_dockerfile():
    """Dockerfile を解析"""
    print("📄 Parsing Dockerfile...")
    
    dockerfile = WORKSPACE_ROOT / "docker" / "Dockerfile"
    packages_in_docker = {}
    
    if dockerfile.exists():
        with open(dockerfile) as f:
            content = f.read()
            
            # RUN pip install を抽出
            for match in re.finditer(r"RUN pip install (.+)", content):
                line = match.group(1)
                # 簡易的な解析
                for pkg in line.split():
                    if "==" in pkg:
                        name, version = pkg.split("==")
                        packages_in_docker[name] = version
    
    print(f"   Found {len(packages_in_docker)} packages")
    return packages_in_docker


def validate_sync(requirements, dockerfile_packages):
    """依存関係同期確認"""
    print("🔄 Validating dependency sync...")
    
    issues = []
    
    # requirements.txt にあるが Dockerfile にない
    for pkg, version in requirements.items():
        if pkg not in dockerfile_packages:
            issues.append({
                "type": "missing_in_dockerfile",
                "package": pkg,
                "version": version,
            })
    
    # Dockerfile にあるが requirements.txt にない
    for pkg, version in dockerfile_packages.items():
        if pkg not in requirements:
            issues.append({
                "type": "missing_in_requirements",
                "package": pkg,
                "version": version,
            })
    
    # バージョンが異なる
    for pkg in requirements:
        if pkg in dockerfile_packages:
            if requirements[pkg] != dockerfile_packages[pkg]:
                issues.append({
                    "type": "version_mismatch",
                    "package": pkg,
                    "requirements_version": requirements[pkg],
                    "dockerfile_version": dockerfile_packages[pkg],
                })
    
    print(f"   Found {len(issues)} sync issues")
    return issues


def test_docker_build():
    """Docker build テスト"""
    print("🐳 Testing Docker build...")
    
    dockerfile = WORKSPACE_ROOT / "docker" / "Dockerfile"
    
    try:
        result = subprocess.run(
            ["docker", "build", "-f", str(dockerfile), "-t", "workspace-test:latest", str(WORKSPACE_ROOT / "docker")],
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        if result.returncode == 0:
            print("   ✅ Docker build successful")
            return True
        else:
            print(f"   ❌ Docker build failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"   ⚠️  Docker build test skipped: {e}")
        return None


def generate_report(sync_issues, build_success):
    """レポート生成"""
    print("\n📊 Generating report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "dependencies": {
            "sync_issues": sync_issues,
            "build_result": "success" if build_success else ("failed" if build_success is None else "skipped"),
        },
        "summary": {
            "total_issues": len(sync_issues),
            "critical_issues": len([x for x in sync_issues if x.get("type") in ["version_mismatch", "missing_in_dockerfile"]]),
        }
    }
    
    return report


def main():
    """メイン実行"""
    print("=" * 60)
    print("Skill 15: Docker Environment Validation")
    print("=" * 60)
    
    requirements = parse_requirements()
    dockerfile_packages = parse_dockerfile()
    sync_issues = validate_sync(requirements, dockerfile_packages)
    build_success = test_docker_build()
    
    report = generate_report(sync_issues, build_success)
    
    # レポート保存
    report_dir = WORKSPACE_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    report_file = report_dir / f"docker-validate-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Docker validation complete")
    print(f"   Total issues: {report['summary']['total_issues']}")
    print(f"   Critical issues: {report['summary']['critical_issues']}")
    print(f"\n📄 Report saved: {report_file}")
    
    return 1 if report["summary"]["critical_issues"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
