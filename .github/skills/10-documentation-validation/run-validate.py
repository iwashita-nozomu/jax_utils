#!/usr/bin/env python3
"""
Skill 10: Documentation Validation

ドキュメント品質の総合検証。
"""

import sys
import re
from pathlib import Path
from datetime import datetime
import json

WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent


def validate_links(doc_dir):
    """リンク検証"""
    print("🔍 Validating links...")
    issues = []
    
    for md_file in doc_dir.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        
        # markdown リンク抽出
        links = re.findall(r'\[([^\]]+)\]\(([^\)]+)\)', content)
        
        for text, url in links:
            if not url.startswith(("http", "#", "/")):
                # 相対パス
                target = (md_file.parent / url.split("#")[0]).resolve()
                if not target.exists() and url.split("#")[0]:  # # のみはスキップ
                    issues.append({
                        "file": str(md_file),
                        "type": "broken_link",
                        "text": text,
                        "url": url,
                    })
    
    print(f"   Found {len(issues)} broken links")
    return issues


def validate_format(doc_dir):
    """形式チェック"""
    print("🔍 Checking format...")
    issues = []
    
    for md_file in doc_dir.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        
        # 基本的な形式チェック
        if not re.search(r'^#\s+\w+', content, re.MULTILINE):
            issues.append({
                "file": str(md_file),
                "type": "missing_title",
            })
    
    print(f"   Found {len(issues)} format issues")
    return issues


def validate_terminology(doc_dir):
    """用語統一チェック"""
    print("🔍 Checking terminology...")
    issues = []
    
    # 用語辞書（例）
    terminology = {
        "machine learning": "機械学習",
        "deep learning": "深層学習",
    }
    
    for md_file in doc_dir.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        
        for en_term, ja_term in terminology.items():
            if en_term.lower() in content.lower():
                issues.append({
                    "file": str(md_file),
                    "type": "inconsistent_terminology",
                    "term": en_term,
                    "suggestion": ja_term,
                })
    
    print(f"   Found {len(issues)} terminology issues")
    return issues


def generate_report(link_issues, format_issues, term_issues):
    """レポート生成"""
    print("\n📊 Generating report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "issues": {
            "links": link_issues,
            "format": format_issues,
            "terminology": term_issues,
        },
        "summary": {
            "total_issues": len(link_issues) + len(format_issues) + len(term_issues),
            "link_issues": len(link_issues),
            "format_issues": len(format_issues),
            "terminology_issues": len(term_issues),
        }
    }
    
    print(f"✅ Documentation validation complete")
    print(f"   Total issues: {report['summary']['total_issues']}")
    
    return report


def main():
    """メイン実行"""
    print("=" * 60)
    print("Skill 10: Documentation Validation")
    print("=" * 60)
    
    doc_dir = WORKSPACE_ROOT / "documents"
    
    link_issues = validate_links(doc_dir)
    format_issues = validate_format(doc_dir)
    term_issues = validate_terminology(doc_dir)
    
    report = generate_report(link_issues, format_issues, term_issues)
    
    # レポート保存
    report_dir = WORKSPACE_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    report_file = report_dir / f"doc-validate-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Report saved: {report_file}")
    
    return 0 if report["summary"]["total_issues"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
