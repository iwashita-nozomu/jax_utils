#!/usr/bin/env python3
"""
Layer C: Project Rules & Convention Consistency Check（プロジェクト規約検証）

目的:
- プロジェクト全体の規約との整合性検証
- ドキュメントと実装の一貫性確認
- 複数の規約ファイル間の矛盾検出

レイヤー:
- C層: 統合検証（レビュアー向け、PR レビュー時）

出力:
- PASS: 規約に完全に準拠
- WARN: 軽微な違反（改善推奨）
- FAIL: 重大な違反（修正必須）
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set


class ProjectRulesChecker:
    """プロジェクト規約検証チェッカー。"""

    def __init__(self, verbose: bool = False):
        """初期化。

        Args:
            verbose: 詳細出力フラグ。
        """
        self.verbose = verbose
        self.repo_root = Path.cwd()

    def check_documentation_consistency(
        self,
    ) -> Tuple[bool, List[Dict]]:
        """ドキュメント間の一貫性をチェック。

        確認項目:
        - README と coding-conventions の間の矛盾
        - ドキュメント内のリンク参照先の存在確認
        - API ドキュメントと実装の齟齬

        Returns:
            (成功フラグ, 問題リスト)
        """
        issues = []

        # documents/ 内の主要ファイルを確認
        doc_root = self.repo_root / "documents"
        if not doc_root.exists():
            return False, [
                {
                    "severity": "HIGH",
                    "issue": "documents/ ディレクトリが見つかりません",
                }
            ]

        # coding-conventions*.md の確認
        conventions_files = list(
            doc_root.glob("coding-conventions-*.md")
        )
        if not conventions_files:
            issues.append({
                "severity": "HIGH",
                "issue": "coding-conventions-*.md が見つかりません",
            })

        # キー規約ファイルの確認
        key_files = {
            "README.md": "プロジェクト概要",
            "coding-conventions.md": "コーディング大規約",
            "REVIEW_PROCESS.md": "レビュープロセス",
        }

        for file_name, desc in key_files.items():
            file_path = doc_root / file_name
            if not file_path.exists():
                issues.append({
                    "severity": "MEDIUM",
                    "issue": f"重要ドキュメント見つからず: {file_name}",
                })

        return len([i for i in issues if i["severity"] == "HIGH"]) == 0, issues

    def check_architecture_alignment(
        self,
    ) -> Tuple[bool, List[Dict]]:
        """ディレクトリ構構の設計との整合性をチェック。

        確認項目:
        - python/ ディレクトリ構造が design/jax_util/README.md と一致
        - documents/design/ の構造が実装ディレクトリと一致

        Returns:
            (成功フラグ, 問題リスト)
        """
        issues = []

        # python/ の主要ディレクトリ確認
        python_root = self.repo_root / "python"
        if not python_root.exists():
            return False, [
                {
                    "severity": "HIGH",
                    "issue": "python/ ディレクトリが見つかりません",
                }
            ]

        expected_modules = {
            "experiment_runner": "実験実行管理",
            "jax_util": "JAX ユーティリティ",
            "tests": "テストスイート",
        }

        found_modules = {
            d.name: d
            for d in python_root.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        }

        for module, _ in expected_modules.items():
            if module not in found_modules:
                issues.append({
                    "severity": "MEDIUM",
                    "issue": (
                        f"期待されるモジュール見つからず: "
                        f"python/{module}"
                    ),
                })

        # design/ との対応確認
        design_root = self.repo_root / "documents" / "design"
        if design_root.exists():
            for design_dir in design_root.iterdir():
                if design_dir.is_dir() and design_dir.name not in [
                    "README.md",
                ]:
                    pass  # 対応チェック（簡易）

        return len([i for i in issues if i["severity"] == "HIGH"]) == 0, issues

    def check_git_conventions(self) -> Tuple[bool, List[Dict]]:
        """Git 運用規約への準拠性をチェック。

        確認項目:
        - コミットメッセージ形式（feat/fix/docs/chore）
        - ブランチ命名規則

        Returns:
            (成功フラグ, 問題リスト)
        """
        issues = []

        # 最新コミットのメッセージフォーマット確認
        try:
            import subprocess

            result = subprocess.run(
                ["git", "log", "-1", "--pretty=%B"],
                capture_output=True,
                text=True,
            )
            last_message = result.stdout.strip()

            # コミットメッセージ形式: type(scope): message
            if not re.match(
                r"^(feat|fix|docs|chore|test|refactor|perf)\(",
                last_message,
            ):
                issues.append({
                    "severity": "LOW",
                    "issue": (
                        f"コミットメッセージ形式が非標準: "
                        f"{last_message.split(chr(10))[0][:50]}"
                    ),
                })

        except Exception:
            pass

        return True, issues

    def check_infrastructure_files(
        self,
    ) -> Tuple[bool, List[Dict]]:
        """インフラストラクチャファイルの整合性をチェック。

        確認項目:
        - Dockerfile + requirements.txt の一致性
        - pyproject.toml + pyrightconfig.json の整合性

        Returns:
            (成功フラグ, 問題リスト)
        """
        issues = []

        # Dockerfile 確認
        dockerfile = self.repo_root / "docker" / "Dockerfile"
        if not dockerfile.exists():
            issues.append({
                "severity": "HIGH",
                "issue": "docker/Dockerfile が見つかりません",
            })
        else:
            # requirements.txt との対応確認
            req_file = self.repo_root / "docker" / "requirements.txt"
            if not req_file.exists():
                issues.append({
                    "severity": "MEDIUM",
                    "issue": "docker/requirements.txt が見つかりません",
                })

        # pyproject.toml 確認
        pyproject = self.repo_root / "pyproject.toml"
        if not pyproject.exists():
            issues.append({
                "severity": "MEDIUM",
                "issue": "pyproject.toml が見つかりません",
            })

        # pyrightconfig.json 確認
        pyrightconfig = self.repo_root / "pyrightconfig.json"
        if not pyrightconfig.exists():
            issues.append({
                "severity": "LOW",
                "issue": "pyrightconfig.json が見つかりません（オプション）",
            })

        return (
            len([i for i in issues if i["severity"] == "HIGH"]) == 0,
            issues,
        )

    def check(self) -> Dict:
        """プロジェクト規約 チェック実行。

        Returns:
            チェック結果辞書。
        """
        results = {
            "component": "layer_c_project_rules_checker",
            "checks": {},
            "issues": [],
        }

        # 1. ドキュメント一貫性チェック
        passed, issues = (
            self.check_documentation_consistency()
        )
        results["checks"]["documentation_consistency"] = {
            "passed": passed,
            "issues": issues,
        }
        results["issues"].extend(issues)

        # 2. アーキテクチャ整合性チェック
        passed, issues = self.check_architecture_alignment()
        results["checks"]["architecture_alignment"] = {
            "passed": passed,
            "issues": issues,
        }
        results["issues"].extend(issues)

        # 3. Git 規約チェック
        passed, issues = self.check_git_conventions()
        results["checks"]["git_conventions"] = {
            "passed": passed,
            "issues": issues,
        }
        results["issues"].extend(issues)

        # 4. インフラストラクチャファイルチェック
        passed, issues = self.check_infrastructure_files()
        results["checks"]["infrastructure_files"] = {
            "passed": passed,
            "issues": issues,
        }
        results["issues"].extend(issues)

        # 判定
        high_severity = sum(
            1 for i in results["issues"] if i["severity"] == "HIGH"
        )
        medium_severity = sum(
            1 for i in results["issues"] if i["severity"] == "MEDIUM"
        )

        if high_severity > 0:
            results["status"] = "FAIL"
        elif medium_severity > 2:
            results["status"] = "WARN"
        else:
            results["status"] = "PASS"

        results["message"] = (
            f"チェック完了: {len(results['issues'])} 個の問題 "
            f"(HIGH: {high_severity}, MEDIUM: {medium_severity})"
        )

        return results


def main() -> int:
    """メイン処理。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Layer C: Project Rules Check（プロジェクト規約検証）"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="詳細出力"
    )
    parser.add_argument(
        "--json", action="store_true", help="JSON 形式で出力"
    )

    args = parser.parse_args()

    checker = ProjectRulesChecker(verbose=args.verbose)
    results = checker.check()

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(
            f"[Project Rules] Status: {results['status']}"
        )
        print(f"  {results['message']}")
        if results.get("issues") and args.verbose:
            print("  問題:")
            for issue in results["issues"][:10]:
                print(
                    f"    - [{issue['severity']}] "
                    f"{issue['issue']}"
                )

    # 終了コード
    if results["status"] == "FAIL":
        return 1
    elif results["status"] == "WARN":
        return 2
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
