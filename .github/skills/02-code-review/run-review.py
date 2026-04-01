#!/usr/bin/env python3
"""
Skill 2: Code Review — コードレビュー統合スキル

目的:
- A層: 基礎検証（型, スタイル, Docstring, テスト）
- B層: 深度検証（アーキテクチャ, テストデザイン）
- C層: 統合検証（プロジェクト規約, 三点セット）

レイヤー別実行:
- CLI: python .github/skills/02-code-review/run-review.py --phase A
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


class SkillCodeReview:
    """Code Review スキル統合。"""

    def __init__(self, verbose: bool = False):
        """初期化。

        Args:
            verbose: 詳細出力フラグ。
        """
        self.verbose = verbose
        self.skill_dir = Path(".github/skills/02-code-review")

    def run_layer_c(self) -> Dict:
        """Layer C: 統合検証を実行。

        対象:
        - プロジェクト規約検証
        - Doc-Test Triplet 検証

        Returns:
            Layer C 統合結果。
        """
        results = {
            "layer": "C",
            "phase": "統合検証",
            "components": {},
            "overall_status": "UNKNOWN",
        }

        # 1. Project Rules Checker
        rules_cmd = [
            "python3",
            str(self.skill_dir / "layer_c_project_rules.py"),
            "--json",
        ]
        try:
            proc = subprocess.run(
                rules_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            rules_result = json.loads(proc.stdout)
            results["components"]["project_rules"] = rules_result
        except Exception as e:
            results["components"]["project_rules"] = {
                "status": "ERROR",
                "error": str(e),
            }

        # 2. Doc-Test Triplet Checker
        triplet_cmd = [
            "python3",
            str(self.skill_dir / "check_doc_test_triplet.py"),
            "--target-dir",
            "python",
            "--json",
        ]
        try:
            proc = subprocess.run(
                triplet_cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            triplet_result = json.loads(proc.stdout)
            results["components"]["doc_test_triplet"] = triplet_result
        except Exception as e:
            results["components"]["doc_test_triplet"] = {
                "status": "ERROR",
                "error": str(e),
            }

        # 統合判定
        statuses = [
            r.get("status")
            for r in results["components"].values()
        ]
        fail_count = sum(1 for s in statuses if s == "FAIL")
        warn_count = sum(1 for s in statuses if s == "WARN")

        if fail_count >= 1:
            results["overall_status"] = "FAIL"
        elif warn_count >= 1:
            results["overall_status"] = "WARN"
        else:
            results["overall_status"] = "PASS"

        return results

    def run_layer_b(self, target_dir: str = "python") -> Dict:
        """Layer B: 深度検証を実行。

        対象:
        - テストアーキテクチャ検証
        - スタイル・ベストプラクティス検証

        Args:
            target_dir: 対象ディレクトリ。

        Returns:
            Layer B 統合結果。
        """
        results = {
            "layer": "B",
            "phase": "深度検証",
            "components": {},
            "overall_status": "UNKNOWN",
        }

        # 1. Test Architecture Checker
        arch_cmd = [
            "python3",
            str(self.skill_dir / "layer_b_test_architecture.py"),
            "--test-dir",
            "python/tests",
            "--json",
        ]
        try:
            proc = subprocess.run(
                arch_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            arch_result = json.loads(proc.stdout)
            results["components"]["test_architecture"] = arch_result
        except Exception as e:
            results["components"]["test_architecture"] = {
                "status": "ERROR",
                "error": str(e),
            }

        # 2. Style & Best Practices Checker
        style_cmd = [
            "python3",
            str(self.skill_dir / "layer_b_style_best_practices.py"),
            "--target-dir",
            target_dir,
            "--json",
        ]
        try:
            proc = subprocess.run(
                style_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            style_result = json.loads(proc.stdout)
            results["components"]["style_best_practices"] = style_result
        except Exception as e:
            results["components"]["style_best_practices"] = {
                "status": "ERROR",
                "error": str(e),
            }

        # 統合判定
        statuses = [
            r.get("status")
            for r in results["components"].values()
        ]
        fail_count = sum(1 for s in statuses if s == "FAIL")
        warn_count = sum(1 for s in statuses if s == "WARN")
        na_count = sum(1 for s in statuses if s == "N/A")

        if fail_count >= 1:
            results["overall_status"] = "FAIL"
        elif warn_count >= 1:
            results["overall_status"] = "WARN"
        elif na_count == 2:
            results["overall_status"] = "N/A"
        else:
            results["overall_status"] = "PASS"

        return results

    def run_layer_a(self, target_dir: str = "python") -> Dict:
        """Layer A: 基礎検証を実行。

        対象:
        - 型チェック (pyright/mypy)
        - スタイルチェック (ruff/black)
        - Docstring チェック
        - テストカバレッジ

        Args:
            target_dir: 対象ディレクトリ。

        Returns:
            Layer A 統合結果。
        """
        results = {
            "layer": "A",
            "phase": "基礎検証",
            "components": {},
            "overall_status": "UNKNOWN",
        }

        # 1. Type Checker
        type_cmd = [
            "python3",
            str(self.skill_dir / "layer_a_type_checker.py"),
            "--target-dir",
            target_dir,
            "--json",
        ]
        try:
            proc = subprocess.run(
                type_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            type_result = json.loads(proc.stdout)
            results["components"]["type_check"] = type_result
        except Exception as e:
            results["components"]["type_check"] = {
                "status": "ERROR",
                "error": str(e),
            }

        # 2. Linting Checker
        lint_cmd = [
            "python3",
            str(self.skill_dir / "layer_a_linting_checker.py"),
            "--target-dir",
            target_dir,
            "--json",
        ]
        try:
            proc = subprocess.run(
                lint_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            lint_result = json.loads(proc.stdout)
            results["components"]["linting"] = lint_result
        except Exception as e:
            results["components"]["linting"] = {
                "status": "ERROR",
                "error": str(e),
            }

        # 3. Docstring Checker
        doc_cmd = [
            "python3",
            str(self.skill_dir / "layer_a_docstring_checker.py"),
            "--target-dir",
            target_dir,
            "--json",
        ]
        try:
            proc = subprocess.run(
                doc_cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            doc_result = json.loads(proc.stdout)
            results["components"]["docstring"] = doc_result
        except Exception as e:
            results["components"]["docstring"] = {
                "status": "ERROR",
                "error": str(e),
            }

        # 4. Test Coverage Checker
        cov_cmd = [
            "python3",
            str(self.skill_dir / "layer_a_test_coverage_checker.py"),
            "--json",
        ]
        try:
            proc = subprocess.run(
                cov_cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            cov_result = json.loads(proc.stdout)
            results["components"]["test_coverage"] = cov_result
        except Exception as e:
            results["components"]["test_coverage"] = {
                "status": "ERROR",
                "error": str(e),
            }

        # 統合判定
        statuses = [
            r.get("status")
            for r in results["components"].values()
        ]
        fail_count = sum(1 for s in statuses if s == "FAIL")
        warn_count = sum(1 for s in statuses if s == "WARN")
        na_count = sum(1 for s in statuses if s == "N/A")

        if fail_count >= 2:
            results["overall_status"] = "FAIL"
        elif fail_count >= 1 or warn_count >= 2:
            results["overall_status"] = "WARN"
        elif na_count == 4:
            results["overall_status"] = "N/A"
        else:
            results["overall_status"] = "PASS"

        return results

    def print_summary(self, results: Dict) -> None:
        """結果をコンソールに出力。

        Args:
            results: チェック結果。
        """
        print("\n" + "=" * 70)
        print(f"Skill 2: Code Review — {results.get('phase', '')}")
        print("=" * 70)

        layer = results.get("layer", "?")
        overall = results.get("overall_status", "?")

        status_symbol = {
            "PASS": "✅",
            "WARN": "⚠️",
            "FAIL": "❌",
            "N/A": "⏭️",
        }

        print(
            f"\n{status_symbol.get(overall, '?')} "
            f"Layer {layer} 統合判定: {overall}"
        )

        print("\n📋 各コンポーネント結果:")
        for name, result in results.get("components", {}).items():
            comp_status = result.get("status", "?")
            symbol = status_symbol.get(comp_status, "?")
            # メッセージを取得（複数のキーをサポート）
            msg = (
                result.get('message', '')
                or result.get('pyright_message', '')
                or result.get('ruff_message', '')
                or "No message"
            )[:60]
            print(f"  {symbol} {name}: {msg}")

        print("\n" + "=" * 70)

    def check(
        self, phase: str = "A", target_dir: str = "python"
    ) -> Dict:
        """Code Review チェック実行。

        Args:
            phase: 実行フェーズ ("A", "B", "C", "all")
            target_dir: 対象ディレクトリ。

        Returns:
            チェック結果。
        """
        results = {
            "skill": "code-review",
            "phases": {},
        }

        if phase in ["A", "all"]:
            results["phases"]["A"] = self.run_layer_a(target_dir)

        if phase in ["B", "all"]:
            results["phases"]["B"] = self.run_layer_b(target_dir)

        if phase in ["C", "all"]:
            results["phases"]["C"] = self.run_layer_c()

        return results


def main() -> int:
    """メイン処理。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Skill 2: Code Review — コードレビュー統合スキル"
    )
    parser.add_argument(
        "--phase",
        choices=["A", "B", "C", "all"],
        default="A",
        help="実行フェーズ（デフォルト: A）",
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

    skill = SkillCodeReview(verbose=args.verbose)
    results = skill.check(phase=args.phase, target_dir=args.target_dir)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for phase, phase_results in results.get("phases", {}).items():
            if (
                phase_results.get("status")
                != "NOT_IMPLEMENTED"
            ):
                skill.print_summary(phase_results)

    # 終了コード
    fail_count = sum(
        1
        for p in results.get("phases", {}).values()
        if p.get("overall_status") == "FAIL"
    )

    if fail_count > 0:
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
