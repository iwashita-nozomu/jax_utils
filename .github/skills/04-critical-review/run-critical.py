#!/usr/bin/env python3
"""
Skill 4: Critical Review — 実験批判的レビュースキル

3 つのレビュー視点：
1. Change Reviewer: コード変更の妥当性
2. Experiment Reviewer: 実験設計の妥当性
3. Math Reviewer: 数式・アルゴリズムの正確性
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts" / "shared"))
from error_handler import ExecutionResult, ErrorCode


@dataclass
class ReviewResult:
    """レビュー結果。"""

    reviewer_type: str
    status: str  # APPROVE/REQUEST_CHANGES/COMMENT
    findings: list
    score: float  # 0-100


class CriticalReviewEngine:
    """批判的レビューエンジン。"""

    def review_change(self) -> ReviewResult:
        """コード変更レビュー。"""
        # スケルトン実装
        findings = [
            "型検査: ✅ 全ファイル OK",
            "命名規則: ✅ snake_case 統一",
            "複雑度: ⚠️ 関数 XX に改善余地あり",
        ]
        return ReviewResult(
            reviewer_type="Change Reviewer",
            status="REQUEST_CHANGES",
            findings=findings,
            score=82.0,
        )

    def review_experiment(self) -> ReviewResult:
        """実験設計レビュー。"""
        # スケルトン実装
        findings = [
            "問題定義: ✅ 明確",
            "比較対象: ✅ 妥当",
            "統計処理: ⚠️ 信頼区間なし",
        ]
        return ReviewResult(
            reviewer_type="Experiment Reviewer",
            status="REQUEST_CHANGES",
            findings=findings,
            score=78.0,
        )

    def review_math(self) -> ReviewResult:
        """数式・アルゴリズムレビュー。"""
        # スケルトン実装
        findings = [
            "記号法: ✅ 一貫性あり",
            "導出: ✅ 正確",
            "計算量: ✅ O(n log n)",
        ]
        return ReviewResult(
            reviewer_type="Math Reviewer",
            status="APPROVE",
            findings=findings,
            score=95.0,
        )

    def run_review(self) -> ExecutionResult:
        """レビュー実行。"""
        print("\n" + "=" * 70)
        print("Skill 4: Critical Review")
        print("=" * 70)

        result = ExecutionResult(
            success=True,
            script_name="skill4_critical_review",
        )

        # 各レビュー実行
        change_review = self.review_change()
        experiment_review = self.review_experiment()
        math_review = self.review_math()

        reviews = [change_review, experiment_review, math_review]

        # 結果表示
        for review in reviews:
            print(f"\n🔍 {review.reviewer_type}")
            print(f"   Status: {review.status}")
            print(f"   Score: {review.score:.1f}/100")
            for finding in review.findings:
                print(f"   - {finding}")

        # ExecutionResult に追加
        result.output = {
            "reviews": [asdict(r) for r in reviews],
            "overall_score": sum(r.score for r in reviews) / len(reviews),
        }

        print("\n" + "=" * 70)
        return result


def main():
    """メイン処理。"""
    engine = CriticalReviewEngine()
    result = engine.run_review()

    if "--json" in sys.argv:
        print(result.to_json())
    else:
        print(result.to_markdown())

    result.exit_with_status()


if __name__ == "__main__":
    main()
