#!/usr/bin/env python3
"""
Skill 7: Health Monitor — プロジェクト健全性監視スキル

5 つの主要メトリクスを監視：
1. Code Quality (型検査・Linting・Docstring)
2. Test Coverage (テストカバレッジ)
3. Security (脆弱性スキャン・Secret 検出)
4. Performance (ビルド時間・実行時間)
5. Documentation (ドキュメント整合性)
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts" / "shared"))
from error_handler import ExecutionResult, ErrorCode


@dataclass
class HealthMetric:
    """健全性メトリック。"""

    name: str
    score: float  # 0-100
    weight: float  # 0-1
    status: str  # GREEN/YELLOW/RED
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換。"""
        return asdict(self)


class HealthMonitor:
    """プロジェクト健全性監視エンジン。"""

    def __init__(self):
        """初期化。"""
        self.metrics: Dict[str, HealthMetric] = {}
        self.health_score = 0.0

    def measure_code_quality(self) -> HealthMetric:
        """Code Quality 測定。"""
        # Skill 2 Layer A の結果を使用
        # スケルトン: 85点と仮定
        score = 85.0
        status = "GREEN" if score >= 80 else "YELLOW" if score >= 60 else "RED"

        return HealthMetric(
            name="Code Quality",
            score=score,
            weight=0.25,
            status=status,
            description="Type Check, Linting, Docstring 整体スコア",
        )

    def measure_test_coverage(self) -> HealthMetric:
        """Test Coverage 測定。"""
        # pytest-cov の結果を使用
        # スケルトン: 78 点と仮定
        score = 78.0
        status = "YELLOW" if score >= 80 else "GREEN" if score >= 60 else "RED"

        return HealthMetric(
            name="Test Coverage",
            score=score,
            weight=0.25,
            status=status,
            description="関数/行/分岐のカバレッジ率",
        )

    def measure_security(self) -> HealthMetric:
        """Security 測定。"""
        # pip-audit + trivy の結果を使用
        # スケルトン: 90 点と仮定
        score = 90.0
        status = "GREEN"

        return HealthMetric(
            name="Security",
            score=score,
            weight=0.20,
            status=status,
            description="脆弱性スキャン・Secret 検出",
        )

    def measure_performance(self) -> HealthMetric:
        """Performance 測定。"""
        # ビルド/テスト実行時間を測定
        # スケルトン: 72 点と仮定
        score = 72.0
        status = "YELLOW"

        return HealthMetric(
            name="Performance",
            score=score,
            weight=0.15,
            status=status,
            description="ビルド時間・テスト実行時間",
        )

    def measure_documentation(self) -> HealthMetric:
        """Documentation 測定。"""
        # Docstring + README の整合性
        # スケルトン: 82 点と仮定
        score = 82.0
        status = "GREEN"

        return HealthMetric(
            name="Documentation",
            score=score,
            weight=0.15,
            status=status,
            description="ドキュメント整合性・更新頻度",
        )

    def run_health_check(self) -> ExecutionResult:
        """健全性チェック実行。"""
        print("\n" + "=" * 70)
        print("Skill 7: Health Monitor")
        print("=" * 70)

        result = ExecutionResult(
            success=True,
            script_name="skill7_health_monitor",
        )

        # 各メトリック測定
        self.metrics["code_quality"] = self.measure_code_quality()
        self.metrics["test_coverage"] = self.measure_test_coverage()
        self.metrics["security"] = self.measure_security()
        self.metrics["performance"] = self.measure_performance()
        self.metrics["documentation"] = self.measure_documentation()

        # 加重スコア計算
        total_score = 0.0
        total_weight = 0.0
        for metric in self.metrics.values():
            total_score += metric.score * metric.weight
            total_weight += metric.weight

        self.health_score = total_score / total_weight if total_weight > 0 else 0.0

        # ステータス判定
        if self.health_score >= 80:
            overall_status = "🟢 EXCELLENT"
        elif self.health_score >= 70:
            overall_status = "🟡 GOOD"
        elif self.health_score >= 60:
            overall_status = "🟠 FAIR"
        else:
            overall_status = "🔴 POOR"

        # メトリクス表示
        print("\n📊 Health Metrics:")
        for metric in self.metrics.values():
            status_emoji = "🟢" if metric.status == "GREEN" else "🟡" if metric.status == "YELLOW" else "🔴"
            print(
                f"  {status_emoji} {metric.name:20} | {metric.score:5.1f} | {metric.description}"
            )

        print(f"\n🏥 Overall Health Score: {self.health_score:.1f}/100 {overall_status}")

        # 結果に追加
        result.output = {
            "health_score": self.health_score,
            "status": overall_status,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "timestamp": datetime.now().isoformat(),
        }

        # 警告チェック
        for metric in self.metrics.values():
            if metric.status != "GREEN":
                result.add_warning(
                    code=ErrorCode.LOW_COVERAGE,
                    message=f"{metric.name} が {metric.status} 状態",
                    context={"score": metric.score},
                    suggestion=f"{metric.name} の改善が推奨されます",
                )

        print("\n" + "=" * 70)

        return result


def main():
    """メイン処理。"""
    import argparse

    parser = argparse.ArgumentParser(description="Skill 7: Health Monitor")
    parser.add_argument("--json", action="store_true", help="JSON 出力")
    parser.add_argument("--verbose", action="store_true", help="詳細出力")
    args = parser.parse_args()

    monitor = HealthMonitor()
    result = monitor.run_health_check()

    if args.json:
        print(result.to_json())
    else:
        print(result.to_markdown())

    result.exit_with_status()


if __name__ == "__main__":
    main()
