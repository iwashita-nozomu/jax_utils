#!/usr/bin/env python3
"""
Skill 5: Research Workflow — 研究・改造ワークフロー管理スキル

3 つの主要機能：
1. Iteration Manager: イテレーション（実験サイクル）管理
2. Progress Tracker: 進捗追跡
3. Meta Analysis: メタ分析・傾向分析
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts" / "shared"))
from error_handler import ExecutionResult, ErrorCode


@dataclass
class Iteration:
    """イテレーション（実験サイクル）。"""

    iteration_id: int
    description: str
    status: str  # PLANNED/RUNNING/COMPLETED
    start_date: str
    end_date: str
    score: float  # 0-100


class ResearchWorkflowManager:
    """研究ワークフロー管理エンジン。"""

    def __init__(self):
        """初期化。"""
        self.iterations = []
        self._init_iterations()

    def _init_iterations(self):
        """ダミーイテレーションを初期化。"""
        self.iterations = [
            Iteration(1, "Baseline 実装", "COMPLETED", "2026-03-01", "2026-03-10", 70.0),
            Iteration(2, "アルゴリズムA 試行", "COMPLETED", "2026-03-11", "2026-03-20", 75.0),
            Iteration(3, "アルゴリズムB 試行", "RUNNING", "2026-03-21", "2026-04-01", 82.0),
            Iteration(4, "ハイパラ最適化", "PLANNED", "2026-04-02", "2026-04-15", None),
        ]

    def track_progress(self) -> dict:
        """進捗追跡。"""
        completed = sum(1 for i in self.iterations if i.status == "COMPLETED")
        running = sum(1 for i in self.iterations if i.status == "RUNNING")
        planned = sum(1 for i in self.iterations if i.status == "PLANNED")
        total = len(self.iterations)

        progress_rate = 100.0 * (completed + running) / total

        return {
            "total_iterations": total,
            "completed": completed,
            "running": running,
            "planned": planned,
            "progress_rate": progress_rate,
            "iterations": [asdict(i) for i in self.iterations],
        }

    def analyze_trajectory(self) -> dict:
        """軌跡分析。"""
        completed_iters = [i for i in self.iterations if i.status == "COMPLETED"]
        scores = [i.score for i in completed_iters if i.score is not None]

        if not scores:
            return {"trend": "insufficient_data", "scores": []}

        trend = "UPTREND" if scores[-1] > scores[0] else "DOWNTREND" if scores[-1] < scores[0] else "FLAT"
        avg_score = sum(scores) / len(scores)

        return {
            "trend": trend,
            "scores": scores,
            "average": avg_score,
            "improvement": scores[-1] - scores[0] if len(scores) > 1 else 0,
        }

    def run_workflow_management(self) -> ExecutionResult:
        """ワークフロー管理実行。"""
        print("\n" + "=" * 70)
        print("Skill 5: Research Workflow Management")
        print("=" * 70)

        result = ExecutionResult(
            success=True,
            script_name="skill5_research_workflow",
        )

        # 進捗追跡
        progress = self.track_progress()
        print(f"\n📊 進捗追跡")
        print(f"   • 完了: {progress['completed']}/{progress['total_iterations']}")
        print(f"   • 実行中: {progress['running']}")
        print(f"   • 予定: {progress['planned']}")
        print(f"   • 進捗率: {progress['progress_rate']:.1f}%")

        # 軌跡分析
        trajectory = self.analyze_trajectory()
        print(f"\n📈 軌跡分析")
        print(f"   • トレンド: {trajectory['trend']}")
        print(f"   • 平均スコア: {trajectory['average']:.1f}")
        print(f"   • 改善度: {trajectory['improvement']:+.1f}")

        # イテレーション表示
        print(f"\n🔄 イテレーション一覧")
        for iter_info in self.iterations:
            status_emoji = "✅" if iter_info.status == "COMPLETED" else "🔄" if iter_info.status == "RUNNING" else "📅"
            score_str = f"{iter_info.score:.0f}" if iter_info.score else "N/A"
            print(f"   {status_emoji} Iter {iter_info.iteration_id}: {iter_info.description} ({score_str})")

        result.output = {
            "progress": progress,
            "trajectory": trajectory,
        }

        print("\n" + "=" * 70)
        return result


def main():
    """メイン処理。"""
    manager = ResearchWorkflowManager()
    result = manager.run_workflow_management()

    if "--json" in sys.argv:
        print(result.to_json())
    else:
        print(result.to_markdown())

    result.exit_with_status()


if __name__ == "__main__":
    main()
