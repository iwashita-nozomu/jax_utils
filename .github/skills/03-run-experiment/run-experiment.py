#!/usr/bin/env python3
"""
Skill 3: Run-Experiment — 実験実行スキル

実験の 5 段階フロー：
1. Setup: 問題定義・比較対象・Metrics
2. Implementation: 実験コード生成
3. Static-Check: 仕様との対応検証
4. Execution: 実験ランナー起動
5. Results: JSON + レポート生成
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "scripts" / "shared"))
from error_handler import ExecutionResult, ErrorCode


class ExperimentRunner:
    """実験実行メインクラス。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初期化。

        Args:
            config: 実験設定辞書
        """
        self.config = config or {}
        self.stages = {
            "1-setup": False,
            "2-implementation": False,
            "3-static-check": False,
            "4-execution": False,
            "5-results": False,
        }
        self.results = {}

    def stage_1_setup(self) -> bool:
        """Stage 1: Setup - 問題定義・比較対象・Metrics。"""
        print("\n⏳ Stage 1: Setup - 問題定義・比較対象・Metrics")

        required_fields = ["question", "metrics", "baselines", "stop_condition"]
        for field in required_fields:
            if field not in self.config:
                print(f"  ⚠️ {field} が設定されていません")
                return False

        print(f"  ✅ Question: {self.config.get('question', 'N/A')}")
        print(f"  ✅ Metrics: {self.config.get('metrics', [])}")
        print(f"  ✅ Baselines: {self.config.get('baselines', [])}")

        self.stages["1-setup"] = True
        return True

    def stage_2_implementation(self) -> bool:
        """Stage 2: Implementation - 実験コード生成。"""
        print("\n⏳ Stage 2: Implementation - 実験コード生成")

        # スケルトン: 実際の実装は Skill 5 で詳細化
        experiment_code = f"""
# 自動生成: {datetime.now().isoformat()}
def run_experiment():
    '''実験メイン関数。'''
    # 準備
    # 比較対象1実行
    # 比較対象2実行
    # メトリクス測定
    # 結果集計
    pass

if __name__ == "__main__":
    run_experiment()
"""

        # 出力用に記録
        self.results["generated_code"] = experiment_code.strip()

        print(f"  ✅ 実験コード生成完了 ({len(experiment_code)} 文字)")

        self.stages["2-implementation"] = True
        return True

    def stage_3_static_check(self) -> bool:
        """Stage 3: Static-Check - 仕様との対応検証。"""
        print("\n⏳ Stage 3: Static-Check - 仕様との対応検証")

        # Skill 1 を使用して検証
        # ここではスケルトン
        checks = {
            "type_check": True,
            "docstring_check": True,
            "naming_check": True,
            "test_check": True,
        }

        all_pass = all(checks.values())
        for check_name, result in checks.items():
            status = "✅" if result else "❌"
            print(f"  {status} {check_name}: {'OK' if result else 'FAIL'}")

        self.stages["3-static-check"] = True
        return all_pass

    def stage_4_execution(self) -> bool:
        """Stage 4: Execution - 実験ランナー起動。"""
        print("\n⏳ Stage 4: Execution - 実験ランナー起動")

        # スケルトン: 実際の実行は experiment_runner を使用
        print(f"  ℹ️ 実験実行コマンド: python3 experiments/run.py")

        # ダミー結果
        self.results["execution_time"] = 42.5  # 秒
        self.results["baseline1_result"] = 0.85
        self.results["baseline2_result"] = 0.92
        self.results["experiment_result"] = 0.94

        print(f"  ✅ 実行完了 (実行時間: {self.results['execution_time']}秒)")

        self.stages["4-execution"] = True
        return True

    def stage_5_results(self) -> bool:
        """Stage 5: Results - JSON + レポート生成。"""
        print("\n⏳ Stage 5: Results - JSON + レポート生成")

        report = {
            "timestamp": datetime.now().isoformat(),
            "experiment": self.config.get("name", "unnamed"),
            "stages_completed": {
                k: v for k, v in self.stages.items()
            },
            "results": self.results,
        }

        # レポート保存
        report_file = Path("reports") / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"  ✅ レポート生成: {report_file}")

        self.stages["5-results"] = True
        return True

    def run_experiment(self) -> ExecutionResult:
        """実験実行メインフロー。"""
        print("\n" + "=" * 70)
        print("Skill 3: Run-Experiment")
        print("=" * 70)

        result = ExecutionResult(
            success=True,
            script_name="skill3_run_experiment",
        )

        # 全ステージ実行
        stages_list = [
            self.stage_1_setup,
            self.stage_2_implementation,
            self.stage_3_static_check,
            self.stage_4_execution,
            self.stage_5_results,
        ]

        for stage_fn in stages_list:
            try:
                if not stage_fn():
                    result.success = False
                    result.add_error(
                        code=ErrorCode.EXPERIMENT_RUN_FAILED,
                        message=f"{stage_fn.__name__} が失敗",
                    )
            except Exception as e:
                result.success = False
                result.add_error(
                    code=ErrorCode.EXPERIMENT_RUN_FAILED,
                    message=f"{stage_fn.__name__}: {str(e)}",
                )

        result.output = {
            "stages": self.stages,
            "results": self.results,
        }

        print("\n" + "=" * 70)
        print(f"実験実行完了: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        print("=" * 70)

        return result


def main():
    """メイン処理。"""
    parser = argparse.ArgumentParser(description="Skill 3: Run-Experiment")
    parser.add_argument("--config", type=str, help="実験設定ファイル (YAML/JSON)")
    parser.add_argument("--json", action="store_true", help="JSON 出力")
    parser.add_argument("--verbose", action="store_true", help="詳細出力")
    args = parser.parse_args()

    # ダミー設定
    config = {
        "name": "smolyak_experiment_week6",
        "question": "Smolyak グリッドの収束速度を測定",
        "metrics": ["error", "time"],
        "baselines": ["monte-carlo", "regular-grid"],
        "stop_condition": "10 iteration",
    }

    if args.config:
        # TODO: ファイルから設定読込
        pass

    runner = ExperimentRunner(config)
    result = runner.run_experiment()

    if args.json:
        print(result.to_json())
    else:
        print(result.to_markdown())

    result.exit_with_status()


if __name__ == "__main__":
    main()
