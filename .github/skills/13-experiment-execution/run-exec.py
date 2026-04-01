#!/usr/bin/env python3
"""
Skill 13: Experiment Execution

実験実行の統一管理。
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import subprocess

WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent


def run_stage(stage_name, config):
    """各段階を実行"""
    print(f"📍 Running stage: {stage_name}")
    
    if stage_name == "train":
        # 訓練実行
        cmd = [
            sys.executable,
            str(WORKSPACE_ROOT / "python" / "train.py"),
            "--config", str(config),
        ]
    elif stage_name == "validate":
        # 検証実行
        cmd = [
            sys.executable,
            str(WORKSPACE_ROOT / "python" / "validate.py"),
            "--config", str(config),
        ]
    elif stage_name == "test":
        # テスト実行
        cmd = [
            sys.executable,
            str(WORKSPACE_ROOT / "python" / "test.py"),
            "--config", str(config),
        ]
    else:
        print(f"⚠️  Unknown stage: {stage_name}")
        return False
    
    try:
        result = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT))
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Stage {stage_name} failed: {e}")
        return False


def generate_report(stages_result):
    """レポート生成"""
    print("\n📊 Generating report...")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "stages": stages_result,
        "summary": {
            "success": all(stages_result.values()),
            "total_stages": len(stages_result),
            "completed_stages": sum(1 for v in stages_result.values() if v),
        }
    }
    
    return report


def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description="Run experiment workflow")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--stages", default="train,validate,test", help="Stages to run (comma-separated)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Skill 13: Experiment Execution")
    print("=" * 60)
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return 1
    
    stages = args.stages.split(",")
    stages_result = {}
    
    for stage in stages:
        success = run_stage(stage.strip(), config_path)
        stages_result[stage.strip()] = success
        if not success:
            print(f"❌ Workflow aborted at stage: {stage}")
            break
    
    report = generate_report(stages_result)
    
    # レポート保存
    report_dir = WORKSPACE_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    report_file = report_dir / f"experiment-exec-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 Report saved: {report_file}")
    
    return 0 if report["summary"]["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
