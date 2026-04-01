#!/usr/bin/env python3
"""
Skill 8: Experiment Initialization

新規実験プロジェクトの初期化・セットアップを自動化
"""

import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

SKILL_DIR = Path(__file__).parent
WORKSPACE_ROOT = SKILL_DIR.parent.parent.parent


def create_exp_structure(exp_name: str, config_level: str = "template") -> dict:
    """実験ディレクトリ構造を生成"""
    exp_dir = Path("experiments") / exp_name
    
    dirs = {
        "template": ["data", "models", "results", "logs", "configs"],
        "full": ["data/raw", "data/processed", "models", "results", "logs/train", "logs/eval", "configs", "notebooks"],
        "minimal": ["data", "models", "results"],
    }
    
    for dir_name in dirs.get(config_level, dirs["template"]):
        (exp_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    return {"exp_dir": str(exp_dir), "created_at": datetime.now().isoformat()}


def generate_config_template(exp_name: str, output_dir: Path) -> dict:
    """config.yaml テンプレートを生成"""
    config_template = {
        "experiment": {
            "name": exp_name,
            "created_at": datetime.now().isoformat(),
            "description": f"Experiment: {exp_name}",
        },
        "data": {
            "path": "data/",
            "batch_size": 32,
            "shuffle": True,
        },
        "training": {
            "learning_rate": 1e-3,
            "epochs": 100,
            "optimizer": "adam",
        },
    }
    
    config_file = output_dir / "experiment.yaml"
    with open(config_file, "w") as f:
        json.dump(config_template, f, indent=2)
    
    return {"config_file": str(config_file)}


def run(namespace: argparse.Namespace) -> dict:
    """Skill 8 を実行"""
    
    if not namespace.exp_name:
        return {"status": "error", "message": "--exp-name is required"}
    
    # ディレクトリ構造を生成
    dir_result = create_exp_structure(namespace.exp_name, namespace.config)
    
    # 出力ディレクトリを取得
    output_dir = Path(namespace.output) / namespace.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Config を生成
    config_result = generate_config_template(namespace.exp_name, output_dir)
    
    return {
        "status": "success",
        "exp_name": namespace.exp_name,
        "directory": dir_result["exp_dir"],
        "config": config_result["config_file"],
        "created_at": dir_result["created_at"],
    }


def main():
    parser = argparse.ArgumentParser(description="Skill 8: Experiment Initialization")
    parser.add_argument("--exp-name", required=True, help="実験プロジェクト名")
    parser.add_argument("--config", choices=["template", "full", "minimal"], default="template", help="テンプレートレベル")
    parser.add_argument("--output", default="experiments", help="出力ディレクトリ")
    parser.add_argument("--verbose", action="store_true", help="詳細出力")
    
    args = parser.parse_args()
    result = run(args)
    
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
