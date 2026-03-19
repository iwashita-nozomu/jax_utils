"""ベンチマーク結果レポート生成ツール。

責務:
- Light/Heavy/Extreme の結果から CSV・HTML・Markdown レポートを生成
- 複数レベル間の性能比較グラフ
- 性能トレンドの可視化
"""

from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any
from datetime import datetime

import numpy as np


def load_benchmark_json(json_file: str | Path) -> dict[str, Any]:
    """ベンチマーク JSON ファイルを読み込む。"""
    with open(json_file, "r") as f:
        return json.load(f)


def extract_scaling_results(
    benchmark_data: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """JSON から各ベンチマーク種別の結果を抽出。"""
    results = {"scaling": [], "refinement": [], "dtype": []}
    
    for benchmark in benchmark_data.get("suite", {}).get("benchmarks", []):
        bench_name = benchmark.get("benchmark", "")
        if bench_name == "initialization_scaling":
            results["scaling"] = benchmark.get("results", [])
        elif bench_name == "level_refinement":
            results["refinement"] = benchmark.get("results", [])
        elif bench_name == "dtype_comparison":
            results["dtype"] = benchmark.get("results", [])
    
    return results


def generate_csv_reports(results_dir: Path) -> None:
    """各レベルの結果を CSV 形式で出力。"""
    for level_file in results_dir.glob("*.json"):
        if level_file.name.startswith("extreme.log"):
            continue
        
        try:
            data = load_benchmark_json(level_file)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        
        level_name = level_file.stem  # "light", "heavy" など
        extracted = extract_scaling_results(data)
        
        # 初期化スケーリング CSV
        if extracted["scaling"]:
            csv_file = results_dir / f"{level_name}_scaling.csv"
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "dimension",
                        "level",
                        "num_evaluation_points",
                        "init_time_mean_sec",
                        "integral_time_mean_sec",
                    ],
                )
                writer.writeheader()
                for row in extracted["scaling"]:
                    writer.writerow({
                        "dimension": row["dimension"],
                        "level": row["level"],
                        "num_evaluation_points": row["num_evaluation_points"],
                        "init_time_mean_sec": row["init_time"]["mean_sec"],
                        "integral_time_mean_sec": row["integral_time"]["mean_sec"],
                    })
        
        # レベル精製 CSV
        if extracted["refinement"]:
            csv_file = results_dir / f"{level_name}_refinement.csv"
            with open(csv_file, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "level",
                        "num_evaluation_points",
                        "init_time_mean_sec",
                        "integral_time_mean_sec",
                    ],
                )
                writer.writeheader()
                for row in extracted["refinement"]:
                    writer.writerow({
                        "level": row["level"],
                        "num_evaluation_points": row["num_evaluation_points"],
                        "init_time_mean_sec": row["init_time"]["mean_sec"],
                        "integral_time_mean_sec": row["integral_time"]["mean_sec"],
                    })


def generate_markdown_report(results_dir: Path) -> None:
    """Markdown 形式の包括的レポートを生成。"""
    
    report = []
    report.append("# Smolyak Integrator ベンチマーク実行レポート\n")
    report.append(f"**生成日**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # 利用可能ファイル確認
    available_levels = []
    for level in ["light", "heavy", "extreme"]:
        json_file = results_dir / f"{level}.json"
        if json_file.exists():
            available_levels.append(level)
    
    report.append(f"## 計測対象レベル: {', '.join(available_levels)}\n\n")
    
    # 各レベルの詳細結果
    for level in available_levels:
        json_file = results_dir / f"{level}.json"
        try:
            data = load_benchmark_json(json_file)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        
        extracted = extract_scaling_results(data)
        
        report.append(f"## {level.upper()} ベンチマーク結果\n\n")
        
        # 初期化スケーリング
        if extracted["scaling"]:
            report.append("### 初期化スケーリング\n\n")
            report.append("| 次元 | 評価点 | 初期化時間 (ms) | 積分時間 (s) |\n")
            report.append("|------|--------|-----------------|----------|\n")
            
            for row in extracted["scaling"]:
                d = row["dimension"]
                n_eval = row["num_evaluation_points"]
                init_ms = row["init_time"]["mean_sec"] * 1000
                integral_s = row["integral_time"]["mean_sec"]
                report.append(
                    f"| {d} | {n_eval} | {init_ms:.2f} | {integral_s:.4f} |\n"
                )
            report.append("\n")
        
        # レベル精製
        if extracted["refinement"]:
            report.append("### レベル精製 (d=3)\n\n")
            report.append("| レベル | 評価点 | 初期化時間 (ms) | 積分時間 (s) |\n")
            report.append("|--------|--------|-----------------|----------|\n")
            
            for row in extracted["refinement"]:
                level_val = row["level"]
                n_eval = row["num_evaluation_points"]
                init_ms = row["init_time"]["mean_sec"] * 1000
                integral_s = row["integral_time"]["mean_sec"]
                report.append(
                    f"| {level_val} | {n_eval} | {init_ms:.2f} | {integral_s:.4f} |\n"
                )
            report.append("\n")
        
        # dtype 比較
        if extracted["dtype"]:
            report.append("### dtype 比較\n\n")
            report.append("| dtype | 初期化時間 (ms) |\n")
            report.append("|-------|----------------|\n")
            
            for row in extracted["dtype"]:
                dtype_str = row["dtype"].split("'")[1]  # "<class 'jax.numpy.float32'>" → "float32"
                init_ms = row["init_time"]["mean_sec"] * 1000
                report.append(f"| {dtype_str} | {init_ms:.2f} |\n")
            report.append("\n")
    
    # 比較分析
    if len(available_levels) >= 2:
        report.append("## レベル間の比較分析\n\n")
        
        light_data = heavy_data = None
        if "light" in available_levels:
            light_data = load_benchmark_json(results_dir / "light.json")
        if "heavy" in available_levels:
            heavy_data = load_benchmark_json(results_dir / "heavy.json")
        
        if light_data and heavy_data:
            light_scaling = extract_scaling_results(light_data)["scaling"]
            heavy_scaling = extract_scaling_results(heavy_data)["scaling"]
            
            report.append("### 初期化時間スケーリング比較\n\n")
            report.append("Light と Heavy の初期化時間の変化:\n\n")
            report.append("| 次元 | Light (ms) | Heavy (ms) | 増加率 |\n")
            report.append("|------|------------|-----------|--------|\n")
            
            for light_row in light_scaling:
                d = light_row["dimension"]
                light_init = light_row["init_time"]["mean_sec"] * 1000
                
                # Heavy で対応する次元を探す
                heavy_init = next(
                    (r["init_time"]["mean_sec"] * 1000
                     for r in heavy_scaling if r["dimension"] == d),
                    None,
                )
                
                if heavy_init is not None:
                    increase_ratio = (heavy_init - light_init) / light_init * 100
                    report.append(
                        f"| {d} | {light_init:.2f} | {heavy_init:.2f} | {increase_ratio:+.1f}% |\n"
                    )
    
    # 推奨事項
    report.append("## 推奨事項\n\n")
    report.append("- **Light**: DevOps・CI/CD 統合用，毎回実行可能\n")
    report.append("- **Heavy**: 改善検証・ベースライン比較用，週 1 回推奨\n")
    report.append(
        "- **Extreme**: 設計限界確認・大規模分析用，月 1 回またはマイルストーン時\n"
    )
    report.append("\n詳細は `benchmark_levels_analysis.md` を参照してください。\n")
    
    # レポート保存
    report_file = results_dir / "BENCHMARK_REPORT.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.writelines(report)
    
    print(f"✓ Markdown レポート生成: {report_file}")


def generate_html_report(results_dir: Path) -> None:
    """HTML 形式のレポートを生成。"""
    
    html = []
    html.append("<!DOCTYPE html>\n")
    html.append("<html lang='ja'>\n")
    html.append("<head>\n")
    html.append("  <meta charset='UTF-8'>\n")
    html.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>\n")
    html.append("  <title>Smolyak Integrator ベンチマークレポート</title>\n")
    html.append("  <style>\n")
    html.append("    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; }\n")
    html.append("    h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }\n")
    html.append("    h2 { color: #34495e; margin-top: 30px; }\n")
    html.append("    table { border-collapse: collapse; width: 100%; margin: 15px 0; }\n")
    html.append("    th, td { border: 1px solid #bdc3c7; padding: 10px; text-align: left; }\n")
    html.append("    th { background-color: #ecf0f1; font-weight: bold; }\n")
    html.append("    tr:nth-child(even) { background-color: #f5f5f5; }\n")
    html.append("    .level-light { background-color: #d5f4e6; }\n")
    html.append("    .level-heavy { background-color: #fadbd8; }\n")
    html.append("    .level-extreme { background-color: #fdeaa8; }\n")
    html.append("  </style>\n")
    html.append("</head>\n")
    html.append("<body>\n")
    html.append(f"  <h1>Smolyak Integrator ベンチマークレポート</h1>\n")
    html.append(f"  <p><strong>生成日:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
    
    # 各レベルの結果
    for level in ["light", "heavy", "extreme"]:
        json_file = results_dir / f"{level}.json"
        if not json_file.exists():
            continue
        
        try:
            data = load_benchmark_json(json_file)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        
        extracted = extract_scaling_results(data)
        
        html.append(f"  <div class='level-{level}'>\n")
        html.append(f"    <h2>{level.upper()} ベンチマーク</h2>\n")
        
        if extracted["scaling"]:
            html.append("    <h3>初期化スケーリング</h3>\n")
            html.append("    <table>\n")
            html.append("      <thead>\n")
            html.append("        <tr><th>次元</th><th>評価点</th><th>初期化 (ms)</th><th>積分 (s)</th></tr>\n")
            html.append("      </thead>\n")
            html.append("      <tbody>\n")
            
            for row in extracted["scaling"]:
                d = row["dimension"]
                n_eval = row["num_evaluation_points"]
                init_ms = row["init_time"]["mean_sec"] * 1000
                integral_s = row["integral_time"]["mean_sec"]
                html.append(
                    f"        <tr><td>{d}</td><td>{n_eval}</td>"
                    f"<td>{init_ms:.2f}</td><td>{integral_s:.4f}</td></tr>\n"
                )
            
            html.append("      </tbody>\n")
            html.append("    </table>\n")
        
        if extracted["refinement"]:
            html.append("    <h3>レベル精製 (d=3)</h3>\n")
            html.append("    <table>\n")
            html.append("      <thead>\n")
            html.append("        <tr><th>レベル</th><th>評価点</th><th>初期化 (ms)</th><th>積分 (s)</th></tr>\n")
            html.append("      </thead>\n")
            html.append("      <tbody>\n")
            
            for row in extracted["refinement"]:
                level_val = row["level"]
                n_eval = row["num_evaluation_points"]
                init_ms = row["init_time"]["mean_sec"] * 1000
                integral_s = row["integral_time"]["mean_sec"]
                html.append(
                    f"        <tr><td>{level_val}</td><td>{n_eval}</td>"
                    f"<td>{init_ms:.2f}</td><td>{integral_s:.4f}</td></tr>\n"
                )
            
            html.append("      </tbody>\n")
            html.append("    </table>\n")
        
        if extracted["dtype"]:
            html.append("    <h3>dtype 比較</h3>\n")
            html.append("    <table>\n")
            html.append("      <thead>\n")
            html.append("        <tr><th>dtype</th><th>初期化 (ms)</th></tr>\n")
            html.append("      </thead>\n")
            html.append("      <tbody>\n")
            
            for row in extracted["dtype"]:
                dtype_str = row["dtype"].split("'")[1]
                init_ms = row["init_time"]["mean_sec"] * 1000
                html.append(
                    f"        <tr><td>{dtype_str}</td><td>{init_ms:.2f}</td></tr>\n"
                )
            
            html.append("      </tbody>\n")
            html.append("    </table>\n")
        
        html.append("  </div>\n")
    
    html.append("</body>\n")
    html.append("</html>\n")
    
    # HTML レポート保存
    report_file = results_dir / "BENCHMARK_REPORT.html"
    with open(report_file, "w", encoding="utf-8") as f:
        f.writelines(html)
    
    print(f"✓ HTML レポート生成: {report_file}")


def main() -> None:
    """すべてのレポートを生成。"""
    results_dir = Path(__file__).parent / "results"
    
    if not results_dir.exists():
        print(f"結果ディレクトリが見つかりません: {results_dir}")
        return
    
    print("=" * 60)
    print("ベンチマークレポート生成")
    print("=" * 60)
    print()
    
    # CSV レポート生成
    print("[1/3] CSV レポート生成中...")
    generate_csv_reports(results_dir)
    
    # Markdown レポート生成
    print("[2/3] Markdown レポート生成中...")
    generate_markdown_report(results_dir)
    
    # HTML レポート生成
    print("[3/3] HTML レポート生成中...")
    generate_html_report(results_dir)
    
    print()
    print("✓ すべてのレポートを生成しました")
    print()
    print("生成ファイル:")
    for f in sorted(results_dir.glob("*REPORT*")):
        print(f"  - {f.name}")
    for f in sorted(results_dir.glob("*_scaling.csv")):
        print(f"  - {f.name}")
    for f in sorted(results_dir.glob("*_refinement.csv")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
