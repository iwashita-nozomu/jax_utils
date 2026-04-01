"""
Skill 1: Static Check — Coverage Analyzer

コード カバレッジ分析・レポート生成
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class CoverageMetric:
    """カバレッジメトリクス"""
    total_lines: int
    covered_lines: int
    missed_lines: int
    percent_covered: float
    branches: Optional[int] = None
    branch_coverage: Optional[float] = None


@dataclass
class CoverageReport:
    """カバレッジレポート"""
    overall: CoverageMetric
    by_file: Dict[str, CoverageMetric]
    low_coverage_files: List[tuple[str, float]]
    duration_ms: float
    success: bool


class CoverageAnalyzer:
    """コードカバレッジ分析"""
    
    def __init__(
        self,
        source_dir: Path = Path("python"),
        min_coverage: float = 70.0,
    ):
        """
        Args:
            source_dir: ソースコードディレクトリ
            min_coverage: 最小要求カバレッジ (%)
        """
        self.source_dir = source_dir
        self.min_coverage = min_coverage
    
    def analyze(self) -> CoverageReport:
        """カバレッジ分析実行
        
        Returns:
            カバレッジレポート
        """
        import time
        start = time.time()
        
        # coverage.py で計測
        try:
            result = subprocess.run(
                [
                    "coverage", "run",
                    "-m", "pytest",
                    "python/tests",
                    "--cov=python",
                    "--cov-report=json",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            
            # JSON リポート読込
            coverage_file = Path(".coverage.json")
            if not coverage_file.exists():
                return CoverageReport(
                    overall=CoverageMetric(0, 0, 0, 0),
                    by_file={},
                    low_coverage_files=[],
                    duration_ms=(time.time() - start) * 1000,
                    success=False,
                )
            
            with open(coverage_file, "r", encoding="utf-8") as f:
                cov_data = json.load(f)
            
            # 全体メトリクス
            totals = cov_data.get("totals", {})
            overall = CoverageMetric(
                total_lines=totals.get("num_statements", 0),
                covered_lines=totals.get("covered_lines", 0),
                missed_lines=totals.get("missing_lines", 0),
                percent_covered=totals.get("percent_covered", 0),
            )
            
            # ファイル別メトリクス
            by_file = {}
            low_coverage_files = []
            
            for file_path, file_data in cov_data.get("files", {}).items():
                if "python/" not in file_path:
                    continue
                
                summary = file_data.get("summary", {})
                metric = CoverageMetric(
                    total_lines=summary.get("num_statements", 0),
                    covered_lines=summary.get("covered_lines", 0),
                    missed_lines=summary.get("missing_lines", 0),
                    percent_covered=summary.get("percent_covered", 0),
                )
                
                by_file[file_path] = metric
                
                # 低カバレッジファイル
                if metric.percent_covered < self.min_coverage:
                    low_coverage_files.append((file_path, metric.percent_covered))
            
            success = overall.percent_covered >= self.min_coverage
            
            return CoverageReport(
                overall=overall,
                by_file=by_file,
                low_coverage_files=sorted(low_coverage_files, key=lambda x: x[1]),
                duration_ms=(time.time() - start) * 1000,
                success=success,
            )
        
        except subprocess.TimeoutExpired:
            return CoverageReport(
                overall=CoverageMetric(0, 0, 0, 0),
                by_file={},
                low_coverage_files=[],
                duration_ms=(time.time() - start) * 1000,
                success=False,
            )
        except Exception as e:
            print(f"Error: {e}")
            return CoverageReport(
                overall=CoverageMetric(0, 0, 0, 0),
                by_file={},
                low_coverage_files=[],
                duration_ms=(time.time() - start) * 1000,
                success=False,
            )
    
    def generate_html_report(self, output_dir: Path = Path("reports/coverage")) -> bool:
        """HTML レポート生成
        
        Args:
            output_dir: 出力ディレクトリ
        
        Returns:
             成功/失敗
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            result = subprocess.run(
                ["coverage", "html", "-d", str(output_dir)],
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            return result.returncode == 0
        
        except Exception as e:
            print(f"Failed to generate HTML report: {e}")
            return False
    
    def get_missing_lines(self, file_path: str) -> List[int]:
        """特定ファイルの未カバーラインを取得
        
        Args:
            file_path: ファイルパス
        
        Returns:
            未カバーライン番号リスト
        """
        try:
            coverage_file = Path(".coverage.json")
            if not coverage_file.exists():
                return []
            
            with open(coverage_file, "r", encoding="utf-8") as f:
                cov_data = json.load(f)
            
            file_info = cov_data.get("files", {}).get(file_path, {})
            return file_info.get("missing_lines", [])
        
        except Exception as e:
            print(f"Error: {e}")
            return []


if __name__ == "__main__":
    print("Coverage Analyzer")
    
    analyzer = CoverageAnalyzer()
    report = analyzer.analyze()
    
    print(f"\nOverall Coverage: {report.overall.percent_covered:.1f}%")
    print(f"Covered: {report.overall.covered_lines}/{report.overall.total_lines}")
    print(f"Files analyzed: {len(report.by_file)}")
    
    if report.low_coverage_files:
        print(f"\nLow coverage files ({len(report.low_coverage_files)}):")
        for file_path, coverage in report.low_coverage_files[:5]:
            print(f"  {file_path}: {coverage:.1f}%")
    
    print(f"\nDuration: {report.duration_ms:.2f}ms")
