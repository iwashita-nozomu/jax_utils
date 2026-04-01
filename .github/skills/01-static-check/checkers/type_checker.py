"""
Skill 1: Static Check — Type Checker (Pyright)

mypy/Pyright による静的型チェック実装
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class TypeCheckResult:
    """型チェック結果"""
    tool: str
    success: bool
    file_count: int
    error_count: int
    warning_count: int
    errors: List[Dict[str, Any]]
    duration_ms: float


class PyrightChecker:
    """Pyright 型チェッカー"""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Args:
            config_file: Pyright 設定ファイル
        """
        self.config_file = config_file or Path("pyrightconfig.json")
    
    def check(self, target_dir: Path = Path("python")) -> TypeCheckResult:
        """型チェック実行
        
        Args:
            target_dir: チェック対象ディレクトリ
        
        Returns:
            チェック結果
        """
        import time
        start = time.time()
        
        cmd = ["pyright", str(target_dir), "--outputjson"]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            # JSON 出力を解析
            output = result.stdout
            
            try:
                data = json.loads(output)
                errors = data.get("generalDiagnostics", [])
            except json.JSONDecodeError:
                errors = []
            
            success = result.returncode == 0
            duration_ms = (time.time() - start) * 1000
            
            return TypeCheckResult(
                tool="pyright",
                success=success,
                file_count=data.get("fileCount", 0) if "data" in locals() else 0,
                error_count=len([e for e in errors if e.get("severity") == "error"]),
                warning_count=len([e for e in errors if e.get("severity") == "warning"]),
                errors=errors,
                duration_ms=duration_ms,
            )
        
        except subprocess.TimeoutExpired:
            return TypeCheckResult(
                tool="pyright",
                success=False,
                file_count=0,
                error_count=1,
                warning_count=0,
                errors=[{"message": "Pyright check timed out"}],
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return TypeCheckResult(
                tool="pyright",
                success=False,
                file_count=0,
                error_count=1,
                warning_count=0,
                errors=[{"message": str(e)}],
                duration_ms=(time.time() - start) * 1000,
            )


class MypyChecker:
    """Mypy 型チェッカー"""
    
    def check(self, target_dir: Path = Path("python")) -> TypeCheckResult:
        """型チェック実行
        
        Args:
            target_dir: チェック対象ディレクトリ
        
        Returns:
            チェック結果
        """
        import time
        start = time.time()
        
        cmd = [
            "mypy",
            str(target_dir),
            "--json",
            "--ignore-missing-imports",
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
            
            errors = []
            try:
                for line in result.stdout.strip().split("\n"):
                    if line:
                        errors.append(json.loads(line))
            except json.JSONDecodeError:
                pass
            
            success = result.returncode == 0
            duration_ms = (time.time() - start) * 1000
            
            return TypeCheckResult(
                tool="mypy",
                success=success,
                file_count=0,
                error_count=len([e for e in errors if e.get("severity") == "error"]),
                warning_count=len([e for e in errors if e.get("severity") == "note"]),
                errors=errors,
                duration_ms=duration_ms,
            )
        
        except subprocess.TimeoutExpired:
            return TypeCheckResult(
                tool="mypy",
                success=False,
                file_count=0,
                error_count=1,
                warning_count=0,
                errors=[{"message": "Mypy check timed out"}],
                duration_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return TypeCheckResult(
                tool="mypy",
                success=False,
                file_count=0,
                error_count=1,
                warning_count=0,
                errors=[{"message": str(e)}],
                duration_ms=(time.time() - start) * 1000,
            )


if __name__ == "__main__":
    print("Pyright Checker Test")
    
    checker = PyrightChecker()
    result = checker.check()
    
    print(f"Tool: {result.tool}")
    print(f"Success: {result.success}")
    print(f"Errors: {result.error_count}")
    print(f"Warnings: {result.warning_count}")
    print(f"Duration: {result.duration_ms:.2f}ms")
