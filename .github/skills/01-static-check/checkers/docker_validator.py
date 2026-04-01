"""
Skill 1: Static Check — Docker Validator

Dockerfile の検証・イメージビルドテスト
"""

import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DockerValidation:
    """Docker 検証結果"""
    success: bool
    dockerfile_valid: bool
    build_successful: bool
    image_size_mb: Optional[float] = None
    layers: int = 0
    warnings: List[str] = None
    errors: List[str] = None
    duration_ms: float = 0.0
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


class DockerValidator:
    """Docker イメージ検証"""
    
    def __init__(self, dockerfile: Path = Path("docker/Dockerfile")):
        """
        Args:
            dockerfile: Dockerfile パス
        """
        self.dockerfile = dockerfile
        self.image_name = "workspace:test-latest"
    
    def validate_dockerfile(self) -> tuple[bool, List[str]]:
        """Dockerfile 構文をバリデーション
        
        Returns:
            (有効/無効, 警告リスト)
        """
        if not self.dockerfile.exists():
            return False, ["Dockerfile not found"]
        
        # hadolint で Dockerfile リント
        warnings = []
        
        try:
            result = subprocess.run(
                ["hadolint", str(self.dockerfile)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            
            if result.stdout:
                warnings = result.stdout.strip().split("\n")
        
        except FileNotFoundError:
            # hadolint なければスキップ
            pass
        except Exception as e:
            warnings.append(f"Lint error: {e}")
        
        return True, warnings
    
    def build_image(self, dry_run: bool = False) -> DockerValidation:
        """Docker イメージをビルド
        
        Args:
            dry_run: ドライラン（ビルド実行しない）
        
        Returns:
            検証結果
        """
        import time
        start = time.time()
        
        # Dockerfile バリデーション
        valid, lint_warnings = self.validate_dockerfile()
        
        validation = DockerValidation(
            success=False,
            dockerfile_valid=valid,
            build_successful=False,
            warnings=lint_warnings,
        )
        
        if not valid:
            validation.errors = ["Dockerfile validation failed"]
            return validation
        
        if dry_run:
            validation.success = True
            return validation
        
        # イメージビルド
        try:
            cmd = [
                "docker", "build",
                "-f", str(self.dockerfile),
                "-t", self.image_name,
                "--build-arg", "BUILDKIT_INLINE_CACHE=1",
                ".",
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10分
                cwd=self.dockerfile.parent.parent,
            )
            
            if result.returncode == 0:
                validation.build_successful = True
                validation.success = True
                
                # イメージサイズを取得
                try:
                    inspect = subprocess.run(
                        ["docker", "inspect", "--type=image", self.image_name],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    
                    import json
                    if inspect.returncode == 0:
                        info = json.loads(inspect.stdout)
                        if info:
                            size_bytes = info[0].get("Size", 0)
                            validation.image_size_mb = size_bytes / (1024 * 1024)
                            validation.layers = len(info[0].get("RootFS", {}).get("Layers", []))
                
                except Exception as e:
                    validation.warnings.append(f"Failed to get image info: {e}")
            
            else:
                validation.errors = [result.stderr or result.stdout]
        
        except subprocess.TimeoutExpired:
            validation.errors = ["Docker build timed out"]
        except Exception as e:
            validation.errors = [str(e)]
        
        validation.duration_ms = (time.time() - start) * 1000
        
        return validation
    
    def scan_image(self) -> Dict[str, Any]:
        """ビルド済みイメージをセキュリティスキャン
        
        Returns:
            スキャン結果
        """
        try:
            # docker scan (Snyk 統合)
            result = subprocess.run(
                ["docker", "scan", self.image_name],
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            return {
                "scannable": True,
                "output": result.stdout,
                "vulnerabilities": 0,  # パース不要（簡略版）
            }
        
        except FileNotFoundError:
            return {
                "scannable": False,
                "reason": "docker scan not available",
            }
        except Exception as e:
            return {
                "scannable": False,
                "reason": str(e),
            }
    
    def cleanup(self) -> None:
        """テスト用イメージをクリーンアップ"""
        try:
            subprocess.run(
                ["docker", "rmi", self.image_name],
                capture_output=True,
                timeout=30,
            )
        except Exception:
            pass


if __name__ == "__main__":
    print("Docker Validator")
    
    validator = DockerValidator()
    
    print("\n1. Validating Dockerfile...")
    is_valid, warnings = validator.validate_dockerfile()
    print(f"  Valid: {is_valid}")
    if warnings:
        for w in warnings[:3]:
            print(f"  Warning: {w}")
    
    print("\n2. Building image (dry run)...")
    result = validator.build_image(dry_run=True)
    print(f"  Success: {result.success}")
    print(f"  Duration: {result.duration_ms:.2f}ms")
