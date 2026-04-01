#!/usr/bin/env python3
"""
Skill 2 全レイヤー統合テストスイート

対象:
- Layer A (基礎検証)
- Layer B (深度検証)  
- Layer C (統合検証)
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, Tuple


class SkillCodeReviewFullTests:
    """Skill 2 全レイヤー統合テスト。"""

    def __init__(self):
        """初期化。"""
        self.skill_dir = Path(".github/skills/02-code-review")
        self.passed = 0
        self.failed = 0

    def run_command(self, cmd: list) -> Tuple[int, str]:
        """コマンド実行。"""
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )
        return result.returncode, result.stdout + result.stderr

    def test_layer_a(self) -> bool:
        """Layer A 統合テスト。"""
        print("⏳ テスト: Layer A - 基礎検証")

        cmd = [
            "python3",
            str(self.skill_dir / "run-review.py"),
            "--phase",
            "A",
            "--json",
        ]

        code, output = self.run_command(cmd)

        try:
            data = json.loads(output)
            assert "phases" in data
            assert "A" in data["phases"]
            layer_a = data["phases"]["A"]
            assert "overall_status" in layer_a
            print(f"  ✅ PASS (status={layer_a['overall_status']})")
            self.passed += 1
            return True
        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            self.failed += 1
            return False

    def test_layer_b(self) -> bool:
        """Layer B 統合テスト。"""
        print("⏳ テスト: Layer B - 深度検証")

        cmd = [
            "python3",
            str(self.skill_dir / "run-review.py"),
            "--phase",
            "B",
            "--json",
        ]

        code, output = self.run_command(cmd)

        try:
            data = json.loads(output)
            assert "phases" in data
            assert "B" in data["phases"]
            layer_b = data["phases"]["B"]
            assert "overall_status" in layer_b
            print(f"  ✅ PASS (status={layer_b['overall_status']})")
            self.passed += 1
            return True
        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            self.failed += 1
            return False

    def test_layer_c(self) -> bool:
        """Layer C 統合テスト。"""
        print("⏳ テスト: Layer C - 統合検証")

        cmd = [
            "python3",
            str(self.skill_dir / "run-review.py"),
            "--phase",
            "C",
            "--json",
        ]

        code, output = self.run_command(cmd)

        try:
            data = json.loads(output)
            assert "phases" in data
            assert "C" in data["phases"]
            layer_c = data["phases"]["C"]
            assert "overall_status" in layer_c
            print(f"  ✅ PASS (status={layer_c['overall_status']})")
            self.passed += 1
            return True
        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            self.failed += 1
            return False

    def test_all_layers(self) -> bool:
        """全レイヤー統合テスト。"""
        print("⏳ テスト: 全レイヤー統合実行")

        cmd = [
            "python3",
            str(self.skill_dir / "run-review.py"),
            "--phase",
            "all",
            "--json",
        ]

        code, output = self.run_command(cmd)

        try:
            data = json.loads(output)
            assert "phases" in data
            assert "A" in data["phases"]
            assert "B" in data["phases"]
            assert "C" in data["phases"]
            print(f"  ✅ PASS (3レイヤー実行)")
            self.passed += 1
            return True
        except Exception as e:
            print(f"  ❌ FAIL: {e}")
            self.failed += 1
            return False

    def run_all_tests(self) -> int:
        """全テスト実行。"""
        print("\n" + "=" * 70)
        print("Skill 2: Code Review — 全レイヤー統合テストスイート")
        print("=" * 70 + "\n")

        results = {
            "test_layer_a": self.test_layer_a(),
            "test_layer_b": self.test_layer_b(),
            "test_layer_c": self.test_layer_c(),
            "test_all_layers": self.test_all_layers(),
        }

        print("\n" + "=" * 70)
        print(f"テスト結果サマリ")
        print("=" * 70)
        print(f"✅ PASS: {self.passed}/{len(results)}")
        print(f"❌ FAIL: {self.failed}/{len(results)}")
        print("=" * 70)

        return 0 if self.failed == 0 else 1


def main() -> int:
    """メイン処理。"""
    tests = SkillCodeReviewFullTests()
    return tests.run_all_tests()


if __name__ == "__main__":
    sys.exit(main())
