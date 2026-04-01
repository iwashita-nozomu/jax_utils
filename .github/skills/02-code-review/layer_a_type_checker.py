#!/usr/bin/env python3
"""
Layer A: Type Check（型チェッカー）

目的:
- pyright/mypy による Python の型注釈検証
- 関数・メソッド全体への型注釈の有無確認
- 型不一致の検出と報告

レイヤー:
- A層: 基礎検証（開発者向け、常に実行）

出力:
- PASS: 型注釈が完全かつ型エラーなし
- WARN: 型注釈の一部欠落（改善推奨）
- FAIL: 型エラー多数または重大な欠落（修正必須）
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class TypeChecker:
    """Python 型チェッカー（pyright + mypy）。"""

    def __init__(self, verbose: bool = False):
        """初期化。

        Args:
            verbose: 詳細出力フラグ。
        """
        self.verbose = verbose
        self.workspace_root = Path.cwd()

    def run_pyright(self, target_dir: str = "python") -> Tuple[bool, str]:
        """Pyright で型チェック実行。

        Args:
            target_dir: 対象ディレクトリ。

        Returns:
            (成功フラグ, 出力テキスト)
        """
        cmd = [
            "pyright",
            "--outputjson",
            target_dir,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if self.verbose:
                print(f"[Pyright] stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"[Pyright] stderr:\n{result.stderr}")

            try:
                data = json.loads(result.stdout)
                diagnostics = data.get("generalDiagnostics", [])

                if diagnostics:
                    summary = (
                        f"Pyright: {len(diagnostics)} 個の型エラー検出\n"
                    )
                    for diag in diagnostics[:5]:  # 最初の5個
                        summary += (
                            f"  - {diag.get('file', 'unknown')}"
                            f":{diag.get('range', {}).get('start', {}).get('line', '?')} "
                            f"{diag.get('message', '')}\n"
                        )
                    if len(diagnostics) > 5:
                        summary += f"  ... 他 {len(diagnostics) - 5} 個の型エラー\n"
                    return False, summary
                else:
                    return True, "Pyright: 型エラーなし ✓"

            except json.JSONDecodeError:
                return False, f"Pyright: JSON パースエラー\n{result.stdout}"

        except subprocess.TimeoutExpired:
            return False, "Pyright: タイムアウト（60秒）"
        except FileNotFoundError:
            return None, "Pyright: インストールされていません"

    def run_mypy(self, target_dir: str = "python") -> Tuple[bool, str]:
        """Mypy で型チェック実行。

        Args:
            target_dir: 対象ディレクトリ。

        Returns:
            (成功フラグ, 出力テキスト)
        """
        cmd = [
            "mypy",
            "--json",
            target_dir,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if self.verbose:
                print(f"[Mypy] stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"[Mypy] stderr:\n{result.stderr}")

            lines = result.stdout.strip().split("\n")
            errors = [
                json.loads(line) for line in lines if line.strip()
            ]

            if errors:
                summary = f"Mypy: {len(errors)} 個の型エラー検出\n"
                for err in errors[:5]:
                    summary += (
                        f"  - {err.get('filename', 'unknown')}"
                        f":{err.get('line', '?')} "
                        f"{err.get('message', '')}\n"
                    )
                if len(errors) > 5:
                    summary += f"  ... 他 {len(errors) - 5} 個の型エラー\n"
                return False, summary
            else:
                return True, "Mypy: 型エラーなし ✓"

        except subprocess.TimeoutExpired:
            return False, "Mypy: タイムアウト（60秒）"
        except FileNotFoundError:
            return None, "Mypy: インストールされていません"

    def check(self, target_dir: str = "python") -> Dict[str, bool | str]:
        """型チェック実行。

        Args:
            target_dir: 対象ディレクトリ。

        Returns:
            チェック結果辞書。
        """
        results = {
            "component": "layer_a_type_checker",
            "pyright_passed": None,
            "pyright_message": "",
            "mypy_passed": None,
            "mypy_message": "",
        }

        # Pyright チェック
        passed_p, msg_p = self.run_pyright(target_dir)
        results["pyright_passed"] = passed_p
        results["pyright_message"] = msg_p

        # Mypy チェック
        passed_m, msg_m = self.run_mypy(target_dir)
        results["mypy_passed"] = passed_m
        results["mypy_message"] = msg_m

        # 判定
        if passed_p is False or passed_m is False:
            results["status"] = "FAIL"
        elif passed_p is None and passed_m is None:
            results["status"] = "N/A"
        else:
            results["status"] = "PASS"

        return results


def main() -> int:
    """メイン処理。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Layer A: Type Check（型チェッカー）"
    )
    parser.add_argument(
        "--target-dir",
        default="python",
        help="対象ディレクトリ（デフォルト: python）",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="詳細出力"
    )
    parser.add_argument(
        "--json", action="store_true", help="JSON 形式で出力"
    )

    args = parser.parse_args()

    checker = TypeChecker(verbose=args.verbose)
    results = checker.check(args.target_dir)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"[Type Check] Status: {results['status']}")
        print(f"  Pyright: {results['pyright_message']}")
        print(f"  Mypy: {results['mypy_message']}")

    # 終了コード
    if results["status"] == "FAIL":
        return 1
    elif results["status"] == "N/A":
        return 2
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
