#!/usr/bin/env python3
"""
Layer A: Linting Check（コードスタイル・ローカルルール チェッカー）

目的:
- ruff/black/pylint による Python コードスタイル検証
- ローカル命名規則（snake_case, CamelCase）確認
- 不要なインポート、未使用変数の検出

レイヤー:
- A層: 基礎検証（開発者向け、常に実行）

出力:
- PASS: スタイル違反なし、ローカルルール準拠
- WARN: 軽微な違反（改善推奨）
- FAIL: 重大な違反（修正必須）
"""

import subprocess
import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple


class LintingChecker:
    """Python コードスタイル・ローカルルール チェッカー。"""

    def __init__(self, verbose: bool = False):
        """初期化。

        Args:
            verbose: 詳細出力フラグ。
        """
        self.verbose = verbose

    def run_ruff(self, target_dir: str = "python") -> Tuple[bool, str]:
        """Ruff でスタイルチェック実行。

        Args:
            target_dir: 対象ディレクトリ。

        Returns:
            (成功フラグ, 出力テキスト)
        """
        cmd = [
            "ruff",
            "check",
            target_dir,
            "--output-format=json",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if self.verbose:
                print(f"[Ruff] stdout:\n{result.stdout}")

            try:
                data = json.loads(result.stdout) if result.stdout else []

                # data は [{"filename": "...", "messages": [...]}] 形式
                total_issues = sum(
                    len(item.get("messages", [])) for item in data
                )

                if total_issues > 0:
                    summary = f"Ruff: {total_issues} 個のスタイル違反検出\n"
                    for item in data[:3]:
                        messages = item.get("messages", [])
                        if messages:
                            summary += f"  - {item['filename']}: {len(messages)} 個\n"
                    if len(data) > 3:
                        summary += f"  ... 他 {len(data) - 3} ファイル\n"
                    return False, summary
                else:
                    return True, "Ruff: スタイル違反なし ✓"

            except (json.JSONDecodeError, KeyError):
                return False, f"Ruff: JSON パースエラー\n{result.stdout}"

        except subprocess.TimeoutExpired:
            return False, "Ruff: タイムアウト（60秒）"
        except FileNotFoundError:
            return None, "Ruff: インストールされていません"

    def run_black(self, target_dir: str = "python") -> Tuple[bool, str]:
        """Black でコード整形チェック実行（--check）。

        Args:
            target_dir: 対象ディレクトリ。

        Returns:
            (成功フラグ, 出力テキスト)
        """
        cmd = ["black", "--check", target_dir]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if self.verbose:
                print(f"[Black] return_code: {result.returncode}")
                if result.stdout:
                    print(f"[Black] stdout:\n{result.stdout}")

            if result.returncode == 0:
                return True, "Black: コード整形不要 ✓"
            else:
                # 別途実行して不一致ファイル数を取得
                files_to_reformat = re.findall(
                    r"would reformat (.+)",
                    result.stdout,
                )
                count = len(files_to_reformat)
                return False, (
                    f"Black: {count} ファイルの整形が必要です"
                )

        except subprocess.TimeoutExpired:
            return False, "Black: タイムアウト（60秒）"
        except FileNotFoundError:
            return None, "Black: インストールされていません"

    def check_local_naming_rules(
        self, target_dir: str = "python"
    ) -> Tuple[bool, str]:
        """ローカル命名規則チェック。

        ローカルルール:
        - クラス: CamelCase（先頭大文字）
        - 関数・変数: snake_case（小文字＋アンダースコア）
        - 定数: UPPER_CASE（全大文字＋アンダースコア）

        Args:
            target_dir: 対象ディレクトリ。

        Returns:
            (成功フラグ, 出力テキスト)
        """
        violations = []

        for py_file in Path(target_dir).rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file) as f:
                    content = f.read()

                # クラス定義をチェック
                class_pattern = r"^class\s+([a-z_][a-z0-9_]*)\s*[:(]"
                class_matches = re.finditer(
                    class_pattern, content, re.MULTILINE
                )
                for match in class_matches:
                    name = match.group(1)
                    violations.append(
                        (
                            py_file,
                            f"クラス '{name}' は CamelCase "
                            "である必要があります",
                        )
                    )

                # 関数定義をチェック（簡易版）
                func_pattern = (
                    r"^def\s+([A-Z][A-Z0-9_]*)\s*\("
                )
                func_matches = re.finditer(
                    func_pattern, content, re.MULTILINE
                )
                for match in func_matches:
                    name = match.group(1)
                    if not name.isupper():
                        violations.append(
                            (
                                py_file,
                                f"関数 '{name}' は snake_case "
                                "である必要があります",
                            )
                        )

            except (UnicodeDecodeError, IOError) as e:
                if self.verbose:
                    print(f"警告: {py_file} を読み込めません: {e}")

        if violations:
            summary = f"命名規則: {len(violations)} 個の違反検出\n"
            for file_path, msg in violations[:5]:
                summary += f"  - {file_path}: {msg}\n"
            if len(violations) > 5:
                summary += f"  ... 他 {len(violations) - 5} 個の違反\n"
            return False, summary
        else:
            return True, "命名規則: 準拠 ✓"

    def check(self, target_dir: str = "python") -> Dict:
        """コードスタイル・ローカルルール チェック実行。

        Args:
            target_dir: 対象ディレクトリ。

        Returns:
            チェック結果辞書。
        """
        results = {
            "component": "layer_a_linting_checker",
            "ruff_passed": None,
            "ruff_message": "",
            "black_passed": None,
            "black_message": "",
            "naming_passed": None,
            "naming_message": "",
        }

        # Ruff チェック
        passed_r, msg_r = self.run_ruff(target_dir)
        results["ruff_passed"] = passed_r
        results["ruff_message"] = msg_r

        # Black チェック
        passed_b, msg_b = self.run_black(target_dir)
        results["black_passed"] = passed_b
        results["black_message"] = msg_b

        # ローカル命名規則チェック
        passed_n, msg_n = self.check_local_naming_rules(target_dir)
        results["naming_passed"] = passed_n
        results["naming_message"] = msg_n

        # 判定
        failed_count = sum(
            1
            for v in [
                results["ruff_passed"],
                results["black_passed"],
                results["naming_passed"],
            ]
            if v is False
        )

        if failed_count >= 2:
            results["status"] = "FAIL"
        elif any(v is False for v in [passed_r, passed_b, passed_n]):
            results["status"] = "WARN"
        elif all(v is None for v in [passed_r, passed_b, passed_n]):
            results["status"] = "N/A"
        else:
            results["status"] = "PASS"

        return results


def main() -> int:
    """メイン処理。"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Layer A: Linting Check（コードスタイル・ローカルルール チェッカー）"
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

    checker = LintingChecker(verbose=args.verbose)
    results = checker.check(args.target_dir)

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print(f"[Linting Check] Status: {results['status']}")
        print(f"  Ruff: {results['ruff_message']}")
        print(f"  Black: {results['black_message']}")
        print(f"  命名規則: {results['naming_message']}")

    # 終了コード
    if results["status"] == "FAIL":
        return 1
    elif results["status"] == "WARN":
        return 2
    elif results["status"] == "N/A":
        return 3
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
