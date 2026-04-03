#!/usr/bin/env python3
"""One-command driver for the level-1-to-10 same-budget Smolyak experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def _default_dimensions() -> str:
    return ",".join(str(value) for value in range(1, 21))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the level-1-to-10 same-budget Smolyak-vs-Monte-Carlo sweep and "
            "generate both the main report and the literature/theory note."
        ),
    )
    parser.add_argument("--platform", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--dimensions", default=_default_dimensions(), help="Comma-separated dimensions to test.")
    parser.add_argument("--levels", default="1,2,3,4,5,6,7,8,9,10", help="Comma-separated Smolyak levels to test.")
    parser.add_argument("--dtypes", default="float32,float64", help="Comma-separated dtypes to test.")
    parser.add_argument("--family", choices=["gaussian", "quadratic", "exponential"], default="gaussian")
    parser.add_argument("--gaussian-alpha", type=float, default=0.8)
    parser.add_argument("--coeff-start", type=float, default=0.2)
    parser.add_argument("--coeff-stop", type=float, default=0.8)
    parser.add_argument("--warm-repeats", type=int, default=1)
    parser.add_argument("--mc-seeds", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--case-timeout-seconds",
        type=float,
        default=5.0,
        help="Per-case timeout. OOM/timeout cases are left as no result in the final report.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(SCRIPT_DIR / "results" / "smolyak_same_budget_accuracy_levels_1_to_10"),
        help="Directory under which a timestamped report directory will be created.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Optional explicit report directory. If omitted, a timestamped directory is created under --output-dir.",
    )
    return parser


def _run_and_parse(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stderr:
        sys.stderr.write(completed.stderr)
        sys.stderr.flush()
    return json.loads(completed.stdout)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    output_root = Path(args.output_dir).resolve()
    report_dir = (
        Path(args.report_dir).resolve()
        if args.report_dir
        else output_root / f"report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    same_budget_command = [
        sys.executable,
        "-m",
        "experiments.smolyak_experiment.report_smolyak_same_budget_accuracy",
        "--platform",
        args.platform,
        "--dimensions",
        args.dimensions,
        "--levels",
        args.levels,
        "--dtypes",
        args.dtypes,
        "--family",
        args.family,
        "--gaussian-alpha",
        str(args.gaussian_alpha),
        "--coeff-start",
        str(args.coeff_start),
        "--coeff-stop",
        str(args.coeff_stop),
        "--warm-repeats",
        str(args.warm_repeats),
        "--mc-seeds",
        str(args.mc_seeds),
        "--seed",
        str(args.seed),
        "--case-timeout-seconds",
        str(args.case_timeout_seconds),
        "--resume-report-dir",
        str(report_dir),
        "--output-dir",
        str(output_root),
    ]

    same_budget_result = _run_and_parse(same_budget_command)

    theory_result: dict[str, object] | None = None
    if args.family == "gaussian":
        theory_command = [
            sys.executable,
            "-m",
            "experiments.smolyak_experiment.report_smolyak_theory_comparison",
            "--summary-json",
            str(report_dir / "summary.json"),
            "--output-dir",
            str(report_dir),
        ]
        theory_result = _run_and_parse(theory_command)

    payload = {
        "report_dir": str(report_dir),
        "summary_json": str(report_dir / "summary.json"),
        "report_md": str(report_dir / "report.md"),
        "same_budget_result": same_budget_result,
        "theory_result": theory_result,
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
