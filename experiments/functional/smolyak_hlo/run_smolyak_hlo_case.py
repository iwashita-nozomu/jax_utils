# Results branch: results/functional-smolyak-hlo
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PYTHON_ROOT = WORKSPACE_ROOT / "python"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_BRANCH_NAME = "results/functional-smolyak-hlo"


# 責務: CLI から platform 指定だけを先に抜き出して JAX import 前に反映する。
def _preparse_platform(argv: list[str], /) -> str:
    default_platform = "cpu"
    for index, token in enumerate(argv):
        if token == "--platform" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--platform="):
            return token.split("=", 1)[1]
    return default_platform


# 責務: CLI の platform 指定を JAX backend 名へ変換する。
def _jax_platform_name(platform: str, /) -> str:
    if platform == "gpu":
        return "cuda"
    return platform


if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = _jax_platform_name(_preparse_platform(sys.argv[1:]))
os.environ["JAX_UTIL_ENABLE_HLO_DUMP"] = "1"

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from jax_util.functional import integrate, initialize_smolyak_integrator
from jax_util.hlo import dump_hlo_jsonl


# 責務: Git コマンドの標準出力を取り出せるときだけ返す。
def _git_stdout(args: list[str], /) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    stdout = completed.stdout.strip()
    return stdout if stdout else None


# 責務: 現在時刻から実験出力ファイル名を決める。
def _timestamped_output_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return RESULTS_DIR / f"smolyak_hlo_{timestamp}.json"


# 責務: JSON に安全な値へ再帰的に正規化する。
def _json_compatible(value: object, /) -> object:
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


# 責務: HLO summary script を呼んで集計 JSON を得る。
def _summarize_hlo_jsonl(jsonl_path: Path, /) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(WORKSPACE_ROOT / "scripts" / "hlo" / "summarize_hlo_jsonl.py"),
            str(jsonl_path),
            "--top",
            "50",
        ],
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "Failed to summarize HLO JSONL.")
    parsed = json.loads(completed.stdout)
    if not isinstance(parsed, dict):
        raise RuntimeError("HLO summary is not a JSON object.")
    return parsed


# 責務: 集計済み HLO から粗いボトルネック候補を文章化する。
def _infer_hlo_hints(summary: dict[str, Any], /) -> list[str]:
    counts: dict[str, int] = {}
    for item in summary.get("top_ops", []):
        if isinstance(item, list) and len(item) == 2:
            counts[str(item[0])] = int(item[1])

    def count_matching(*needles: str) -> int:
        total = 0
        for op_name, count in counts.items():
            lowered = op_name.lower()
            if any(needle in lowered for needle in needles):
                total += count
        return total

    hints: list[str] = []
    control_flow = count_matching("while", "call")
    data_movement = count_matching("gather", "slice", "concatenate", "reshape", "broadcast")
    arithmetic = count_matching("dot", "multiply", "add", "subtract")
    scatter_count = count_matching("scatter")
    dot_count = count_matching("dot")

    if control_flow > 0:
        hints.append("制御フロー (`while` / `call`) が HLO に含まれています。loop 構造が kernel 分割や launch 回数へ影響している可能性があります。")
    if data_movement > arithmetic:
        hints.append("算術演算より gather/slice/reshape などのデータ移動系演算が多く、計算より index 処理が支配的な可能性があります。")
    if scatter_count > 0:
        hints.append("scatter が現れており、prefix 更新や点構成のための書き戻しが HLO に残っています。点構成の表現が GPU 実行コストを押し上げている可能性があります。")
    if arithmetic > 0 and dot_count == 0:
        hints.append("主要演算に `dot` 系が見えず、GEMM 主体ではない構造です。GPU の高スループットを引き出しにくい可能性があります。")
    if not hints:
        hints.append("単純な op 数だけでは支配項が明確でありません。HLO 本文と profiler を併用して確認してください。")
    return hints


# 責務: 係数ベクトルを等間隔で構成する。
def _build_coefficients(dimension: int, start: float, stop: float, dtype_name: str, /) -> jax.Array:
    dtype = getattr(jnp, dtype_name)
    return jnp.linspace(start, stop, dimension, dtype=dtype)


class ExponentialIntegrand(eqx.Module):
    coeffs: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.asarray([jnp.exp(jnp.dot(self.coeffs, x))], dtype=self.coeffs.dtype)


# 責務: 単発積分の lowering 対象を返す。
def _single_integral(coeffs: jax.Array, integrator: Any) -> jax.Array:
    return integrate(ExponentialIntegrand(coeffs), integrator)


# 責務: `fori_loop` を含む反復積分の lowering 対象を返す。
def _repeated_integral(coeffs: jax.Array, integrator: Any, num_repeats: int) -> jax.Array:
    def body(_: int, acc: jax.Array) -> jax.Array:
        return acc + _single_integral(coeffs, integrator)

    initial = jnp.zeros((1,), dtype=coeffs.dtype)
    total = lax.fori_loop(0, num_repeats, body, initial)
    return total / num_repeats


# 責務: 実験コード側で単発積分 wrapper を JIT lowering 対象へ固定する。
@eqx.filter_jit
def _compiled_single_integral(coeffs: jax.Array, integrator: Any) -> jax.Array:
    return _single_integral(coeffs, integrator)


# 責務: 実験コード側で反復積分 wrapper を JIT lowering 対象へ固定する。
@eqx.filter_jit
def _compiled_repeated_integral(coeffs: jax.Array, integrator: Any, num_repeats: int) -> jax.Array:
    return _repeated_integral(coeffs, integrator, num_repeats)


# 責務: 1 ケースの HLO を JSONL と summary へ保存する。
def _run_case(args: argparse.Namespace, output_path: Path, /) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_path.with_suffix(".jsonl")
    summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
    jsonl_path.write_text("", encoding="utf-8")

    integrator = initialize_smolyak_integrator(
        dimension=args.dimension,
        level=args.level,
        prepared_level=args.prepared_level,
        dtype=getattr(jnp, args.dtype),
        chunk_size=args.chunk_size,
    )
    coeffs = _build_coefficients(args.dimension, args.coeff_start, args.coeff_stop, args.dtype)

    dump_hlo_jsonl(
        _compiled_single_integral,
        coeffs,
        integrator,
        out_path=jsonl_path,
        tag="smolyak_single_integral",
    )
    dump_hlo_jsonl(
        _compiled_repeated_integral,
        coeffs,
        integrator,
        args.num_repeats,
        out_path=jsonl_path,
        tag="smolyak_repeated_integral",
    )

    summary = _summarize_hlo_jsonl(jsonl_path)
    summary_path.write_text(json.dumps(_json_compatible(summary), ensure_ascii=False, indent=2), encoding="utf-8")

    result: dict[str, Any] = {
        "experiment": "smolyak_hlo",
        "results_branch": RESULTS_BRANCH_NAME,
        "worktree_path": str(WORKSPACE_ROOT),
        "script_path": str(Path(__file__).resolve()),
        "git_branch": _git_stdout(["branch", "--show-current"]),
        "git_commit": _git_stdout(["rev-parse", "HEAD"]),
        "platform": args.platform,
        "dimension": args.dimension,
        "level": args.level,
        "prepared_level": args.prepared_level if args.prepared_level is not None else args.level,
        "dtype": args.dtype,
        "chunk_size": args.chunk_size,
        "num_repeats": args.num_repeats,
        "coeff_start": args.coeff_start,
        "coeff_stop": args.coeff_stop,
        "json_path": str(output_path),
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "summary": summary,
        "hints": _infer_hlo_hints(summary),
    }

    output_path.write_text(json.dumps(_json_compatible(result), ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(output_path, RESULTS_DIR / "latest.json")
    shutil.copyfile(jsonl_path, RESULTS_DIR / "latest.jsonl")
    shutil.copyfile(summary_path, RESULTS_DIR / "latest_summary.json")
    return result


# 責務: CLI 引数を組み立てる。
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump and summarize HLO for a single Smolyak integration case.")
    parser.add_argument("--platform", type=str, choices=("cpu", "gpu"), default=os.environ.get("JAX_PLATFORMS", "cpu"))
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--prepared-level", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float32", choices=("float16", "bfloat16", "float32", "float64"))
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--num-repeats", type=int, default=16)
    parser.add_argument("--coeff-start", type=float, default=-0.55)
    parser.add_argument("--coeff-stop", type=float, default=0.65)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    output_path = args.output if args.output is not None else _timestamped_output_path()
    result = _run_case(args, output_path)
    print(json.dumps(_json_compatible({
        "json": result["json_path"],
        "jsonl": result["jsonl_path"],
        "summary": result["summary_path"],
        "hints": result["hints"],
    }), ensure_ascii=False))


if __name__ == "__main__":
    main()
