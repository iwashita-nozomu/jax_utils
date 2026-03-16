from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import jax

from ..base import get_bool_env


# 責務: JSONL に安全に書ける値へ再帰的に正規化します。
def _to_jsonable(value: Any) -> Any:
    """JSON へ変換可能な形へ正規化します。

    Notes
    -----
    - HLO 文字列は巨大になり得るため、ここでは「壊れない」ことを優先します。
    - JAX Array などはこのユーティリティでは扱いません（HLO取得結果は文字列中心の想定）。
    """
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


# 責務: HLO テキストのサイズ情報を数えます。
def _text_metrics(text: str, /) -> dict[str, int]:
    return {
        "chars": len(text),
        "lines": len([line for line in text.splitlines() if line.strip()]),
        "utf8_bytes": len(text.encode("utf-8")),
    }


# 責務: lowering 済み関数から dialect ごとの HLO 情報を取り出します。
def _collect_ir_metadata(lowered: Any, /) -> tuple[str, str, dict[str, object]]:
    stablehlo_text: str | None = None
    hlo_text: str | None = None
    hlo_proto_bytes: int | None = None

    try:
        stablehlo_ir = lowered.compiler_ir(dialect="stablehlo")
        stablehlo_text = str(stablehlo_ir)
    except Exception:
        stablehlo_text = None

    try:
        hlo_ir = lowered.compiler_ir(dialect="hlo")
        if hasattr(hlo_ir, "as_hlo_text"):
            hlo_text = str(hlo_ir.as_hlo_text())
        else:
            hlo_text = str(hlo_ir)
        if hasattr(hlo_ir, "as_serialized_hlo_module_proto"):
            hlo_proto_bytes = len(hlo_ir.as_serialized_hlo_module_proto())
    except Exception:
        hlo_text = None
        hlo_proto_bytes = None

    if stablehlo_text is not None:
        preferred_dialect = "stablehlo"
        preferred_text = stablehlo_text
    elif hlo_text is not None:
        preferred_dialect = "hlo"
        preferred_text = hlo_text
    else:
        raise RuntimeError("Failed to get HLO text from lowered compiler IR.")

    size: dict[str, object] = {
        "preferred_dialect": preferred_dialect,
        "preferred_text": _text_metrics(preferred_text),
        "stablehlo_text": _text_metrics(stablehlo_text) if stablehlo_text is not None else None,
        "hlo_text": _text_metrics(hlo_text) if hlo_text is not None else None,
        "hlo_proto_bytes": hlo_proto_bytes,
    }
    return preferred_dialect, preferred_text, size


# 責務: compile 後に取れる size / cost 情報を JSON 化します。
def _collect_compiled_metadata(lowered: Any, /) -> dict[str, object]:
    try:
        compiled = lowered.compile()
    except Exception as exc:
        return {
            "available": False,
            "error": str(exc),
        }

    cost_analysis: dict[str, float] | None = None
    memory_analysis: dict[str, int | float] | None = None

    try:
        raw_cost = compiled.cost_analysis()
        if isinstance(raw_cost, dict):
            cost_analysis = {str(key): float(value) for key, value in raw_cost.items()}
    except Exception:
        cost_analysis = None

    try:
        raw_memory = compiled.memory_analysis()
        memory_analysis = {
            "generated_code_size_in_bytes": int(raw_memory.generated_code_size_in_bytes),
            "argument_size_in_bytes": int(raw_memory.argument_size_in_bytes),
            "output_size_in_bytes": int(raw_memory.output_size_in_bytes),
            "alias_size_in_bytes": int(raw_memory.alias_size_in_bytes),
            "temp_size_in_bytes": int(raw_memory.temp_size_in_bytes),
            "host_generated_code_size_in_bytes": int(raw_memory.host_generated_code_size_in_bytes),
            "host_argument_size_in_bytes": int(raw_memory.host_argument_size_in_bytes),
            "host_output_size_in_bytes": int(raw_memory.host_output_size_in_bytes),
            "host_alias_size_in_bytes": int(raw_memory.host_alias_size_in_bytes),
            "host_temp_size_in_bytes": int(raw_memory.host_temp_size_in_bytes),
        }
    except Exception:
        memory_analysis = None

    return {
        "available": True,
        "cost_analysis": cost_analysis,
        "memory_analysis": memory_analysis,
    }


# 責務: HLO を 1 レコードの JSONL として追記保存します。
def dump_hlo_jsonl(
    func: Callable[..., Any],
    /,
    *args: Any,
    out_path: str | Path,
    tag: str,
    **kwargs: Any,
) -> None:
    """HLO を JSONL で 1 行出力します。

    Parameters
    ----------
    func:
        解析対象の関数。
    args, kwargs:
        `jax.jit(func).lower(*args, **kwargs)` に渡す引数。
    out_path:
        JSONL の出力先（追記）。
    tag:
        解析対象を識別するためのラベル（例: "minres_step"）。

    Notes
    -----
    - 実行制御は `JAX_UTIL_ENABLE_HLO_DUMP` に従います。
    - HLO 取得は重い処理になり得るため、既定では無効 (`False`) です。
    """
    # Notes:
    # - テストや対話環境では、この関数呼び出しより後に環境変数が変更される場合があります。
    # - そのためフラグはモジュール定数を参照せず、都度評価します。
    if not get_bool_env("JAX_UTIL_ENABLE_HLO_DUMP", False):
        return

    lowered = jax.jit(func).lower(*args, **kwargs)
    preferred_dialect, hlo_text, size = _collect_ir_metadata(lowered)
    record: dict[str, object] = {
        "case": "hlo",
        "tag": tag,
        "dialect": preferred_dialect,
        "hlo": hlo_text,
        "size": size,
        "compiled": _collect_compiled_metadata(lowered),
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(record), ensure_ascii=False))
        f.write("\n")
