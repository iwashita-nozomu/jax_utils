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


# 責務: lowering 済み関数から利用可能な HLO テキストを取得します。
def _get_hlo_text(func: Callable[..., Any], /, *args: Any, **kwargs: Any) -> str:
    """JAX lowering から HLO 文字列を取得します。

    優先順:
    1) stablehlo
    2) hlo

    Raises
    ------
    RuntimeError
        HLO の取得に失敗した場合。
    """
    lowered = jax.jit(func).lower(*args, **kwargs)

    for dialect in ("stablehlo", "hlo"):
        try:
            ir = lowered.compiler_ir(dialect=dialect)
            return str(ir)
        except Exception:
            continue

    raise RuntimeError("Failed to get HLO text from lowered compiler IR.")


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

    hlo_text = _get_hlo_text(func, *args, **kwargs)
    record: dict[str, object] = {
        "case": "hlo",
        "tag": tag,
        "dialect": "stablehlo_or_hlo",
        "hlo": hlo_text,
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(record), ensure_ascii=False))
        f.write("\n")
