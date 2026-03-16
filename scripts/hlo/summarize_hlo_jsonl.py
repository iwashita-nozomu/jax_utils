from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from collections import Counter
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """JSONL を 1 行ずつ読み込みます。"""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            obj = json.loads(s)
            if isinstance(obj, dict):
                yield obj


def _count_hlo_ops(hlo_text: str) -> Counter[str]:
    """HLO テキストから雑に命令種別っぽいトークンを数えます。

    Notes
    -----
    - HLO の厳密パーサは導入せず、簡易な統計を取るためのヒューリスティックです。
    - 将来的に stablehlo の構文が変わった場合はここを調整します。
    """
    counter: Counter[str] = Counter()
    for line in hlo_text.splitlines():
        t = line.strip()
        if not t:
            continue
        if t.startswith("//"):
            continue
        if "=" not in t:
            continue
        rhs = t.split("=", 1)[1].strip()
        op = rhs.split("(", 1)[0].strip()
        if not op:
            continue
        counter[op] += 1
    return counter


def _get_record_key(rec: dict[str, Any], key: str) -> str:
    value = rec.get(key)
    if value is None:
        return ""
    return str(value)


def _count_lines(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


def _update_metric_totals(
    totals: dict[str, dict[str, float]],
    key: str,
    value: Any,
    /,
) -> None:
    if not isinstance(value, (int, float)):
        return
    metric = totals.setdefault(key, {"total": 0.0, "max": 0.0, "count": 0.0})
    numeric = float(value)
    metric["total"] += numeric
    metric["count"] += 1.0
    if numeric > metric["max"]:
        metric["max"] = numeric


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize HLO JSONL.")
    parser.add_argument("jsonl", type=str, help="Path to JSONL (produced by dump_hlo_jsonl).")
    parser.add_argument("--top", type=int, default=30, help="Show top-k ops.")
    parser.add_argument(
        "--only-case",
        type=str,
        default="hlo",
        help="Filter by record['case'] (default: hlo). Use empty string to disable.",
    )
    args = parser.parse_args()

    path = Path(args.jsonl)
    total_records = 0
    total_selected = 0
    tags: Counter[str] = Counter()
    dialects: Counter[str] = Counter()
    ops_total: Counter[str] = Counter()
    hlo_lines_total = 0
    hlo_chars_total = 0
    size_totals: dict[str, dict[str, float]] = {}
    compiled_memory_totals: dict[str, dict[str, float]] = {}
    cost_analysis_totals: dict[str, dict[str, float]] = {}

    for rec in _iter_jsonl(path):
        total_records += 1
        if args.only_case:
            if _get_record_key(rec, "case") != args.only_case:
                continue

        total_selected += 1
        tag = _get_record_key(rec, "tag")
        if tag:
            tags[tag] += 1

        dialect = _get_record_key(rec, "dialect")
        if dialect:
            dialects[dialect] += 1
        hlo = rec.get("hlo")
        if isinstance(hlo, str):
            ops_total.update(_count_hlo_ops(hlo))
            hlo_lines_total += _count_lines(hlo)
            hlo_chars_total += len(hlo)

        size = rec.get("size")
        if isinstance(size, dict):
            for nested_key in ("preferred_text", "stablehlo_text", "hlo_text"):
                nested = size.get(nested_key)
                if isinstance(nested, dict):
                    for metric_name, metric_value in nested.items():
                        _update_metric_totals(size_totals, f"{nested_key}.{metric_name}", metric_value)
            _update_metric_totals(size_totals, "hlo_proto_bytes", size.get("hlo_proto_bytes"))

        compiled = rec.get("compiled")
        if isinstance(compiled, dict):
            memory_analysis = compiled.get("memory_analysis")
            if isinstance(memory_analysis, dict):
                for key, value in memory_analysis.items():
                    _update_metric_totals(compiled_memory_totals, str(key), value)
            cost_analysis = compiled.get("cost_analysis")
            if isinstance(cost_analysis, dict):
                for key, value in cost_analysis.items():
                    _update_metric_totals(cost_analysis_totals, str(key), value)

    out: dict[str, Any] = {
        "jsonl": str(path),
        "total_records": total_records,
        "total_selected": total_selected,
        "filter": {"only_case": args.only_case},
        "tags": tags.most_common(),
        "dialects": dialects.most_common(),
        "hlo_text": {
            "total_lines": hlo_lines_total,
            "total_chars": hlo_chars_total,
        },
        "size": size_totals,
        "compiled_memory": compiled_memory_totals,
        "cost_analysis": cost_analysis_totals,
        "top_ops": ops_total.most_common(args.top),
    }
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
