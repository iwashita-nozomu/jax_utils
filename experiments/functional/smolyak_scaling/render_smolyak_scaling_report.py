# Results branch: results/functional-smolyak-scaling
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


STATUS_COLORS = {
    "ok": "#2f7d4a",
    "failed": "#c9573b",
    "timeout": "#d79b2e",
    "missing": "#d9d5cc",
}
PANEL_BACKGROUND = "#f7f2e8"
GRID_LINE = "#cabfae"
TEXT_COLOR = "#2b2926"
ACCENT_COLOR = "#1f5f7a"


# 責務: JSON 結果ファイルを読み込み、辞書として返す。
def load_results(path: Path, /) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


# 責務: 結果オブジェクトから整数列を昇順ユニークリストとして取り出す。
def _sorted_unique_ints(values: Sequence[object], /) -> list[int]:
    ints = sorted({int(value) for value in values if isinstance(value, int)})
    return ints


# 責務: 結果 JSON からケース辞書列を安全に取り出す。
def _cases(results: Mapping[str, object], /) -> list[Mapping[str, object]]:
    raw_cases = results.get("cases")
    if not isinstance(raw_cases, list):
        raise TypeError("results['cases'] must be a list.")
    cases: list[Mapping[str, object]] = []
    for case in raw_cases:
        if isinstance(case, Mapping):
            cases.append(case)
    return cases


# 責務: 実験結果に含まれる dtype 名列を優先順つきで返す。
def _dtype_names(results: Mapping[str, object], cases: Sequence[Mapping[str, object]], /) -> list[str]:
    raw_dtypes = results.get("dtype_names")
    if isinstance(raw_dtypes, list):
        names = [str(value) for value in raw_dtypes]
        if names:
            return names
    seen: list[str] = []
    for case in cases:
        name = str(case.get("dtype_name", "unknown"))
        if name not in seen:
            seen.append(name)
    return seen


# 責務: case 辞書から整数値を取り出せるときだけ返す。
def _maybe_int(case: Mapping[str, object], key: str, /) -> int | None:
    value = case.get(key)
    return int(value) if isinstance(value, int) else None


# 責務: case 辞書から浮動小数値を取り出せるときだけ返す。
def _maybe_float(case: Mapping[str, object], key: str, /) -> float | None:
    value = case.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


# 責務: 新旧 JSON の差を吸収しつつ可視化用メトリクス値を返す。
def _metric_value(case: Mapping[str, object], metric_name: str, /) -> float | None:
    if metric_name == "mean_abs_err":
        mean_value = _maybe_float(case, "mean_abs_err")
        return mean_value if mean_value is not None else _maybe_float(case, "abs_err")
    return _maybe_float(case, metric_name)


# 責務: dtype・level・dimension ごとのケース辞書を引ける表を構築する。
def _case_lookup(cases: Sequence[Mapping[str, object]], /) -> dict[tuple[str, int, int], Mapping[str, object]]:
    lookup: dict[tuple[str, int, int], Mapping[str, object]] = {}
    for case in cases:
        dtype_name = str(case.get("dtype_name", "unknown"))
        level = _maybe_int(case, "level")
        dimension = _maybe_int(case, "dimension")
        if level is None or dimension is None:
            continue
        lookup[(dtype_name, level, dimension)] = case
    return lookup


# 責務: 数値を人が読みやすい短い文字列へ整形する。
def _format_short_number(value: float | None, /, *, scientific_threshold: float = 1.0e-3) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    abs_value = abs(value)
    if abs_value == 0.0:
        return "0"
    if abs_value < scientific_threshold or abs_value >= 1.0e4:
        return f"{value:.2e}"
    if abs_value >= 100.0:
        return f"{value:.1f}"
    if abs_value >= 10.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


# 責務: byte 数を MiB 単位の見やすい文字列へ整形する。
def _format_mib_label(value: float | None, /) -> str:
    if value is None:
        return "-"
    return f"{value / (1024.0 * 1024.0):.2f} MiB"


# 責務: 0..1 の正規化値を落ち着いた暖色系グラデーションへ写す。
def _color_from_unit(unit: float, /) -> str:
    clipped = min(1.0, max(0.0, unit))
    red = round(245 + clipped * (180 - 245))
    green = round(237 + clipped * (92 - 237))
    blue = round(222 + clipped * (88 - 222))
    return f"#{red:02x}{green:02x}{blue:02x}"


# 責務: 凡例用の連続値カラーバー SVG を構築する。
def _render_colorbar_svg(
    *,
    title: str,
    min_label: str,
    max_label: str,
    width: int = 320,
    height: int = 54,
) -> str:
    stops = "".join(
        f'<stop offset="{offset:.0f}%" stop-color="{_color_from_unit(offset / 100.0)}" />'
        for offset in range(0, 101, 10)
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="{PANEL_BACKGROUND}" rx="12" />
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="0%">
      {stops}
    </linearGradient>
  </defs>
  <text x="16" y="18" font-size="13" font-family="monospace" fill="{TEXT_COLOR}">{html.escape(title)}</text>
  <rect x="16" y="24" width="{width - 32}" height="12" fill="url(#grad)" stroke="{GRID_LINE}" />
  <text x="16" y="50" font-size="11" font-family="monospace" fill="{TEXT_COLOR}">{html.escape(min_label)}</text>
  <text x="{width - 16}" y="50" text-anchor="end" font-size="11" font-family="monospace" fill="{TEXT_COLOR}">{html.escape(max_label)}</text>
</svg>
"""


# 責務: 数値ヒートマップに使う正規化関数と凡例ラベルを返す。
def _numeric_scale(values: Sequence[float], /, *, log_scale: bool) -> tuple[float, float, str, str]:
    positive_values = [value for value in values if math.isfinite(value)]
    if not positive_values:
        return 0.0, 1.0, "-", "-"
    if log_scale:
        transformed = [math.log10(max(value, 1.0e-300)) for value in positive_values]
        vmin = min(transformed)
        vmax = max(transformed)
        return vmin, vmax, f"log10={vmin:.2f}", f"log10={vmax:.2f}"
    vmin = min(positive_values)
    vmax = max(positive_values)
    return vmin, vmax, _format_short_number(vmin), _format_short_number(vmax)


# 責務: 数値をスケールに応じて 0..1 へ正規化する。
def _normalize_value(value: float, vmin: float, vmax: float, /, *, log_scale: bool) -> float:
    transformed = math.log10(max(value, 1.0e-300)) if log_scale else value
    if vmax <= vmin:
        return 0.5
    return (transformed - vmin) / (vmax - vmin)


# 責務: セルの値ラベルを見やすい文字列へ整形する。
def _format_metric_label(metric_name: str, value: float | None, /) -> str:
    if value is None:
        return "-"
    if metric_name == "storage_bytes":
        return _format_mib_label(value)
    if metric_name == "avg_integral_seconds":
        return f"{value * 1.0e6:.1f} us"
    return _format_short_number(value)


# 責務: dtype ごとの数値ヒートマップ SVG を構築する。
def render_numeric_heatmap(
    *,
    title: str,
    metric_name: str,
    cases: Sequence[Mapping[str, object]],
    dtype_names: Sequence[str],
    dimensions: Sequence[int],
    levels: Sequence[int],
    log_scale: bool,
) -> str:
    lookup = _case_lookup(cases)
    values = [
        value
        for case in cases
        if case.get("status") == "ok"
        for value in [_metric_value(case, metric_name)]
        if value is not None and value > 0.0
    ]
    vmin, vmax, min_label, max_label = _numeric_scale(values, log_scale=log_scale)

    cell_w = 92
    cell_h = 34
    left_margin = 88
    top_margin = 52
    panel_gap = 28
    panel_h = top_margin + cell_h * len(levels) + 24
    width = left_margin + cell_w * len(dimensions) + 24
    height = panel_h * len(dtype_names) + panel_gap * max(0, len(dtype_names) - 1) + 26
    show_labels = len(dimensions) * len(levels) <= 64

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{PANEL_BACKGROUND}" />',
        f'<text x="18" y="24" font-size="18" font-family="monospace" fill="{TEXT_COLOR}">{html.escape(title)}</text>',
        f'<text x="{width - 18}" y="24" text-anchor="end" font-size="12" font-family="monospace" fill="{ACCENT_COLOR}">{html.escape(min_label)} -> {html.escape(max_label)}</text>',
    ]

    for dtype_index, dtype_name in enumerate(dtype_names):
        panel_y = 34 + dtype_index * (panel_h + panel_gap)
        svg.append(
            f'<rect x="10" y="{panel_y}" width="{width - 20}" height="{panel_h - 6}" rx="14" fill="#fffdfa" stroke="{GRID_LINE}" />'
        )
        svg.append(
            f'<text x="22" y="{panel_y + 20}" font-size="14" font-family="monospace" fill="{TEXT_COLOR}">dtype={html.escape(dtype_name)}</text>'
        )

        for col, dimension in enumerate(dimensions):
            x = left_margin + col * cell_w
            svg.append(
                f'<text x="{x + cell_w / 2:.1f}" y="{panel_y + 38}" text-anchor="middle" font-size="11" font-family="monospace" fill="{TEXT_COLOR}">d={dimension}</text>'
            )
        for row, level in enumerate(levels):
            y = panel_y + top_margin + row * cell_h
            svg.append(
                f'<text x="{left_margin - 14}" y="{y + 22}" text-anchor="end" font-size="11" font-family="monospace" fill="{TEXT_COLOR}">l={level}</text>'
            )
            for col, dimension in enumerate(dimensions):
                x = left_margin + col * cell_w
                case = lookup.get((dtype_name, level, dimension))
                value = _metric_value(case, metric_name) if case is not None and case.get("status") == "ok" else None
                if value is None or value <= 0.0:
                    fill = STATUS_COLORS["missing"] if case is None else STATUS_COLORS.get(str(case.get("status")), STATUS_COLORS["failed"])
                else:
                    fill = _color_from_unit(_normalize_value(value, vmin, vmax, log_scale=log_scale))
                svg.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" fill="{fill}" stroke="{GRID_LINE}" rx="6" />'
                )
                if show_labels:
                    label = _format_metric_label(metric_name, value)
                    svg.append(
                        f'<text x="{x + (cell_w - 4) / 2:.1f}" y="{y + 20}" text-anchor="middle" font-size="9" font-family="monospace" fill="{TEXT_COLOR}">{html.escape(label)}</text>'
                    )

    svg.append("</svg>")
    return "\n".join(svg)


# 責務: dtype ごとの実行状態ヒートマップ SVG を構築する。
def render_status_heatmap(
    *,
    cases: Sequence[Mapping[str, object]],
    dtype_names: Sequence[str],
    dimensions: Sequence[int],
    levels: Sequence[int],
) -> str:
    lookup = _case_lookup(cases)
    cell_w = 92
    cell_h = 34
    left_margin = 88
    top_margin = 52
    panel_gap = 28
    panel_h = top_margin + cell_h * len(levels) + 24
    width = left_margin + cell_w * len(dimensions) + 24
    height = panel_h * len(dtype_names) + panel_gap * max(0, len(dtype_names) - 1) + 56

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{PANEL_BACKGROUND}" />',
        f'<text x="18" y="24" font-size="18" font-family="monospace" fill="{TEXT_COLOR}">execution status</text>',
    ]
    legend_x = 18
    for index, (status, color) in enumerate((("ok", STATUS_COLORS["ok"]), ("failed", STATUS_COLORS["failed"]), ("timeout", STATUS_COLORS["timeout"]), ("missing", STATUS_COLORS["missing"]))):
        x = legend_x + index * 90
        svg.append(f'<rect x="{x}" y="32" width="14" height="14" fill="{color}" stroke="{GRID_LINE}" rx="3" />')
        svg.append(f'<text x="{x + 20}" y="43" font-size="11" font-family="monospace" fill="{TEXT_COLOR}">{status}</text>')

    for dtype_index, dtype_name in enumerate(dtype_names):
        panel_y = 52 + dtype_index * (panel_h + panel_gap)
        svg.append(
            f'<rect x="10" y="{panel_y}" width="{width - 20}" height="{panel_h - 6}" rx="14" fill="#fffdfa" stroke="{GRID_LINE}" />'
        )
        svg.append(
            f'<text x="22" y="{panel_y + 20}" font-size="14" font-family="monospace" fill="{TEXT_COLOR}">dtype={html.escape(dtype_name)}</text>'
        )
        for col, dimension in enumerate(dimensions):
            x = left_margin + col * cell_w
            svg.append(
                f'<text x="{x + cell_w / 2:.1f}" y="{panel_y + 38}" text-anchor="middle" font-size="11" font-family="monospace" fill="{TEXT_COLOR}">d={dimension}</text>'
            )
        for row, level in enumerate(levels):
            y = panel_y + top_margin + row * cell_h
            svg.append(
                f'<text x="{left_margin - 14}" y="{y + 22}" text-anchor="end" font-size="11" font-family="monospace" fill="{TEXT_COLOR}">l={level}</text>'
            )
            for col, dimension in enumerate(dimensions):
                x = left_margin + col * cell_w
                case = lookup.get((dtype_name, level, dimension))
                status = "missing" if case is None else str(case.get("status", "failed"))
                fill = STATUS_COLORS.get(status, STATUS_COLORS["failed"])
                svg.append(
                    f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" fill="{fill}" stroke="{GRID_LINE}" rx="6" />'
                )
                svg.append(
                    f'<text x="{x + (cell_w - 4) / 2:.1f}" y="{y + 20}" text-anchor="middle" font-size="9" font-family="monospace" fill="{TEXT_COLOR}">{html.escape(status)}</text>'
                )

    svg.append("</svg>")
    return "\n".join(svg)


# 責務: dtype ごとの成功 frontier をレベル対最大成功次元で可視化する。
def render_frontier_svg(
    *,
    cases: Sequence[Mapping[str, object]],
    dtype_names: Sequence[str],
    levels: Sequence[int],
    dimensions: Sequence[int],
) -> str:
    palette = ["#1f5f7a", "#c9573b", "#7d6b2f", "#6d4ea1"]
    ok_cases = [case for case in cases if case.get("status") == "ok"]
    max_dimension = max(dimensions) if dimensions else 1
    min_level = min(levels) if levels else 1
    max_level = max(levels) if levels else 1

    width = 760
    height = 360
    left = 72
    right = 24
    top = 28
    bottom = 48
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_coord(level: int) -> float:
        if max_level == min_level:
            return left + plot_w / 2.0
        return left + (level - min_level) * plot_w / (max_level - min_level)

    def y_coord(dimension: int) -> float:
        if max_dimension <= 1:
            return top + plot_h / 2.0
        return top + plot_h - (dimension - 1) * plot_h / (max_dimension - 1)

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{PANEL_BACKGROUND}" />',
        f'<text x="18" y="22" font-size="18" font-family="monospace" fill="{TEXT_COLOR}">success frontier by dtype</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fffdfa" stroke="{GRID_LINE}" />',
    ]

    for level in levels:
        x = x_coord(level)
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="{GRID_LINE}" stroke-dasharray="3 4" />')
        svg.append(f'<text x="{x:.1f}" y="{height - 18}" text-anchor="middle" font-size="11" font-family="monospace" fill="{TEXT_COLOR}">{level}</text>')
    for dimension in dimensions:
        y = y_coord(dimension)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="{GRID_LINE}" stroke-dasharray="3 4" />')
        svg.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-size="11" font-family="monospace" fill="{TEXT_COLOR}">{dimension}</text>')

    svg.append(f'<text x="{left + plot_w / 2:.1f}" y="{height - 4}" text-anchor="middle" font-size="12" font-family="monospace" fill="{TEXT_COLOR}">level</text>')
    svg.append(f'<text x="20" y="{top + plot_h / 2:.1f}" transform="rotate(-90 20 {top + plot_h / 2:.1f})" text-anchor="middle" font-size="12" font-family="monospace" fill="{TEXT_COLOR}">max success dimension</text>')

    for dtype_index, dtype_name in enumerate(dtype_names):
        color = palette[dtype_index % len(palette)]
        frontier_points: list[tuple[int, int]] = []
        for level in levels:
            success_dimensions = [
                dimension
                for case in ok_cases
                for dimension in [_maybe_int(case, "dimension")]
                if case.get("dtype_name") == dtype_name and case.get("level") == level and dimension is not None
            ]
            if success_dimensions:
                frontier_points.append((level, max(success_dimensions)))
        if not frontier_points:
            continue
        point_string = " ".join(f"{x_coord(level):.1f},{y_coord(dimension):.1f}" for level, dimension in frontier_points)
        svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{point_string}" />')
        for level, dimension in frontier_points:
            svg.append(f'<circle cx="{x_coord(level):.1f}" cy="{y_coord(dimension):.1f}" r="4" fill="{color}" />')
        legend_y = 26 + dtype_index * 18
        legend_x = width - 160
        svg.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 20}" y2="{legend_y}" stroke="{color}" stroke-width="3" />')
        svg.append(f'<text x="{legend_x + 28}" y="{legend_y + 4}" font-size="11" font-family="monospace" fill="{TEXT_COLOR}">{html.escape(dtype_name)}</text>')

    svg.append("</svg>")
    return "\n".join(svg)


# 責務: 実験結果の概要を HTML でまとめて、生成図への導線を作る。
def render_index_html(
    *,
    results: Mapping[str, object],
    dtype_names: Sequence[str],
    figures: Sequence[tuple[str, str]],
) -> str:
    meta_rows = [
        ("experiment", str(results.get("experiment", "-"))),
        ("started_at_utc", str(results.get("started_at_utc", "-"))),
        ("platform", str(results.get("platform", "-"))),
        ("num_cases", str(results.get("num_cases", "-"))),
        ("num_repeats", str(results.get("num_repeats", "-"))),
        ("num_accuracy_problems", str(results.get("num_accuracy_problems", "-"))),
        ("dtypes", ", ".join(dtype_names)),
    ]
    rows_html = "\n".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>"
        for key, value in meta_rows
    )
    figures_html = "\n".join(
        f'<section><h2>{html.escape(title)}</h2><img src="{html.escape(path)}" alt="{html.escape(title)}" /></section>'
        for title, path in figures
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Smolyak Scaling Report</title>
  <style>
    body {{ font-family: "Iosevka", "Menlo", monospace; margin: 24px; color: {TEXT_COLOR}; background: {PANEL_BACKGROUND}; }}
    table {{ border-collapse: collapse; margin-bottom: 24px; background: #fffdfa; }}
    th, td {{ border: 1px solid {GRID_LINE}; padding: 8px 12px; text-align: left; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    section {{ margin-bottom: 24px; }}
    img {{ max-width: 100%; height: auto; border: 1px solid {GRID_LINE}; background: #fffdfa; }}
  </style>
</head>
<body>
  <h1>Smolyak Scaling Report</h1>
  <table>
    {rows_html}
  </table>
  {figures_html}
</body>
</html>
"""


# 責務: 結果 JSON から可視化一式を生成して出力先へ保存する。
def generate_report(input_path: Path, output_dir: Path, /) -> None:
    results = load_results(input_path)
    cases = _cases(results)
    dtype_names = _dtype_names(results, cases)
    dimensions = _sorted_unique_ints([case.get("dimension") for case in cases])
    levels = _sorted_unique_ints([case.get("level") for case in cases])

    output_dir.mkdir(parents=True, exist_ok=True)
    figures: list[tuple[str, str]] = []

    figure_specs = [
        ("status heatmap", "status.svg", render_status_heatmap(cases=cases, dtype_names=dtype_names, dimensions=dimensions, levels=levels)),
        ("mean absolute error heatmap", "mean_abs_err.svg", render_numeric_heatmap(title="mean absolute error", metric_name="mean_abs_err", cases=cases, dtype_names=dtype_names, dimensions=dimensions, levels=levels, log_scale=True)),
        ("absolute error variance heatmap", "var_abs_err.svg", render_numeric_heatmap(title="absolute error variance", metric_name="var_abs_err", cases=cases, dtype_names=dtype_names, dimensions=dimensions, levels=levels, log_scale=True)),
        ("average time heatmap", "avg_integral_seconds.svg", render_numeric_heatmap(title="average integral time", metric_name="avg_integral_seconds", cases=cases, dtype_names=dtype_names, dimensions=dimensions, levels=levels, log_scale=True)),
        ("storage heatmap", "storage_bytes.svg", render_numeric_heatmap(title="storage bytes", metric_name="storage_bytes", cases=cases, dtype_names=dtype_names, dimensions=dimensions, levels=levels, log_scale=True)),
        ("success frontier", "frontier.svg", render_frontier_svg(cases=cases, dtype_names=dtype_names, levels=levels, dimensions=dimensions)),
    ]

    for title, filename, content in figure_specs:
        (output_dir / filename).write_text(content, encoding="utf-8")
        figures.append((title, filename))

    colorbar_svg = _render_colorbar_svg(title="numeric heatmaps", min_label="lighter = smaller", max_label="darker = larger")
    (output_dir / "legend.svg").write_text(colorbar_svg, encoding="utf-8")
    figures.insert(0, ("legend", "legend.svg"))

    index_html = render_index_html(results=results, dtype_names=dtype_names, figures=figures)
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render SVG/HTML figures from a Smolyak scaling JSON result.")
    parser.add_argument("--input", type=Path, required=True, help="Path to a result JSON file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to place the generated report.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    default_output_dir = input_path.parent / f"{input_path.stem}_report"
    output_dir = (args.output_dir or default_output_dir).resolve()
    generate_report(input_path, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
