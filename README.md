# jax_util

Common Python module.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## HLO (JSONL) utility

`jax_util.hlo.dump_hlo_jsonl` で採取した JSONL を、簡易に集計・解析するスクリプトを用意しています。

```bash
python scripts/hlo/summarize_hlo_jsonl.py path/to/hlo.jsonl --top 50
```

主な出力項目:

- `total_records` / `total_selected`: レコード総数 / フィルタ後件数
- `tags`: `tag` の出現回数
- `dialects`: `dialect` の出現回数
- `hlo_text.total_lines` / `hlo_text.total_chars`: HLO テキスト総量の目安
- `top_ops`: HLO テキストから雑に抽出した op っぽいトークンの頻度
