# jax_util

Common Python module.

## Tools & Workflows

**ツール・スクリプト一覧と使用方法：**
- [documents/tools/TOOLS_DIRECTORY.md](/workspace/documents/tools/TOOLS_DIRECTORY.md) — ツール詳細目録（20個のスクリプト・ツール対応）
- [documents/FILE_CHECKLIST_OPERATIONS.md](/workspace/documents/FILE_CHECKLIST_OPERATIONS.md) — **作業別チェックリスト・実行手順書（8個フロー対応）**

**推奨される作業開始フロー：**
```bash
# 1. 新規ブランチ作成
bash scripts/tools/create_worktree.sh my-feature-name

# 2. スコープ編集＆コミット
cd .worktrees/my-feature-name
vim WORKTREE_SCOPE.md
git add WORKTREE_SCOPE.md && git commit -m "chore(worktree): initialize scope"

# 3. 実装＆テスト
# ... code changes ...
bash ../scripts/run_pytest_with_logs.sh
pyright python/

# 4. ドキュメント更新
python ../scripts/tools/format_markdown.py
python ../scripts/tools/audit_and_fix_links.py
```

詳細は [FILE_CHECKLIST_OPERATIONS.md](./documents/FILE_CHECKLIST_OPERATIONS.md) の**チェックリスト1～8**を参照してください。

## Notes

- 軽量な実験メモは [notes/README.md](/workspace/notes/README.md) に置きます。
- topic ごとの知識整理は [notes/themes/README.md](/workspace/notes/themes/README.md) に置きます。
- 実務向けの横断的な知識メモは [notes/knowledge/README.md](/workspace/notes/knowledge/README.md) に置きます。
- branch ごとのメモ入口は [notes/branches/README.md](/workspace/notes/branches/README.md) に置きます。
- 日付ごとの作業ログは [diary/README.md](/workspace/diary/README.md) に置きます。
- 規約や設計の一次情報は引き続き [documents/README.md](/workspace/documents/README.md) を参照します。

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
