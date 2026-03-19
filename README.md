# jax_util

Common Python module.

## Notes

- 軽量な実験メモは [notes/README.md](/workspace/notes/README.md) に置きます。
- topic ごとの知識整理は [notes/themes/README.md](/workspace/notes/themes/README.md) に置きます。
- 実務向けの横断的な知識メモは [notes/knowledge/README.md](/workspace/notes/knowledge/README.md) に置きます。
- branch ごとのメモ入口は [notes/branches/README.md](/workspace/notes/branches/README.md) に置きます。
- 日付ごとの作業ログは [diary/README.md](/workspace/diary/README.md) に置きます。
- 規約や設計の一次情報は引き続き [documents/README.md](/workspace/documents/README.md) を参照します。

- 現在の作業タスク一覧は `task.md` に記載しています。

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
- `top_ops`: HLO テキストから雑に抽出した op っぽいトークンの頻度 これをベースに拡充して

## 次作業者向け手順 (Next worker)

目的: 本リポジトリの静的解析修正作業をワークツリー/ブランチ単位で安全に進め、証跡を残すための最短手順。

1) ブランチ方針（重要）
 - main の直接編集は禁止。すべてワークツリーまたは feature ブランチで作業すること。
 - 自動修正は小さなブランチで実行し、結果を `reports/` に保存すること。

2) 優先される作業（まず行うこと）
 - 参照: `reviews/PRIORITIZED_FIXES.md`（優先ファイル・手順を記載）
 - 自動修正対象: インポート整理 (`I001`) を中心に `ruff --fix` を使う。

3) 実行コマンド（例）

```bash
# ブランチを作成
git switch -c fix/ruff-imports-$(date +%Y%m%d)

# 自動修正（安全な修正のみ）
ruff --fix --select I001 .

# フォーマット
black --line-length 100 .

# 再解析とログ保存
ruff --format json . > reports/static-analysis/ruff_after_fix_run.json
pyright > reports/static-analysis/pyright_after_fix_run.txt || true
pytest -q || true

# 変更をコミットしてプッシュ
git add -A
git commit -m "fix: ruff import/formatting fixes (automatic)"
git push -u origin HEAD
```

4) 記録とレビュー
 - 自動修正後は `reviews/PRIORITIZED_FIXES.md` に該当ファイルと結果（レポートへのリンク）を追記する。
 - 手作業が必要な `E402`/`E501` はファイル毎に担当者を割当て、`reviews/` に修正案を残すこと。

5) CI と報告
 - 各修正ブランチで CI を回し、`reports/` に結果を保存する。
 - `notes/` または `diary/` に簡潔な作業ログ（何を直したか、検証結果、残課題）を追記する。

6) ワークツリーのスコープ確認
 - 新しいワークツリーを作る際は必ず `WORKTREE_SCOPE.md` を配置すること。
 - スコープが未定義の既存ワークツリーは作業前に `notes/worktrees/worktree_scope_report.md` を更新して状態を明示する。

問い合わせ先 / 参照
 - 優先修正一覧: `reviews/PRIORITIZED_FIXES.md`
 - 静的解析レポート: `reports/static-analysis/`
 - エージェント運用方針: `.github/AGENTS.md`

――
この手順を守って作業してください。疑問点は `notes/` に記録し、`auditor` ロールに通知してください。
