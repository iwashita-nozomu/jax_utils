# AGENTS.md

## Project overview

`jax_util` is a common Python module.

This repository contains reusable Python utilities, scripts, and supporting notes around JAX-related workflows.  
When making changes, prefer small, local, well-validated edits that preserve the existing structure and developer workflow.

## Primary references

Consult these entry points first when a task touches notes, process, or design context:

- Lightweight experimental notes: `notes/README.md`
- Topic-based knowledge organization: `notes/themes/README.md`
- Cross-cutting practical knowledge: `notes/knowledge/README.md`
- Branch-specific note entry points: `notes/branches/README.md`
- Date-based work logs: `diary/README.md`
- Primary source for conventions and design documents: `documents/README.md`

If a task produces documentation or knowledge capture, update the appropriate existing location instead of adding ad hoc markdown files elsewhere.

## Repository map

Use this as the default navigation guide:

- `jax_util/`: main Python package
- `tests/`: automated tests
- `scripts/`: utility scripts and CLI helpers
- `notes/`: lightweight notes, experiments, thematic summaries, branch memos
- `diary/`: dated work logs
- `documents/`: formal design docs, conventions, and primary references

## Development setup

Install development dependencies:

```bash
pip install -e ".[dev]"
Core validation commands

Run the full test suite:

pytest

Run lint:

ruff check .

When changing Python code, run at least the relevant tests plus ruff check . before considering the task complete.

Definition of done

A coding task is complete only when all of the following are true:

The requested change is implemented with minimal, focused edits.

Relevant tests pass locally.

ruff check . passes.

Any behavior, usage, or output changes are reflected in the appropriate docs or notes.

Bug fixes and parsing/aggregation changes include or update tests when practical.

If validation could not be run, that is stated explicitly rather than implied.

Preferred workflow

For most tasks, follow this order:

Read the relevant module and nearby tests first.

Check documents/README.md when design intent or conventions may matter.

Make the smallest viable change.

Run relevant tests.

Run ruff check .

Update docs or notes if behavior, usage, or operational knowledge changed.

Coding guidelines
General

Prefer small, explicit, maintainable functions over clever abstractions.

Preserve backward compatibility unless the task explicitly requires a breaking change.

Avoid unnecessary dependency additions.

Do not rename public functions, modules, or CLI flags unless required.

Follow existing repository patterns before introducing new ones.

Keep changes scoped to the request; do not mix unrelated cleanup into the same edit.

Python

Keep implementations simple and readable.

Use type hints where they improve clarity or match surrounding code.

Reuse existing helpers before introducing new utility layers.

Handle expected failure modes explicitly.

Avoid overly broad exception handling.

Prefer observable, testable behavior over implicit side effects.

Scripts and CLIs

Scripts under scripts/ should remain easy to run from the command line.

Preserve existing CLI behavior unless explicitly asked to change it.

If adding flags or options, keep names and help text straightforward and consistent.

Prefer additive output changes over incompatible format changes unless a breaking change is requested.

Testing expectations

Add or update tests when changing:

parsing behavior

filtering behavior

aggregation or counting logic

CLI output structure

edge-case handling for missing or malformed input

public API behavior

For data-processing utilities, prefer tests that validate observable outputs rather than implementation details.

For bug fixes, add a regression test when practical.

Documentation and note placement rules

Do not create random top-level memo files.

Choose the destination based on purpose:

quick experimental memo → notes/

topic or theme knowledge organization → notes/themes/

practical cross-cutting knowledge → notes/knowledge/

branch-specific memo → notes/branches/

dated work log → diary/

formal rule or design documentation → documents/

If the work produces durable operational knowledge, prefer updating an existing README in these directories before creating a new file.

HLO (JSONL) utility guidance

This repository includes utilities for analyzing JSONL captured by jax_util.hlo.dump_hlo_jsonl.

Current summary script:

python scripts/hlo/summarize_hlo_jsonl.py path/to/hlo.jsonl --top 50

Current key output fields include:

total_records / total_selected

tags

dialects

hlo_text.total_lines / hlo_text.total_chars

top_ops

When extending this area:

prefer additive metrics over incompatible output changes

keep terminal output easy to inspect

make token or op extraction heuristics explicit in code comments

separate parsing, aggregation, and presentation logic where practical

preserve usefulness for large JSONL inputs

handle missing or partial fields gracefully

If summary semantics change, update the relevant documentation or notes.

File-specific guidance
When editing scripts/hlo/summarize_hlo_jsonl.py

Preserve current CLI usage unless explicitly asked to change it.

Prefer additive metrics over incompatible output changes.

Keep memory usage reasonable for large JSONL files.

If parsing logic becomes more complex, extract helper functions and test them directly.

Keep output fields stable when possible so downstream ad hoc usage does not break unexpectedly.

When editing code under jax_util/hlo/

Preserve compatibility with JSONL already emitted by existing tooling when possible.

Document schema assumptions in code comments or docs when they are non-obvious.

Handle missing values gracefully for newly introduced fields.

Keep emitted data easy to summarize and inspect from scripts.

Boundaries

Unless explicitly requested, do not:

restructure the package extensively

introduce heavyweight dependencies

change public APIs unnecessarily

edit unrelated notes or documents

mix formal documentation with temporary investigation notes

add speculative optimizations without measurement or clear justification

change CLI output formats in a breaking way

manually edit generated artifacts or vendored code

Change hygiene

When reporting completed work, include:

what changed

where it changed

how it was validated

any follow-up work that remains

If something could not be validated, say so clearly.

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
