# task

## 方針

- `task.md` は **ファイル単位**で整理します。
- 細部の設計は `documents/` に逐次追記します。

## 実装対象ファイル（ファイル単位）

- 現時点での未対応はありません。

## 次期タスク一覧（短期優先）

以下は main 側を変更せずにワークツリー／ブランチで対応するタスクです。各タスクはファイル単位で分割し、小さなブランチで実行してください。

- fix/ruff-imports: `ruff --fix` を安全に実行してインポート整備を行い、`black --line-length 100` を適用する。成果物: 修正差分、`reports/static-analysis/ruff_after_fix_run.json`。
- triage/E402_E501: `E402`（module-level import not at top）と重要な `E501`（line-too-long）をファイル単位で割り当て、手作業で修正する。修正案は `reviews/` に残す。
- type-annotations: `pyright` 出力に基づいて、型注釈を追加する小さな PR を作る。破壊的変更は避け、テストを付ける。
- tests/isolation: GPU 依存テストは `integration/gpu` に分離するか、マークで除外して通常 CI では無効化する運用案を `documents/` に提案する。
- worktree-scope: 既存の `work/*` ワークツリーで `WORKTREE_SCOPE.md` がないものを列挙し、オーナーへ scope 記入を依頼する（状態は `notes/worktrees/worktree_scope_report.md` に記録）。

実行手順（短縮）:

1. ブランチ作成: `git switch -c fix/ruff-imports-YYYYMMDD`
2. 自動修正: `ruff --fix --select I001 .` → `black --line-length 100 .`
3. ログ保存: `ruff --format json . > reports/static-analysis/ruff_after_fix_run.json`
4. コミット & プッシュ
5. `reviews/PRIORITIZED_FIXES.md` と `notes/` に結果を記載

備考: すべてのコード変更は main へ直接行わず、必ずブランチで行ってください。
