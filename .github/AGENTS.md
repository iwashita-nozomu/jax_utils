# Agent Team — Coordination Guide

このリポジトリ内で自動化エージェント（レビュア／ドキュメント更新／CI補助）を運用するための運用指針です。

目的:

- エージェントを役割別に整理し、作業を徹底的に実行・検証して完了まで追跡すること。

必須ロール（徹底実行）:

- coordinator: タスク割当・進捗管理を行い、すべてのエージェントを総動員して作業を完遂する。各タスクには完了判定基準（規約チェック、テスト通過、ドキュメント整備）を必須で定義する。
- reviewer: 規約に基づく徹底レビューを行う。コード品質（PEP8/型注釈）、セキュリティ、テストカバレッジ、ドキュメント整合性を確認し、不備があれば必ず差し戻す。
- integrator: ブランチ統合と衝突解消を実行し、統合後に自動テスト（ユニット・統合）と静的解析（pyright, linters）を必須で実行する。
- auditor: 統合作業とレビューの証跡（差分、テストログ、解析レポート）を収集し、監査レポートを作成する。

運用方針（徹底）:

- 各作業はワークツリー単位で行い、ワークツリー作成時に必ず `WORKTREE_SCOPE.md` を置くこと。スコープがないワークツリーは作業開始不可とする。
- 統合は小さな単位で段階的に行い、各段階で CI（pytest, pyright 等）を必須で通すこと。
- **Markdown ファイル編集後は必ず `mdformat` で書式を直してください** →
  ```bash
  mdformat path/to/file.md
  ```
  変更は原則としてワークツリー側で行い、main 側のファイルを直接編集してコンフリクトを生じさせない。main 側修正が必要な場合は integrator が調整ブランチを作り、全エージェントで検証後に適用する。

## Reviewer チェックリスト（必須）

各 PR / ワークツリー統合前に、以下を 100% 確認してください。

### コード品質

- [ ] **型注釈**: `python -m pyright ./python/jax_util` で「0 errors, 0 warnings」か確認
- [ ] **Docstring**: モジュール・クラス・public 関数に docstring（1行説明 + Args/Returns）があるか
- [ ] **テストカバレッジ**: solvers 以上のモジュール用に JSON ログ出力があるか確認
- [ ] **禁止ファイルなし**: `if __name__ == "__main__":` が本体モジュール (`./python/jax_util/`) 内にないか確認

### 規約遵守

- [ ] **Import 統一**: `from jax_util.*` （相対 import `from ..`）で統一しているか
- [ ] **循環 import なし**: 依存方向が `base → solvers → optimizers → experiment_runner` に従っているか
- [ ] **命名規約**: ファイル名・関数名が公開/非公開 (`_prefix`) で統一しているか
- [ ] **テスト命名**: `test_<module>_<function>_<scenario>` 形式か確認

### ドキュメント

- [ ] **Markdown 形式**: mdformat 後の編集ファイルはミスがないか（相対パス、リンク）
- [ ] **コメント質**: 責務コメント（`# 責務: ...`）が関数内に記載されているか

### ワークツリー・Git

- [ ] **スコープ document**: `WORKTREE_SCOPE.md` が ワークツリー root に配置されているか
- [ ] **Git commit メッセージ**: conventional commits 形式か（`feat:`, `fix:`, `docs:` など）
- [ ] **branch name**: `work/<task-name>` または `results/<experiment-name>` 形式か

### CI 実行

- [ ] **pytest 全通過**: `python -m pytest python/tests/ -q` が 0 failures か
- [ ] **pyright**: 型エラーなし
- [ ] **pydocstyle**: docstring 警告なし（許容設定）
- [ ] **ruff**: import, style エラーなし

### 条件付き

- [ ] **新規モジュール追加**: `__all__` で公開 API を明示しているか
- [ ] **実験段階コード**: neuralnetwork / hlo の場合、`__init__.py` に「実験段階」と明記しているか

再審査が必要な場合は、`review:needs-revision` ラベルを付与し、改善箇所を comment で detail に指摘してください。

報告・議論:

- すべてのエージェント実行はログを残し、`notes/` にレポートを追加すること。
- 必要な場合、エージェント間の議論を記録して意思決定の理由を残す（例: `.github/agents/discussion.md`）。
