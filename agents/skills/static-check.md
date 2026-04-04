# Static Check

## Purpose

実装変更の直後に、速く回せる基礎検査をまとめて扱います。

## Use This For

- 型エラーの早期検出
- テスト失敗の早期検出
- Markdown や文書整合の基礎確認
- Docker / 実行環境の破綻確認

## Inputs

- 変更対象ファイル
- 必要なら対象ディレクトリや対象コマンド

## Outputs

- pass / fail の要約
- 修正が必要なファイル一覧
- 実行コマンドと最小再現情報

## Implementation Surface

- `.github/skills/01-static-check/`
- `.github/skills/09-type-checking/`
- `.github/skills/10-documentation-validation/`
- `.github/skills/11-test-execution/`
- `.github/skills/15-docker-environment/`

## Boundary

- 深い設計レビューや実験解釈は扱いません。
- 変更レビューが必要なら `agents/skills/code-review.md` を使います。
