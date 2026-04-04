# Comprehensive Review

## Purpose

repo 全体を横断して、文書、skill、ツール、統合設定の破綻をまとめて検査します。

## Use This For

- 文書体系の棚卸し
- skill 間の重複や未整合の確認
- 自動化や integration point の確認
- 改造後の全体レビュー
- repo-wide な整理や workflow 改造の完了判定

## Inputs

- 対象ディレクトリ
- レビュー対象の phase
- 必要なら出力形式

## Outputs

- phase ごとの findings
- broken link や重複定義の一覧
- 次に片付けるべき項目の要約

## Default Placement In Workflow

- 文書整理、workflow 整理、notes 吸収の変更後 review として使います。
- repo-wide な棚卸しや構造整理では、これを正本の review とします。
- scope に experiments、benchmark protocol、report、research docs が入る場合は `agents/skills/research-perspective-review.md` を併用します。

## Implementation Surface

- `.github/skills/06-comprehensive-review/`

## Boundary

- 局所 diff のレビューだけなら `agents/skills/code-review.md` を使います。
- repo-wide review の最上位入口としては `agents/skills/project-review.md` を使います。
- 研究系の独立視点 review は `agents/skills/research-perspective-review.md` を使います。
