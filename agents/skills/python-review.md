# Python Review

## Purpose

Python 差分を、静的解析、型追跡、warning の扱いを中心に独立レビューします。

## Use This For

- `python/` 配下の変更レビュー
- `pyright` warning / error の解消確認
- 型境界、戻り値、`dtype`、`Protocol` まわりの追跡
- `cast`、`Any`、`object`、`ignore` に頼った実装の検出
- `ruff`、docstring、test 整合を含む Python 固有の review

## Must Read Before Reviewing

- `documents/coding-conventions-python.md`
- `documents/conventions/python/04_type_annotations.md`
- `documents/conventions/python/07_type_checker.md`
- `documents/coding-conventions-project.md`
- `pyproject.toml`
- 対象 diff
- 必要なら対象 module の test

## Inputs

- diff または対象 Python ファイル
- 受け入れ条件
- `pyright` / `ruff` / `pytest` の結果
- 関連 docs / tests

## Outputs

- findings-first の review
- static analysis と type trace に基づく required change
- warning を残してよいかどうかの明示判定

## Mandatory Checklist

- 触った Python path に対する `pyright` の結果を確認する
- test を触った場合は test path への `pyright` 追加実行を確認する
- `ruff check` の warning を放置していない
- 引数と戻り値の型が意味のある型で書かれている
- `Scalar` / `Vector` / `Matrix` と型境界の規約を破っていない
- `dtype` の受け渡しが公開 API / 内部実装の規約に沿っている
- `cast`、`Any`、`object`、`pyright: ignore`、`# type: ignore` を安易に増やしていない
- 変更した型が呼び出し元と呼び出し先まで追跡されている
- docstring、test、実装の三点が矛盾していない

## Review Stance

- Python diff では、見た目より `pyright` と `ruff` を優先して疑います。
- warning は「あとで直す」前提で流さず、その warning が設計上許容かを詰めます。
- 型を追い切れない変更は、動きそうでも止めます。
- `ignore` や `cast` は最後の手段であり、説明なしの追加を許可しません。

## Default Commands

- baseline module review: `pyright`
- touched test path: `pyright python/tests/<subdir-or-file>`
- focused module review: `pyright python/jax_util/<module>`
- lint review: `ruff check <target>`

## Boundary

- 言語非依存の diff review は `agents/skills/code-review.md` を使います。
- 速い機械検査だけなら `agents/skills/static-check.md` を使います。
- 実験結果や report の妥当性は `critical-review` や `report-review` を使います。

## Implementation Surface

- `.github/skills/19-python-review/`
