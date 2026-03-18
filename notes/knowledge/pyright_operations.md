# Pyright Operations

## Config

- `pyright` の正本設定は `pyproject.toml` の `[tool.pyright]` に置く。
- `pyrightconfig.json` は重複や設定ずれの原因になるので置かない。
- baseline の対象は `python/jax_util/` のうち `neuralnetwork` と `solvers/archive` を除く実装とする。

## Baseline Run

- repo root で `pyright` を実行する。
- baseline が落ちる状態を常態化させない。
- import path の確認が必要なときも、まず repo root から実行する。

## Targeted Run

- `neuralnetwork` を触ったときは `pyright python/jax_util/neuralnetwork` を追加で回す。
- test を触ったときは `pyright python/tests/<subdir-or-file>` を追加で回す。
- 大きな変更では、baseline と touched path の両方を残す。

## Recording

- 既知の `pyright` エラーを一時的に残すなら、理由と scope を `reviews/` か `task.md` に書く。
- `ignore` を入れたときは、なぜ必要かをコード直前コメントで説明する。
- config を広げる前に、今の baseline が clean かを確認する。

## Common Failures

- `pyproject.toml` と別設定ファイルが食い違って、どちらが効いているか分からなくなる。
- repo root 以外で実行して import 解決がぶれる。
- experimental module のエラーを baseline へ混ぜて、通常開発の確認が回らなくなる。
