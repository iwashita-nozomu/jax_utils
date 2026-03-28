# `experiment_runner` 終了コード

この文書は、`python/experiment_runner/protocols.py` における
`Worker.__call__(case, context) -> int` の意味を整理します。

## 1. 基本方針

- `task: Callable[[T, TaskContext], U]` の `U` は論理結果です。
- 現在の最小 `StandardWorker` では、この `U` を runner へ返しません。
- runner / scheduler が見るのは `Worker.__call__(...) -> int` の終了コードだけです。

## 2. 最小の終了コード

- `0`
  - worker が task を正常に完了したことを表します。
- `1`
  - worker が例外により task を完了できなかったことを表します。

## 3. scheduler / runner 側の扱い

- scheduler は `on_finish(case, context, exit_code)` で完了通知を受けます。
- per-case の失敗は終了コード `1` として回収し、runner 自体は継続する方針です。
- child process の traceback は worker が標準エラーへ出すことがありますが、runner が停止することとは分けて扱います。
