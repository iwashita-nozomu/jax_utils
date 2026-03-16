# Path Resolution

## Markdown の画像パス

- Markdown の画像は、できるだけその file から見た相対パスで書く。
- `/workspace/...` の絶対パスは、Markdown preview で崩れることがある。
- `notes/themes/` から `notes/assets/` を見るなら `../assets/...` を使う。
- 長く残したい図は `main` の `notes/assets/` にコピーする。
- worktree の中にある画像へ直接リンクし続けない。

## Markdown のリンク

- `main` に残す note では、できるだけ `main` 側の file へリンクする。
- results worktree の file へリンクするときは、それが一時物か恒久物かを意識する。
- 外向けに読ませる note では、本文の核心をリンク先へ逃がしすぎない。

## Python 実行パス

- 基本の import root は `/workspace/python`。
- script を直接実行して import が壊れるなら、`sys.path` を足すより runner 側で `PYTHONPATH` を明示する方が分かりやすい。
- 子プロセスを起こす実験では、child に渡る `PYTHONPATH` を明示する。

## 実験結果の保存先

- 実験中の raw data は results branch に置く。
- `main` に持ち帰るのは、再集計に必要な final JSON と note。
- 画像を `main` に埋め込むなら、`notes/assets/` に置く。

## よくある失敗

- 絶対パス画像を Markdown に埋め込んで preview で出ない。
- worktree の path を `main` の恒久リンクだと思い込む。
- child process 側で `PYTHONPATH` が抜けて import が壊れる。
