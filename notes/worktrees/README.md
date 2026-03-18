# Worktree Extraction Notes

このディレクトリには、削除予定の worktree から `main` へ吸い上げたメモを置きます。

## Purpose

- 削除前の worktree にしか残っていない知見を `main` 側へ退避する
- 実験 branch や tuning branch の一時的な判断を、後で参照できる形にする
- worktree 自体は消しても、判断の履歴は失わないようにする

## What To Extract

最低限、次を整理してから worktree を消します。

- branch 名
- worktree の用途
- 関連する results branch や experiment branch
- 主要な結果ファイルやログの所在
- 観測したボトルネックや failure pattern
- 次に残すべき `Idea:` / `Interpretation:` / `Consideration:`
- Dockerfile 更新候補や、解析に使ったツールのメモ
- その worktree を一度しか開かなくても状況が分かる quick reference

## Naming

- ファイル名は `worktree_<topic>_YYYY-MM-DD.md` のように、対象が分かる形にします。
- 1 つの file に unrelated な worktree を混ぜません。
- 文書タイトルや先頭 H1 は、日付ではなく worktree のトピックを主にして付けます。
- 日付は file 名や metadata に残し、タイトル自体は `Smolyak Tuning Worktree Extraction` のように topic-first にします。

## Reading Principle

- worktree note はリンク集だけで終わらせず、その file 単体で branch の役割、主要変更、重要結果、最終判断が読めるようにします。
- 詳細ログや raw 結果へのリンクは補助として置きますが、核心の結論は本文に残します。

## Notes

- [worktree_results_smolyak_scaling_tuned_2026-03-18.md](/workspace/notes/worktrees/worktree_results_smolyak_scaling_tuned_2026-03-18.md)

## Source And Idea Marking

- 文献や既存結果から得た内容は、対象の文献や結果ファイルを明示します。
- 自分の発案は `Idea:`、結果の読みは `Interpretation:` や `Consideration:` で区別します。
