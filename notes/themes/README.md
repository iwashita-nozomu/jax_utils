# Theme Notes

`notes/themes/` には、複数の実験、branch、worktree から得た知見を、話題ごとにまとめます。

- `notes/experiments/`
  - 個別実験の report と結果解釈
- `notes/branches/`
  - branch ごとの要約と入口
- `notes/worktrees/`
  - 削除済み worktree から吸い出した判断

に対し、このディレクトリでは「その話題について、いま何が分かったと言えるか」と、「実装上どんな工夫が効き、何がうまくいかなかったか」を topic 単位で再整理します。

## Format

- 1 theme 1 file を基本とします。
- 歴史の全記録ではなく、現時点の知識として再利用したい項目を優先します。
- `Known`, `Likely`, `Open` のような知識ラベルに加えて、`Worked`, `Did Not Work`, `Coding Pattern`, `Pitfall` のような実装ラベルを使ってよいです。
- うまくいかなかった案も、「なぜやめたか」「どこで詰まったか」が分かる形で残します。
- 根拠となる report や result JSON へのリンクは末尾に集約してよいです。
