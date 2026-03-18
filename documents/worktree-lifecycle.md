# Worktree 運用規約

この文書は、worktree の作成、使用、削除までの流れを 1 か所にまとめたものです。
他環境へ引き渡す worktree、実験専用 worktree、短期の作業用 worktree を共通に扱います。

## 1. 目的

- `main` の作業と長時間実験や大きな refactor を分離する
- branch ごとの責務を混ぜない
- worktree を消した後も、判断と結果を `main` 側に残す
- `notes/`、`diary/`、実験結果 JSON の持ち帰り方を一定にする

## 2. 作成前の確認

- `main` と対象 branch を `origin` と同期する
- `main` worktree と対象 branch の worktree が clean であることを確認する
- 既存の worktree を使い回さず、必要なら新しい worktree を切る

## 3. 作成ルール

- worktree は既定で `/workspace/.worktrees/<name>` に置く
- branch 名は目的が分かる名前にする
  - 例: `results/<topic>`
  - 例: `work/<topic>-YYYYMMDD`
- 実験結果を保存する worktree は `results/<topic>` branch に対応づける
- 一時的な開発 worktree は `work/<topic>-<date>` を原則とする

## 4. `WORKTREE_SCOPE.md`

- 他環境へ渡す worktree や、変更範囲を限定したい worktree には root に `WORKTREE_SCOPE.md` を置く
- 作業を始める前に、必ず `WORKTREE_SCOPE.md` を読む
- `WORKTREE_SCOPE.md` には少なくとも次を明記する
  - branch 名
  - worktree の目的
  - 変更してよいディレクトリ
  - 原則変更しないディレクトリ
  - 参照必須の文書
  - `main` に持ち帰る予定の `notes/` の置き場
  - その worktree 固有の追加ルール
- テンプレートは [WORKTREE_SCOPE_TEMPLATE.md](/workspace/documents/WORKTREE_SCOPE_TEMPLATE.md) を使う

## 5. 作業中のルール

- `main` の worktree では、コード編集・文書更新・通常テストのみを行う
- 長時間実験は専用 worktree で実行する
- worktree ごとに目的外の変更を混ぜない
- 実験固有のコード変更は、まず対応する results worktree で確認してから `main` へ戻す
- `main` に先に入れるのは、共通コードや規約更新のような実験非依存の変更に限る

## 6. notes / diary / 実験結果の扱い

- 長期に残す判断や考察は `main` の `notes/` へ持ち帰る
- 日付依存の作業ログは `main` の `diary/` へ残す
- 実験ごとの考察は `notes/experiments/` に置く
- worktree 固有の判断は `notes/worktrees/` に置く
- `main` に残したい要約・観測・判断のうち、規約・レビュー・実コードに属さないものは `notes/` に置く
- 複数 branch から得た一般化知見は `notes/themes/` に整理する
- 後から図や集計を再生成したい実験は、最低限の final JSON を `notes/experiments/results/` に持ち帰る
- 巨大な raw JSONL、HTML、SVG、ログは原則として results branch に残し、`main` へ常設しない
- `main` の note から raw 結果を参照するときは、本文の核心をリンク先へ逃がさず、必要最小限の JSON と要約を `main` 側にも残す
- 最新の結果は必ずmainに持ち帰る。ファイルサイズは気にしない
- worktree を削除する前に、残すべき `notes/` は `main` 側へ commit 済み、または `main` に merge 済みでなければならない

## 7. worktree を閉じる前のチェック

- その worktree で得た知見を `notes/` のどこへ残すか決めたか
- `main` に残すべき要約・観測・判断を `notes/` へ反映したか
- `diary/` に残すべき日付依存ログを分けたか
- `notes/experiments/results/` に持ち帰る final JSON を選んだか
- raw 結果を results branch に残すか、不要として捨てるかを決めたか
- branchの結果は必ずmainに持ちかえり、図、数式を用いて整理する

## 8. 削除前の整理

- worktree を消す前に、その worktree に残っている知見を `main` へ吸い出す
- 吸い出し先は `notes/worktrees/` を原則とする
- 吸い出した `notes/` は、worktree 側だけで終わらせず `main` に commit または merge してから削除へ進む
- 最低限、次を残す
  - branch 名
  - worktree の用途
  - 関連する結果やログ
  - 主要な観測
  - 次の `Idea:` / `Interpretation:` / `Consideration:`
  - `WORKTREE_SCOPE.md` があった場合は、その制約と実際の運用差分
- 実験 worktree の場合は、`notes/experiments/` と `notes/experiments/results/` の更新要否も確認する
- 公開や再利用を前提とする図は、必要なら `notes/assets/` に持ち帰る

## 9. 削除の条件

- 変更が branch に残っている
  - commit / push 済み、または残さないと判断済み
- `main` 側に残すべき note や最小限の JSON が整理済み
- `main` に残すべき `notes/` が `main` から実際に参照できる
- その worktree 固有の制約や判断が `notes/worktrees/` などから追える

## 10. 削除後

- `git worktree list` で一覧から消えたことを確認する
- 関連する branch と note の入口を `main` から辿れる状態にしておく
- 追加の作業が出たら、新しい worktree を切る
