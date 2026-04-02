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
- 既存の worktree を使い回しません。新しい worktree を切ります。

## 3. 作成ルール

- worktree は既定で `/workspace/.worktrees/<name>` に置く
- branch 名は目的が分かる名前にする
  - 例: `results/<topic>`
  - 例: `work/<topic>-YYYYMMDD`
- 実験結果を保存する worktree は `results/<topic>` branch に対応づける
- 一時的な開発 worktree は `work/<topic>-<date>` を使う

## 4. `WORKTREE_SCOPE.md`

- 他環境へ渡す worktree や、変更範囲を限定したい worktree には root に `WORKTREE_SCOPE.md` を置く
- 作業を始める前に、必ず `WORKTREE_SCOPE.md` を読む
- `WORKTREE_SCOPE.md` には少なくとも次を明記する
  - branch 名
  - worktree の目的
  - 変更を許可するディレクトリ
  - 実験実行 role が書いてよい runtime output ディレクトリ
  - 変更を禁止するディレクトリ
  - 参照必須の文書
  - `main` に持ち帰る予定の `notes/` の置き場
  - 作業中に追記する action log の置き場
  - その worktree 固有の追加ルール
- テンプレートは [WORKTREE_SCOPE_TEMPLATE.md](/workspace/documents/WORKTREE_SCOPE_TEMPLATE.md) を使う

## 5. 作業中のルール

- `main` の worktree では、コード編集・文書更新・通常テストのみを行う
- 長時間実験は専用 worktree で実行する
- worktree ごとに目的外の変更を混ぜない
- 実験固有のコード変更は、まず対応する results worktree で確認してから `main` へ戻す
- `main` に先に入れるのは、共通コードや規約更新のような実験非依存の変更に限る
- 作業中の「一挙手一投足」は、必ず 1 か所の append-only な action log に残す
- 研究・実験改造では、問い、定式化、比較対象、各改造の狙いも action log に残す
- 実験 run は 1 回の fresh 実行で完走させる前提とし、途中停止した run をそのまま再開しない
- action log は、意味のある単位ごとに短く追記する
  - 例: scope 更新、編集開始、テスト実行、実験開始、実験停止、carry-over 判断、branch 統合
- action log では、`Question:`、`Formulation:`、`Comparison Target:`、`Change:`、`Decision:`、`Branch Reflection:` を明示できます
- action log の既定位置は `notes/worktrees/worktree_<topic>_YYYY-MM-DD.md` とする
- worktree 内で下書きするときも、最終配置と同じ相対パスに置く
  - 例: `.worktrees/<name>/notes/worktrees/worktree_<topic>_YYYY-MM-DD.md`
  - 例: `.worktrees/<name>/notes/experiments/<topic>.md`
  - 例: `.worktrees/<name>/notes/branches/<branch_topic>.md`
- `experimenter` を使う task では、`WORKTREE_SCOPE.md` の `## Runtime Output Directories` を必須にする

## 6. notes / diary / 実験結果の扱い

- 長期に残す判断や考察は `main` の `notes/` へ持ち帰る
- 日付依存の作業ログは `main` の `diary/` へ残す
- 実験ごとの考察は `notes/experiments/` に置く
- worktree 固有の判断は `notes/worktrees/` に置く
- `main` に残したい要約・観測・判断のうち、規約・レビュー・実コードに属さないものは `notes/` に置く
- 複数 branch から得た一般化知見は `notes/themes/` に整理する
- 後から図や集計を再生成したい実験は、最低限の final JSON を `notes/experiments/results/` に持ち帰る
- 巨大な raw JSONL、HTML、SVG、ログは results branch に残します。`main` へ常設することを禁止します
- `main` の note から raw 結果を参照するときは、本文の核心をリンク先へ逃がさず、必要最小限の JSON と要約を `main` 側にも残す
- `main` へ持ち帰るのは、再解析に必要な最小 final JSON と、その意味を説明する note だけにします
- 途中停止した partial run は carry-over の正本にしません。停止理由の診断材料として results branch 側へ残します
- worktree を削除する前に、残すべき `notes/` は `main` 側へ commit 済み、または `main` に merge 済みでなければならない

## 7. worktree を閉じる前のチェック

- その worktree で得た知見を `notes/` のどこへ残すか決めたか
- action log が途中で途切れず、何をして何をやめたか追えるか
- `main` に残すべき要約・観測・判断を `notes/` へ反映したか
- `diary/` に残すべき日付依存ログを分けたか
- `notes/experiments/results/` に持ち帰る final JSON を選んだか
- raw 結果を results branch に残すか、不要として捨てるかを決めたか
- branch の結果を `main` にどう要約するか決め、図や数式が必要な場合は note 側で整理したか
- run が途中停止した場合に、停止理由を action log に記録し、次回は fresh run に切り替えると決めたか

## 8. 削除前の整理

- worktree を消す前に、その worktree に残っている知見を `main` へ吸い出す
- 吸い出し先は `notes/worktrees/` に固定する
- 吸い出した `notes/` は、worktree 側だけで終わらせず `main` に commit または merge してから削除へ進む
- 最低限、次を残す
  - branch 名
  - worktree の用途
  - 関連する結果やログ
  - action log の要約
  - 主要な観測
  - 次の `Idea:` / `Interpretation:` / `Consideration:`
  - `WORKTREE_SCOPE.md` があった場合は、その制約と実際の運用差分
- 実験 worktree の場合は、`notes/experiments/` と `notes/experiments/results/` の更新要否も確認する
- branch の入口は `notes/branches/` に置き、関連 note と raw 結果の所在を 1 か所から辿れるようにする
- 公開や再利用を前提とする図は `notes/assets/` に持ち帰る

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

## 11. 推奨ワークフロー

1. `scripts/setup_worktree.sh <branch-name> [worktree-path]` で worktree を切る
1. `WORKTREE_SCOPE.md` を埋め、action log と carry-over target を決める
1. `documents/research-workflow.md` を読み、問い、定式化、比較対象を決める
1. `notes/worktrees/worktree_<topic>_YYYY-MM-DD.md` を開き、作業開始を記録する
1. コード編集、テスト、実験、停止判断のたびに action log を追記する
1. run が止まったら partial を継がず、理由を記録して fresh run を切り直す
1. 実験の意味ある観測は `notes/experiments/<topic>.md` に整理する
1. `main` に残す最小 final JSON を `notes/experiments/results/` に選ぶ
1. `notes/branches/<branch_topic>.md` から、scope、note、result の入口を整える
1. `main` に carry-over を取り込んでから worktree を削除する
