# Branch Scope と Git ワークフロー

この文書は、branch 名、branch の責務、commit / push、merge / rebase の判断をまとめた正本です。
worktree の作成と carry-over の流れは [worktree-lifecycle.md](/workspace/documents/worktree-lifecycle.md) を参照します。

## 1. 基本方針

- 1 branch = 1 topic を原則とします。
- branch の責務が広がったら、branch を分けるか `WORKTREE_SCOPE.md` を更新します。
- 実装コードと長時間実験の生成物は、必要に応じて branch を分けます。
- `main` は統合先であり、試行錯誤や途中生成物の置き場にはしません。

## 2. branch 名

- 通常の実装 branch は `work/<topic>-YYYYMMDD` を原則とします。
- 結果保存 branch は `results/<topic>` を原則とします。
- branch 名は目的が読める英語句で付けます。

## 3. Scope の固定

- branch を切ったら、対応する worktree root に `WORKTREE_SCOPE.md` を置きます。
- `WORKTREE_SCOPE.md` には editable directories、carry-over target、action log を明記します。
- branch の入口は `notes/branches/<branch_topic>.md` に置き、scope と関連 note をそこから辿れるようにします。

## 4. コミット・プッシュ

- commit は branch の責務に収まる差分だけを含めます。
- `WORKTREE_SCOPE.md` を更新した場合は、早い段階で commit します。
- push 前に、その branch で必須の test / lint / document check を実行します。
- 初回 push は `git push -u origin <branch-name>` を使います。

## 5. Conflict 解決と merge / rebase

- `main` 取り込みは、branch の目的に必要な最小限に留めます。
- 履歴を読みやすく保つため、ローカル整理には `rebase` を使ってよいです。
- 統合時の安全性と文脈保持を優先する場合は `merge` を選びます。
- 別 branch と同じファイルを触っている場合は、先に `notes/branches/` と `notes/worktrees/` で衝突リスクを明示します。

## 6. 削除前チェック

- branch の目的が `notes/branches/` から辿れる
- `main` に持ち帰る note / final JSON が整理済み
- raw 結果を残す場所が決まっている
- `git worktree list` と `git branch -v` で後片付け対象が分かる
