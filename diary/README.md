# Diary

`diary/` は、日付ごとの作業ログや判断の流れを残すためのディレクトリです。

- ファイル名は `YYYY-MM-DD.md` を基本にします。
- 実験メモや考察のうち、日付依存の強いものは `diary/` に置きます。
- 長期的に参照したい実験の要約や branch 対応は `notes/` に残します。
- 同じ日の作業は既存の `YYYY-MM-DD.md` に追記します。
- 細かい判断や途中の迷いも削らずに残し、時系列の流れが後で追える状態を優先します。
- `main` に置いた過去日の diary 本文は書き換えません。
- 補足や訂正が必要な場合は、既存本文を差し替えず、同じファイル末尾へ `Correction:` / `Addendum:` を付けて追記します。

## Format

- 文書は Markdown で書きます。
- GitHub Flavored Markdown の数式記法を使ってよく、インライン数式は `$...$`、独立数式は `$$...$$` を使います。
- 箇条書きだけでなく、短い経緯や判断理由を書いてかまいません。
- 日中に何度も更新するときは、追記単位で小見出しを足せます。

## Source And Idea Marking

- 文献や資料から得た情報を書くときは、対象の文献を明示します。
- 少なくとも、文献名、著者名、年、必要な場合はファイル名や branch 上の保存先を残します。
- 自分のアイデアは、文献上の事実と混ざらないように `Idea:` などの明示ラベルを付けます。
- 自分の考察や推測は、必要に応じて文献を確認したうえで `Interpretation:` や `Consideration:` などのラベルで区別します。
- 文献未確認の仮説を書くときは、その場で事実のように断定せず、未確認であることを明記します。

## Entries

- [2026-03-16.md](/workspace/diary/2026-03-16.md)
- [2026-03-17.md](/workspace/diary/2026-03-17.md)
- [2026-03-18.md](/workspace/diary/2026-03-18.md)
- [2026-03-19.md](/workspace/diary/2026-03-19.md)
- [2026-03-20.md](/workspace/diary/2026-03-20.md)
- [2026-03-21_week2_complete.md](/workspace/diary/2026-03-21_week2_complete.md)
- [2026-04-01_FINAL_COMPLETION.md](/workspace/diary/2026-04-01_FINAL_COMPLETION.md)
