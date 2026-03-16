# Experiment Notes

このディレクトリには、`main` から辿れる形で残しておきたい実験メモを置きます。

実験メモは、原則として実験ごとに 1 ファイル以上を割り当てます。

一つのファイルに unrelated な実験を混ぜず、

- 実験名
- 日付
- branch

が分かる単位で分けます。

- branch 名
- 結果の所在
- 定性的な考察
- 追加調査のメモ

大きな JSON、HTML、SVG、ログ本体は results branch 側に置き、`main` にはメモだけを残します。
ただし、後から別の図を再生成できるよう、各 branch の最終結果として必要最小限の JSON は `notes/experiments/results/` に持ち帰ります。

数式が必要な場合は、GitHub Flavored Markdown の数式記法をそのまま使って構いません。

タイトルや先頭 H1 は、日付より実験トピックを主にして付けます。
日付が必要な場合でも、`Legacy Smolyak Results` のようにトピックを先にし、日付は副情報として扱います。

過去の実験メモは、`main` に置いたあと原則として書き換えません。
補足や訂正が必要な場合は、

- 既存ファイル末尾へ `Addendum:` / `Correction:` として追記する
- または新しい日付付きメモを追加して参照を張る

のどちらかにします。

## Recommended Structure

- `Source:`
  - 文献や既存結果から確定的に言えること
- `Idea:`
  - 今後試す設計案や仮説
- `Interpretation:` / `Consideration:`
  - 実験結果の読みや比較、設計判断

実験 note では、少なくとも次の情報が 1 ファイル内で見えるようにします。

- branch 名
- source JSON / JSONL
- `main` に持ち帰った final JSON
- 主要な観測値
- 定性的な結論
- 次の課題

文献から得た記述は、対象文献を具体的に明示します。

## Final JSON

- `main` へ持ち帰る final JSON は `notes/experiments/results/` に置きます。
- raw な JSONL や大量ログまでは持ち込まず、後段の図再生成に必要な最小限の JSON を選びます。
- note 本文からは、その JSON へのリンクを必ず張ります。
