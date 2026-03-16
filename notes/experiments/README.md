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

数式が必要な場合は、GitHub Flavored Markdown の数式記法をそのまま使って構いません。

## Recommended Structure

- `Source:`
  - 文献や既存結果から確定的に言えること
- `Idea:`
  - 今後試す設計案や仮説
- `Interpretation:` / `Consideration:`
  - 実験結果の読みや比較、設計判断

文献から得た記述は、対象文献を具体的に明示します。
