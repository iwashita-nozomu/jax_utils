# Experiment Notes

このディレクトリには、`main` から辿れる形で残しておきたい実験メモを置きます。
研究の問い、定式化、比較対象、段階的改造の正本は [documents/research-workflow.md](../../documents/research-workflow.md) を参照してください。
準備、実装、静的チェック、実行、結果レポートの標準手順は [documents/experiment-workflow.md](../../documents/experiment-workflow.md) を参照してください。
実験レポートの構成と体裁は [documents/experiment-report-style.md](../../documents/experiment-report-style.md) を参照してください。

Experiment と benchmark の使い分けは [benchmark_vs_experiment.md](../knowledge/benchmark_vs_experiment.md) を参照してください。

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

report として読むなら、まず大見出しは次の順を推奨します。

- `Abstract`
- `Question and Context`
- `Protocol`
- `Results`
- `Discussion`
- `Limitations`
- `Reproducibility Record`
- `Artifacts and Carry-Over`
- `Critical Review`
- `Conclusion`

テンプレートは [REPORT_TEMPLATE.md](/workspace/notes/experiments/REPORT_TEMPLATE.md) を使えます。

そのうえで、下位ラベルとして次を残します。

- `Question:`
- `Formulation:`
- `Equation:`
- `Assumptions:`
- `Comparison Target:`
- `Metrics:`
- `Source:`
  - 文献や既存結果から確定的に言えること
- `Change:`
- `Validation Plan:`
- `Result Summary:`
- `Quantitative Summary:`
- `Comparison Table:`
- `Idea:`
  - 今後試す設計案や仮説
- `Interpretation:` / `Consideration:`
  - 実験結果の読みや比較、設計判断
- `Critical Review:`
- `Limitation:`
- `Decision:`
- `Branch Reflection:`

実験 note では、少なくとも次の情報が 1 ファイル内で見えるようにします。

- 問い
- 定式化
- 数式または擬似数式
- 仮定
- branch 名
- source JSON / JSONL
- `main` に持ち帰った final JSON
- 比較対象
- 主要 metric
- 主要な観測値
- 定量サマリ
- 比較表または比較箇条書き
- 定性的な結論
- `Results` と `Discussion` の書き分け
- 結論と根拠 figure / table の対応
- 批判的レビューまたは overclaim への自己点検
- 次の課題
- 実験中に何を走らせ、どこで条件を変えたかの action log

途中停止した partial run を正式結果として扱わず、必要なら停止理由と再実行判断だけを note に残します。

文献から得た記述は、対象文献を具体的に明示します。

## Report Writing Notes

- `Abstract` は最後に書きます。
- `Results` では観測事実を先に書き、`Discussion` ではその意味を書きます。
- 図表を使う場合は、本文だけでも主結果が読めるようにします。
- figure / table caption は、それ単体でも条件と主結果が分かるように書きます。
- headline metric の横に success rate と failure kind を見える形で置きます。
- figure には軸名、単位、scale を入れます。
- 読み方が一目で分からない graph には、caption か本文で `How to read:` を 1 文入れます。
- `Conclusion` では、各主張に対応する figure / table を明示します。
- 次元や level の sweep は、原則として 1 ずつ上げる連続レンジを使います。飛び飛びの点だけなら、その理由を `Protocol` に書きます。

## Final JSON

- `main` へ持ち帰る final JSON は `notes/experiments/results/` に置きます。
- raw な JSONL や大量ログまでは持ち込まず、後段の図再生成に必要な最小限の JSON を選びます。
- note 本文からは、その JSON へのリンクを必ず張ります。
- carry-over の正本は完走した run の final JSON を原則とし、partial run の JSON は持ち帰り対象にしません。

## In-Worktree Drafting

- worktree で実験を進めるときも、最終的に `main` に置くのと同じ `notes/experiments/<topic>.md` へ追記します。
- 詳細な一挙手一投足は `notes/worktrees/` の action log に残し、この note には実験として意味のある条件変更と観測を要約します。
- 数式、比較対象、各改造の意図は `notes/worktrees/` にだけ閉じず、この note 側にも要約を残します。
