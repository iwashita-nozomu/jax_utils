# Critical Review

## Purpose

実験結果や設計上の主張が、与えられた evidence で本当に支えられているかを批判的に確認します。

## Use This For

- 実験結果の過剰主張チェック
- 比較公平性の確認
- figure / summary / conclusion の妥当性確認
- reviewer、statistician、domain expert 視点での再点検
- 実験や benchmark を完了扱いにしてよいかの判定

## Inputs

- result
- report
- 比較対象
- 実験条件

## Outputs

- evidence sufficiency の判定
- overclaim や comparison bias の指摘
- 追加で必要な run や分析

## Review Stance

- 人ではなく主張に対して敵対的に振る舞います。
- 「この結論を崩すには何を見ればよいか」を先に考えます。
- 反例、比較条件の穴、統計や図の弱さ、関連文献との不整合を優先して探します。
- 文献や仕様を掘ってでも、結論を弱める根拠がないか確認します。
- 実験結果つきの依頼では、甘い要約より厳しい finding を優先します。

## Default Placement In Workflow

- `Research-Driven Change` では、比較条件の確認と結果主張の前に使います。
- `experiment-lifecycle` を使う run では、run 後の report / summary / conclusion を閉じる前に使います。
- user-facing report を閉じる前には、その後に `report-review` も通します。
- 実験結果つきの依頼では、これを省略して完了扱いにしません。

## Implementation Surface

- `.github/skills/04-critical-review/`

## Boundary

- 実験を実際に回す運用自体は `agents/skills/experiment-lifecycle.md` を使います。
- user-facing report の readability、数値の見せ方、figure / table 導線の独立確認は `agents/skills/report-review.md` を使います。
