# Literature Survey

## Purpose

研究、設計、比較実験の前提になる文献調査を、検索、選定、比較、反証候補の抽出まで含めて整理します。

## Use This For

- 先行研究の全体像を掴みたいとき
- 関連論文、サーベイ、technical report を探したいとき
- ある主張を支える文献と弱める文献の両方を集めたいとき
- baseline、評価指標、failure mode の根拠を文献ベースで決めたいとき
- `references/` や `notes/themes/` に残す文献索引を作りたいとき

## Must Read Before Working

- `documents/research-workflow.md`
- `documents/workflow-references.md`
- `references/README.md`
- 必要なら対象 topic の既存 `references/*.md`
- 必要なら対象 topic の既存 `notes/themes/*.md`

## Inputs

- 調べたい問い
- topic と除外範囲
- 想定キーワード
- time range や source の制約
- 出力先の note / reference path

## Outputs

- 調査スコープ
- 検索語の束
- 主要文献一覧
- 支持根拠と反証候補の整理
- baseline / protocol / failure mode に関する文献ベースの候補
- `references/` または `notes/` に残す要約

## Mandatory Checklist

- summary や blog より primary source を優先する
- peer-reviewed、preprint、vendor doc、blog を区別して記録する
- 支持文献だけでなく、結論を弱める文献や限定条件も探す
- query、探索日、採用理由、除外理由を残す
- 対象タスクに近い problem setting、data regime、hardware regime を区別する
- 文献から直接言えることと、自分の解釈を分ける
- survey または review paper がある topic では、それを優先して押さえる
- 最終的に使う主張ごとに source を辿れるようにする

## Standard Flow

1. 問いを 1 文で固定する
1. inclusion / exclusion を決める
1. query pack を作る
1. survey、代表論文、最近の比較論文を優先して集める
1. 支持文献と反証候補を分ける
1. 各文献について、setting、claim、limitations、使えそうな部分を短く抜く
1. baseline、metric、failure mode、artifact policy に効く文献を抜き出す
1. `Known`, `Contested`, `Open` に整理する
1. `references/` か `notes/` に出力する

## Deliverable Shape

- `Question`
- `Scope`
- `Search Log`
- `Primary Sources`
- `Contrary Or Narrowing Sources`
- `Known`
- `Contested`
- `Open`
- `Implications For Implementation Or Experiment`

## Review Stance

- 文献調査では「都合の良い引用集」にしません。
- 近い setting でも、目的関数、データ条件、計算資源、報告指標が違うなら別物として扱います。
- survey がある topic では、単発論文の寄せ集めで済ませません。
- 強い主張ほど、反例、例外条件、古いが重要な一次資料を探します。

## Boundary

- official vendor docs や API 仕様の確認は `.codex/agents/docs_researcher.toml` を使います。
- 研究全体の outer loop は `agents/skills/research-workflow.md` を使います。
- 実験結果の批判的評価は `agents/skills/critical-review.md` を使います。
- user-facing report の品質確認は `agents/skills/report-review.md` を使います。

## Implementation Surface

- `.github/skills/26-literature-survey/`
