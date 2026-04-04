# Agent Task Workflows

この文書は、恒久的なエージェントチームで頻出すると想定される 10 個の task と、その workflow を定義する shared catalog です。
Codex、Claude、GitHub Copilot などの runtime 差分には依存しません。
実験主導タスクでの批判的レビュー観点は `documents/experiment-critical-review.md` を参照してください。

## 目的

- よくある作業を role activation に落とし込む
- task ごとに specialist role の有無を明確にする
- 類似 task をまとめて、運用を過剰に複雑化させない
- repo を直接編集する役は `implementer` に一本化する
- 各 review の直後に、対象 role が feedback を受けて修正してから次段へ進む
- 実験主導タスクでは `experimenter -> experiment_reviewer -> report_reviewer -> experimenter / implementer` の反復を正本にする

## Research-Driven Change Family の明示ループ

`Research-Driven Change` では、外部調査つき実装、性能改善、比較検証を次の outer loop で扱います。

1. 外部調査と比較条件を固定する
1. baseline または current state を同じ protocol で記録する
1. `implementer` が 1 つの change を入れる
1. `experimenter` が同じ protocol で run する
1. `experiment_reviewer` が evidence と overclaim を批判的にレビューする
1. `experimenter` が report draft を作る
1. `report_reviewer` が reader-facing な report をレビューする
1. `report_rewrite_required` なら report だけを書き直す
1. `extra_validation_required` なら同じ仮説で追加検証を行う
1. `rerun_required` なら fresh run をやり直す。必要なら `implementer` が code か protocol を修正する
1. exit criteria を満たすまで 3-10 を反復する

### Niche Subflow: HLO 解析と compiler tuning

HLO dump、XLA flag、compiler behavior を仮説の中心に置く場合は、上の loop に次を追加します。

1. baseline と change 後で、同じ関数、同じ case、同じ backend 条件で HLO dump を採取する
1. `jax_util.hlo.dump` を正面 API とし、tag と dump path を固定する
1. `scripts/hlo/summarize_hlo_jsonl.py` で HLO 差分を集計する
1. `jax_util.xla_env` と `initializer(context)` を通して env を固定する
1. 1 反復で同時に複数の XLA flag と code change を混ぜない
1. HLO 差分、runtime metric、failure kind を同じ note / report で並べる
1. HLO の変化だけで改善と判定せず、`experiment_reviewer` が定量 evidence を要求する

## 想定タスク 10 個

| ID  | 想定タスク                                                | 主担当 workflow family | 有効化する専門ロール                         |
| --- | --------------------------------------------------------- | ---------------------- | -------------------------------------------- |
| T1  | solver / optimizer の局所バグ修正                         | Scoped Correction      | なし                                         |
| T2  | アルゴリズム改善のための外部調査つき実装                  | Research-Driven Change | `researcher`, `experimenter`, `report_reviewer` |
| T3  | `base` / `solvers` / `optimizers` をまたぐ大規模 refactor | Large Delivery         | `scheduler`                                  |
| T4  | 新 API / 新機能の追加                                     | Large Delivery         | `scheduler`                                  |
| T5  | CI failure / flaky test の切り分けと修正                  | Scoped Correction      | なし                                         |
| T6  | `experiment_runner` の拡張や運用改善                      | Platform and Infra     | `scheduler`, `infra_steward`, `experimenter`, `report_reviewer` |
| T7  | Docker / GitHub Actions / 自動化基盤の拡張                | Platform and Infra     | `infra_steward`                              |
| T8  | 性能ボトルネック調査と benchmark 主導の最適化             | Research-Driven Change | `researcher`, `experimenter`, `report_reviewer`, `scheduler`    |
| T9  | 実装変更に追随する設計文書・README・テスト整合            | Scoped Correction      | なし                                         |
| T10 | 依存更新、runtime 互換性対応、JAX 周辺 upgrade            | Platform and Infra     | `researcher`, `infra_steward`, `scheduler`   |

## 各タスクの workflow

### T1. solver / optimizer の局所バグ修正

1. `manager` が再現条件、受け入れ条件、影響範囲をまとめて固定する。
1. `manager_reviewer` が intake をレビューする。
1. `manager` が review を受けて scope を修正する。
1. `designer` が修正方針を設計する。
1. `design_reviewer` が設計をレビューする。
1. `designer` が review を受けて設計を修正する。
1. `implementer` が修正を入れる。
1. `change_reviewer` が局所回帰を確認する。
1. `implementer` が review を受けて修正する。
1. `final_reviewer` が独立レビューを行う。
1. `implementer` が final review を受け、指摘があれば修正する。
1. `verifier` が `pre_review.sh` または対象テストを実行する。

### T2. アルゴリズム改善のための外部調査つき実装

1. `manager` が「何を改善したいか」と評価指標を書く。
1. `manager_reviewer` が intake をレビューする。
1. `manager` が review を受けて intake を修正する。
1. `researcher` が paper / docs / web を調べて `research_notes.md` にまとめる。
1. `research_reviewer` が research をレビューする。
1. `researcher` が review を受けて research を修正する。
1. `manager` が採用案と実験の exit criteria を決める。
1. `designer` が実装方針を設計する。
1. `design_reviewer` が設計をレビューする。
1. `designer` が review を受けて設計を修正する。
1. `experimenter` が baseline run と比較プロトコルを記録する。
1. `experiment_reviewer` が baseline の妥当性と比較公平性を批判的にレビューする。
   レビュー観点は `documents/experiment-critical-review.md` を使う。
1. `experimenter` が review を受けて experiment log を修正する。
1. ここから `Research-Driven Change` の loop を開始する。
1. `implementer` が 1 つの change を実装する。
1. `change_reviewer` が各 chunk を逐次レビューする。
1. `implementer` が review を受けて修正する。
1. `experimenter` が同じ比較プロトコルで再実行する。
1. `experiment_reviewer` が result interpretation と overclaim をレビューする。
   review では math validity、literature connection、evidence sufficiency、figure validity も確認する。
1. `experimenter` が user-facing report draft を作る。
1. `report_reviewer` が report の概要、数値、図表、結論と根拠の対応をレビューする。
1. `experimenter` が `report_rewrite_required` を受けた場合、同じ result で report を書き直して `report_reviewer` へ戻す。
1. `experimenter` が `extra_validation_required` を受けた場合、同じ比較プロトコルに追加検証を加えて `experiment_reviewer` へ戻す。
1. `implementer` が `rerun_required` または protocol 修正要求を受けた場合、code か protocol を修正し、fresh rerun のために loop 先頭へ戻す。
1. 両 review が通り、exit criteria を満たすまで loop を閉じない。
1. `final_reviewer` が research と diff の整合を見る。
1. `implementer` が final review を受け、指摘があれば修正する。
1. `verifier` がテストと benchmark を実行する。

### T3. `base` / `solvers` / `optimizers` をまたぐ大規模 refactor

1. `manager` が refactor のゴールと非ゴールを書く。
1. `manager_reviewer` が intake をレビューする。
1. `manager` が review を受けて intake を修正する。
1. `scheduler` が `schedule.md` に milestone と依存関係を書く。
1. `schedule_reviewer` が milestone をレビューする。
1. `scheduler` が review を受けて schedule を修正する。
1. `manager` が chunk 順序と担当境界を確定する。
1. `designer` が chunk 境界を含む設計を作る。
1. `design_reviewer` が設計をレビューする。
1. `designer` が review を受けて設計を修正する。
1. `implementer` が chunk ごとに変更する。
1. `change_reviewer` が各 milestone を通す。
1. `implementer` が review を受けて逐次修正する。
1. `final_reviewer` が設計一貫性を確認する。
1. `implementer` が final review を受け、指摘があれば修正する。
1. `verifier` が主要テストと CI を実行する。

### T4. 新 API / 新機能の追加

1. `manager` が API contract と成功条件を書く。
1. `manager_reviewer` が intake をレビューする。
1. `manager` が review を受けて intake を修正する。
1. `scheduler` が実装、テスト、文書更新の順序を決める。
1. `schedule_reviewer` が順序計画をレビューする。
1. `scheduler` が review を受けて schedule を修正する。
1. `designer` が API 設計を作る。
1. `design_reviewer` が API 設計をレビューする。
1. `designer` が review を受けて設計を修正する。
1. `implementer` が API、テスト、ドキュメントを実装する。
1. `change_reviewer` が公開 API と backward compatibility を確認する。
1. `implementer` が review を受けて修正する。
1. `final_reviewer` が利用者目線で最終レビューする。
1. `implementer` が final review を受け、指摘があれば修正する。
1. `verifier` が CI と関連 examples を確認する。

### T5. CI failure / flaky test の切り分けと修正

1. `manager` が failure symptom と再現条件を書く。
1. `manager_reviewer` が intake をレビューする。
1. `manager` が review を受けて symptom 定義を修正する。
1. `manager` が「再現」「原因切り分け」「修正」の 3 段階に分ける。
1. `designer` が切り分けと修正方針を設計する。
1. `design_reviewer` が方針をレビューする。
1. `designer` が review を受けて方針を修正する。
1. `implementer` が原因箇所を修正する。
1. `change_reviewer` が過剰修正でないことを確認する。
1. `implementer` が review を受けて修正する。
1. `final_reviewer` が再発防止の観点でレビューする。
1. `implementer` が final review を受け、指摘があれば修正する。
1. `verifier` が失敗していたコマンドを再実行する。

### T6. `experiment_runner` の拡張や運用改善

1. `manager` が実験運用上の課題を書く。
1. `manager_reviewer` が intake をレビューする。
1. `manager` が review を受けて課題定義を修正する。
1. `infra_steward` が infra surface を `infra_notes.md` にまとめる。
1. `infra_reviewer` が infra 方針をレビューする。
1. `infra_steward` が review を受けて infra plan を修正する。
1. `scheduler` が rollout 順序を決める。
1. `schedule_reviewer` が rollout をレビューする。
1. `scheduler` が review を受けて schedule を修正する。
1. `designer` が runner / ops 設計を作る。
1. `design_reviewer` が設計をレビューする。
1. `designer` が review を受けて設計を修正する。
1. `experimenter` が現行 runner の baseline operation を記録する。
1. `experiment_reviewer` が観測性、比較条件、run 完走性をレビューする。
   review では `documents/experiment-critical-review.md` の relevant 項目を使う。
1. `experimenter` が review を受けて baseline 記録を修正する。
1. `implementer` が runner / scheduler / docs を更新する。
1. `change_reviewer` が運用 regressions を確認する。
1. `implementer` が review を受けて修正する。
1. `experimenter` が同じ運用シナリオを再実行する。
1. `experiment_reviewer` が regression と overclaim をレビューする。
   report と code の両方を確認する必要がある場合は、その両方を見て evidence sufficiency を確認する。
1. `experimenter` が運用 report draft を作る。
1. `report_reviewer` が report の根拠、数値、図表、読者向け説明をレビューする。
1. `experimenter` が report review を受けて report を書き直す。追加検証や rerun が必要なら実験手順へ戻す。
1. `implementer` が experiment review または report review を受け、指摘があれば修正する。
1. `final_reviewer` が安全性と observability を見る。
1. `implementer` が final review を受け、指摘があれば修正する。
1. `verifier` が runner 系テストを実行する。

### T7. Docker / GitHub Actions / 自動化基盤の拡張

1. `manager` が desired workflow と成功条件を書く。
1. `manager_reviewer` が intake をレビューする。
1. `manager` が review を受けて desired workflow を修正する。
1. `infra_steward` が対象 infra を整理する。
1. `infra_reviewer` が infra 方針をレビューする。
1. `infra_steward` が review を受けて infra plan を修正する。
1. `manager` が rollout と rollback 方針を決める。
1. `designer` が rollout / permission 設計を作る。
1. `design_reviewer` が設計をレビューする。
1. `designer` が review を受けて設計を修正する。
1. `implementer` が workflow / Docker / script を修正する。
1. `change_reviewer` が運用破壊リスクを確認する。
1. `implementer` が review を受けて修正する。
1. `final_reviewer` が secrets / permissions / maintenance cost を見る。
1. `implementer` が final review を受け、指摘があれば修正する。
1. `verifier` が構文と dry-run 相当の検証を行う。

### T8. 性能ボトルネック調査と benchmark 主導の最適化

1. `manager` が対象指標を定義する。
1. `manager_reviewer` が intake をレビューする。
1. `manager` が review を受けて評価条件を修正する。
1. `researcher` が手法候補や既知の最適化案を調べる。
1. `research_reviewer` が research をレビューする。
1. `researcher` が review を受けて research を修正する。
1. `scheduler` が baseline、実装、再計測の順序を組む。
1. `schedule_reviewer` が計測順序をレビューする。
1. `scheduler` が review を受けて順序を修正する。
1. `designer` が最適化方針を設計する。
1. `design_reviewer` が設計をレビューする。
1. `designer` が review を受けて設計を修正する。
1. `experimenter` が baseline benchmark を実行する。
1. `experiment_reviewer` が metric 妥当性と比較条件をレビューする。
   review では figure validity と literature connection も確認する。
1. `experimenter` が review を受けて benchmark log を修正する。
1. ここから `Research-Driven Change` の loop を開始する。
1. `implementer` が測定コードと最適化を入れる。
1. `change_reviewer` が可読性や数値安定性を確認する。
1. `implementer` が review を受けて修正する。
1. `experimenter` が同じ benchmark を再実行する。
1. `experiment_reviewer` が改善主張の妥当性をレビューする。
   結論に必要な data と figure が揃っているかも確認する。
1. `experimenter` が benchmark report draft を作る。
1. `report_reviewer` が report の概要、数値、図表、結論と根拠の対応をレビューする。
1. `experimenter` が `report_rewrite_required` を受けた場合、同じ result で report を書き直して `report_reviewer` へ戻す。
1. `experimenter` が `extra_validation_required` を受けた場合、同じ benchmark protocol に追加検証を加えて `experiment_reviewer` へ戻す。
1. `implementer` が `rerun_required` または benchmark protocol 修正要求を受けた場合、code か protocol を修正し、fresh rerun のために loop 先頭へ戻す。
1. 両 review が通り、exit criteria を満たすまで loop を閉じない。
1. `implementer` が experiment review または report review を受け、指摘があれば修正する。
1. `final_reviewer` が benchmark の妥当性を確認する。
1. `implementer` が final review を受け、指摘があれば修正する。
1. `verifier` が baseline 比較を実行する。

### T9. 実装変更に追随する設計文書・README・テスト整合

1. `manager` が変更対象の doc/test 範囲を書く。
1. `manager_reviewer` が intake をレビューする。
1. `manager` が review を受けて対象範囲を修正する。
1. `manager` が code first か docs first かを決める。
1. `designer` が文書反映方針を設計する。
1. `design_reviewer` が方針をレビューする。
1. `designer` が review を受けて設計を修正する。
1. `implementer` が doc、test、README を更新する。
1. `change_reviewer` が内容とリンク整合を確認する。
1. `implementer` が review を受けて修正する。
1. `final_reviewer` が利用者目線で読みやすさを確認する。
1. `implementer` が final review を受け、指摘があれば修正する。
1. `verifier` が markdown formatting と test を確認する。

### T10. 依存更新、runtime 互換性対応、JAX 周辺 upgrade

1. `manager` が upgrade の動機と互換性要件を書く。
1. `manager_reviewer` が intake をレビューする。
1. `manager` が review を受けて要件を修正する。
1. `researcher` が upstream release note と breaking change を調べる。
1. `research_reviewer` が research をレビューする。
1. `researcher` が review を受けて research を修正する。
1. `infra_steward` が CI / Docker / runtime 影響を整理する。
1. `infra_reviewer` が infra 影響をレビューする。
1. `infra_steward` が review を受けて infra plan を修正する。
1. `scheduler` が rollout を段階化する。
1. `schedule_reviewer` が rollout をレビューする。
1. `scheduler` が review を受けて rollout を修正する。
1. `designer` が migration 設計を作る。
1. `design_reviewer` が設計をレビューする。
1. `designer` が review を受けて設計を修正する。
1. `implementer` が dependency、コード、テスト、docs を更新する。
1. `change_reviewer` が migration risk を逐次確認する。
1. `implementer` が review を受けて修正する。
1. `final_reviewer` が upgrade 方針の妥当性を確認する。
1. `implementer` が final review を受け、指摘があれば修正する。
1. `verifier` が full CI を確認する。

## 冗長なタスクの整理

10 個の task をそのまま別 workflow として運用すると冗長なので、実運用では以下の 4 family に整理する。

| Workflow Family        | 代表 task   | まとめる理由                                                             |
| ---------------------- | ----------- | ------------------------------------------------------------------------ |
| Scoped Correction      | T1, T5, T9  | 局所的な変更で、specialist role なしでも完結しやすい                     |
| Research-Driven Change | T2, T8      | 外部調査に加え、`experimenter` による実験反復が本質になる                |
| Large Delivery         | T3, T4      | milestone 管理と chunk review が重要で、`scheduler` が共通して必要       |
| Platform and Infra     | T6, T7, T10 | infra surface、rollout、運用安全性が重要で、`infra_steward` が中心になる |

## 実務上の使い方

- まず task を 10 個のどれかに寄せる。
- 次に、それを 4 つの workflow family のどれで運用するか決める。
- specialist role は family に応じて有効化する。
- family に入らない task が出たら、新 task を増やす前に既存 family に吸収できないかを確認する。
