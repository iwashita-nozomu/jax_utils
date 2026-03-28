# Agent Task Workflows

この文書は、恒久的なエージェントチームで頻出すると想定される 10 個のタスクと、その workflow を定義するカタログです。

## 目的

- よくある作業を role activation に落とし込む
- task ごとに specialist role の有無を明確にする
- 類似 task をまとめて、運用を過剰に複雑化させない

## 想定タスク 10 個

| ID  | 想定タスク                                                | 主担当 workflow family | 有効化する専門ロール                       |
| --- | --------------------------------------------------------- | ---------------------- | ------------------------------------------ |
| T1  | solver / optimizer の局所バグ修正                         | Scoped Correction      | なし                                       |
| T2  | アルゴリズム改善のための外部調査つき実装                  | Research-Driven Change | `researcher`                               |
| T3  | `base` / `solvers` / `optimizers` をまたぐ大規模 refactor | Large Delivery         | `scheduler`                                |
| T4  | 新 API / 新機能の追加                                     | Large Delivery         | `scheduler`                                |
| T5  | CI failure / flaky test の切り分けと修正                  | Scoped Correction      | なし                                       |
| T6  | `experiment_runner` の拡張や運用改善                      | Platform and Infra     | `scheduler`, `infra_steward`               |
| T7  | Docker / GitHub Actions / 自動化基盤の拡張                | Platform and Infra     | `infra_steward`                            |
| T8  | 性能ボトルネック調査と benchmark 主導の最適化             | Research-Driven Change | `researcher`, `scheduler`                  |
| T9  | 実装変更に追随する設計文書・README・テスト整合            | Scoped Correction      | なし                                       |
| T10 | 依存更新、runtime 互換性対応、JAX 周辺 upgrade            | Platform and Infra     | `researcher`, `infra_steward`, `scheduler` |

## 各タスクの workflow

### T1. solver / optimizer の局所バグ修正

1. `intent_analyst` が再現条件と受け入れ条件を `intent_brief.md` に書く。
1. `coordinator` が影響範囲を 1 〜 3 ファイル程度に固定する。
1. `editor` が修正を入れる。
1. `change_reviewer` が局所回帰を確認する。
1. `final_reviewer` が独立レビューを行う。
1. `verifier` が `pre_review.sh` または対象テストを実行する。

### T2. アルゴリズム改善のための外部調査つき実装

1. `intent_analyst` が「何を改善したいか」と評価指標を書く。
1. `researcher` が paper / docs / web を調べて `research_notes.md` にまとめる。
1. `coordinator` が採用案を決め、`editor` に渡す。
1. `editor` が実装する。
1. `change_reviewer` が各 chunk を逐次レビューする。
1. `final_reviewer` が research と diff の整合を見る。
1. `verifier` がテストと benchmark を実行する。

### T3. `base` / `solvers` / `optimizers` をまたぐ大規模 refactor

1. `intent_analyst` が refactor のゴールと非ゴールを書く。
1. `scheduler` が `schedule.md` に milestone と依存関係を書く。
1. `coordinator` が chunk 順序を確定する。
1. `editor` が chunk ごとに変更する。
1. `change_reviewer` が各 milestone を通す。
1. `final_reviewer` が設計一貫性を確認する。
1. `verifier` が主要テストと CI を実行する。

### T4. 新 API / 新機能の追加

1. `intent_analyst` が API contract と成功条件を書く。
1. `scheduler` が実装、テスト、文書更新の順序を決める。
1. `editor` が API、テスト、ドキュメントを実装する。
1. `change_reviewer` が公開 API と backward compatibility を確認する。
1. `final_reviewer` が利用者目線で最終レビューする。
1. `verifier` が CI と関連 examples を確認する。

### T5. CI failure / flaky test の切り分けと修正

1. `intent_analyst` が failure symptom と再現条件を書く。
1. `coordinator` が「再現」「原因切り分け」「修正」の 3 段階に分ける。
1. `editor` が原因箇所を修正する。
1. `change_reviewer` が過剰修正でないことを確認する。
1. `final_reviewer` が再発防止の観点でレビューする。
1. `verifier` が失敗していたコマンドを再実行する。

### T6. `experiment_runner` の拡張や運用改善

1. `intent_analyst` が実験運用上の課題を書く。
1. `infra_steward` が infra surface を `infra_notes.md` にまとめる。
1. `scheduler` が rollout 順序を決める。
1. `editor` が runner / scheduler / docs を更新する。
1. `change_reviewer` が運用 regressions を確認する。
1. `final_reviewer` が安全性と observability を見る。
1. `verifier` が runner 系テストを実行する。

### T7. Docker / GitHub Actions / 自動化基盤の拡張

1. `intent_analyst` が desired workflow と成功条件を書く。
1. `infra_steward` が対象 infra を整理する。
1. `coordinator` が rollout と rollback 方針を決める。
1. `editor` が workflow / Docker / script を修正する。
1. `change_reviewer` が運用破壊リスクを確認する。
1. `final_reviewer` が secrets / permissions / maintenance cost を見る。
1. `verifier` が構文と dry-run 相当の検証を行う。

### T8. 性能ボトルネック調査と benchmark 主導の最適化

1. `intent_analyst` が対象指標を定義する。
1. `researcher` が手法候補や既知の最適化案を調べる。
1. `scheduler` が baseline、実装、再計測の順序を組む。
1. `editor` が測定コードと最適化を入れる。
1. `change_reviewer` が可読性や数値安定性を確認する。
1. `final_reviewer` が benchmark の妥当性を確認する。
1. `verifier` が baseline 比較を実行する。

### T9. 実装変更に追随する設計文書・README・テスト整合

1. `intent_analyst` が変更対象の doc/test 範囲を書く。
1. `coordinator` が code first か docs first かを決める。
1. `editor` が doc、test、README を更新する。
1. `change_reviewer` が内容とリンク整合を確認する。
1. `final_reviewer` が利用者目線で読みやすさを確認する。
1. `verifier` が markdown formatting と test を確認する。

### T10. 依存更新、runtime 互換性対応、JAX 周辺 upgrade

1. `intent_analyst` が upgrade の動機と互換性要件を書く。
1. `researcher` が upstream release note と breaking change を調べる。
1. `infra_steward` が CI / Docker / runtime 影響を整理する。
1. `scheduler` が rollout を段階化する。
1. `editor` が dependency、コード、テスト、docs を更新する。
1. `change_reviewer` が migration risk を逐次確認する。
1. `final_reviewer` が upgrade 方針の妥当性を確認する。
1. `verifier` が full CI を確認する。

## 冗長なタスクの整理

10 個の task をそのまま別 workflow として運用すると冗長なので、実運用では以下の 4 family に整理する。

| Workflow Family        | 代表 task   | まとめる理由                                                             |
| ---------------------- | ----------- | ------------------------------------------------------------------------ |
| Scoped Correction      | T1, T5, T9  | 局所的な変更で、specialist role なしでも完結しやすい                     |
| Research-Driven Change | T2, T8      | 外部調査と比較評価が本質で、`researcher` が共通して必要                  |
| Large Delivery         | T3, T4      | milestone 管理と chunk review が重要で、`scheduler` が共通して必要       |
| Platform and Infra     | T6, T7, T10 | infra surface、rollout、運用安全性が重要で、`infra_steward` が中心になる |

## 実務上の使い方

- まず task を 10 個のどれかに寄せる。
- 次に、それを 4 つの workflow family のどれで運用するか決める。
- specialist role は family に応じて有効化する。
- family に入らない task が出たら、新 task を増やす前に既存 family に吸収できないかを確認する。
