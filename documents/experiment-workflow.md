# 実験の標準手順

この文書は、repo 内で実験を進めるときの統合入口です。
個別の `experiments/<topic>/README.md` は、その実験やモジュール固有の使い方として残し、この文書では topic をまたぐ汎用的な実験方法を扱います。

扱う流れは次の 5 段階です。

1. 準備
1. 実験コードの実装
1. 実験コードの静的チェック
1. 実験実行
1. 結果レポート

さらに、実験を進めながらコード自体を改造する必要がある場合は、結果とレポートを毎回生成し、サブエージェントによる批判的レビューを挟んで反復する workflow を標準にします。

## 1. この文書の役割

この文書は実験実務の入口です。詳細は次に分けます。

- 問い、定式化、比較設計、claim の更新
  - [research-workflow.md](/workspace/documents/research-workflow.md)
- 実験コードと生成物の運用規約
  - [coding-conventions-experiments.md](/workspace/documents/coding-conventions-experiments.md)
- レポート本文の構成と figure / table の書き方
  - [experiment-report-style.md](/workspace/documents/experiment-report-style.md)
- 批判的レビューの観点と手順
  - [experiment-critical-review.md](/workspace/documents/experiment-critical-review.md)
- `experiment_runner`、scheduler、monitor の使い分け
  - [experiment_runner_usage.md](/workspace/notes/experiments/experiment_runner_usage.md)
- エージェントごとの task workflow
  - [TASK_WORKFLOWS.md](/workspace/agents/TASK_WORKFLOWS.md)

## 2. 段階別手順

### 1. 準備

実装や run に入る前に、最低限次を固定します。

- `Question:`
  - 今回の実験で何を確かめたいか。速度、精度、メモリ、failure pattern、安定性のどれが主題か。
- `Comparison Target:`
  - main 実装、旧実装、baseline、外部 reference のどれと比べるか。
- `Metrics:`
  - `summary.json` と report に何を残すか。少なくとも時間、成功率、failure kind、主要誤差を含めます。
- `Stop Condition:`
  - smoke で止めるのか、verified まで進めるのか、正式な比較表や report まで必要なのか。
- `Fairness Notes:`
  - 同じ case set、同じ timeout、同じ hardware、同じ seed policy、同じ allocator 方針をどこまで維持するか。
- `Artifact Plan:`
  - 実験ディレクトリ、`result/<run_name>/` の出力先、`experiments/report/<run_name>.md` の置き場を先に固定します。
- `Naming Plan:`
  - topic 名、run_name、result ディレクトリ名、report 名の規則を先に決め、topic README か対応する正本文書へ残します。
- `Branch Plan:`
  - 同じ branch で進めるか、必要があれば別 branch / worktree を使うかを先に決めます。既定は同じ branch です。

次に、branch / worktree の分離要否を決めます。

- 通常の実験
  - 既存の branch 上で、そのまま進めます。
- 隔離が必要な実験
  - 長時間 run、巨大生成物、破壊的な試行がある場合に限って別 branch / worktree の使用を許可します。
- action log
  - 詳細な作業ログが必要な場合は `notes/worktrees/worktree_<topic>_YYYY-MM-DD.md` を使います。

準備段階で固定する置き場は次です。

- 実験コード
  - `experiments/<topic>/`
- runtime 生成物
  - `experiments/<topic>/result/<run_name>/`
- 1 回の実験 report
  - `experiments/report/<run_name>.md`
- 複数 run をまたぐ要約や知見
  - `notes/experiments/<topic>.md` または `notes/themes/`

top-level の `reports/` は project-wide な review、automation、management report の置き場として扱い、topic ごとの experiment report の正本には使いません。`notes/experiments/` は run ごとの一次 report ではなく、横断的な要約の置き場として使います。

準備段階で固定する命名は次です。

- topic ディレクトリ名
  - `snake_case`
- run_name
  - `<topic>_<variant>_<YYYYMMDDTHHMMSSZ>`
- runtime 生成物
  - `result/<run_name>/summary.json`
  - `result/<run_name>/cases.jsonl`
  - `result/<run_name>/run.log`
  - 図を出力する場合は `result/<run_name>/figures/`
- report 名
  - `experiments/report/<run_name>.md`

準備段階で確認するものは次です。

- topic の `README.md`
- 直近の experiment report
- `summary.json` / `cases.jsonl` の schema
- `git status --short`
- 既定の出力先と命名が topic README に書かれているか

`Interpretation:`
準備段階の目的は、今回の run が debug なのか、verified なのか、正式比較なのかを曖昧にしないことです。

### 2. 実験コードの実装

実験コードは、「問いと比較」を表現する薄い層として実装します。
process 管理や GPU 割当は runner 側の責務であり、実験 script 側に重複実装しません。

推奨構成は次です。

- `README.md`
  - 実験目的、コード配置、標準コマンド、出力先、report の入口、命名規則を書く。
- `cases.py`
  - case 定義、difficulty range、resource estimate を置く。
- `experimentcode.py`
  - orchestration と CLI に集中させる。
- `result/`
  - `result/<run_name>/` ごとに JSON、JSONL、ログ、図を置く。

実装時点で、少なくとも次の配置と名前を README に明記します。

- 実験コードの topic パス
- `result/<run_name>/` の canonical 出力先
- `experiments/report/<run_name>.md` の置き場
- 関連する `notes/` を使う場合はその入口
- run_name の形式

`experiment_runner` を使う場合の入口は次です。

- 一般的な実験
  - `StandardWorker`
  - `StandardScheduler` または `StandardFullResourceScheduler`
  - `StandardRunner`
- host 側で pid / worker slot を強く観測したい実験
  - `build_worker_slots()`
  - `run_cases_with_subprocess_scheduler()`
  - 監視が必要な場合は `RuntimeMonitor`

`experiment_runner` に委譲するものは次です。

- case ごとの fresh child process lifecycle
- timeout と child cleanup
- child / parent の diagnostics 記録
- `environment_variables` の child 反映

実装時にやらないことは次です。

- 実験 script 内で独自の mini-runner を書く
- GPU slot 管理を script 側で持つ
- `CUDA_VISIBLE_DEVICES` や `XLA_*` を script 側で直接組み立てる
- JAX / XLA env が必要な場合に `jax_util.xla_env` を通さない
- native crash / signal / timeout の回収を script 側で独自実装する
- partial run を前提にした resume protocol を作る
- ad hoc な result path 命名を増やす

### 3. 実験コードの静的チェック

長時間 run の前に、少なくとも静的に次を確認します。

- `pyright`
  - 実験コード、`experiment_runner` 利用部分、集計コードの型整合を確認する。
- `ruff check`
  - import、未使用変数、到達不能コード、雑な例外処理を早めに落とす。
- CLI help
  - `python experimentcode.py --help` が通ることを確認する。
- import path
  - top-level import と package path が壊れていないことを確認する。
- 出力 schema
  - `summary.json` に必要な key が揃うよう、集計コードを静的に読んでおく。

静的チェックの段階では、まだ正式な benchmark conclusion を出しません。
ここでの目的は「長時間 run を始めても、型・import・引数の破綻で止まらない状態」にすることです。

`Note:`
pickle 可否、JAX import 後の env 汚染、GPU visibility の実際の反映は静的チェックだけでは分かりません。
それらは次の実行段階で smoke / verified として確認します。

### 4. 実験実行

実験実行は、少なくとも次の段階に分けます。

#### 4.1 smoke

最小の CPU run か、ごく狭い case range で次を確認します。

- import が通る
- worker が起動する
- JSONL が追記される
- `summary.json` が生成される
- report 再生成の入口が成立する

#### 4.2 verified

本番に近い backend と env で、worker 数を絞って narrow run を行います。
ここで確認するのは次です。

- GPU visibility
- allocator 設定
- worker slot / timeout の挙動
- failure kind の記録
- `summary.json` と `cases.jsonl` の整合

#### 4.3 formal run

比較表や report の根拠にする run は、条件を固定した fresh run として 1 回で完走させます。

- case range
- timeout
- dtype
- platform
- workers per GPU
- allocator 方針
- 出力先

は run 開始前に固定し、途中で script を書き換えながら継ぎ足しません。

#### 4.4 long run のルール

- 長時間 run でも、別 branch / worktree は必須ではありません。隔離が必要なときだけ使います。
- run は 1 つの run_name と 1 つの出力先に閉じた fresh 実行として扱います。
- case ごとの JSONL は progress 記録と failure 診断のために保存します。
- partial run の保存は診断材料に限って許可します。正規の再開点としての使用を禁止します。
- 止まった run は `Stop Reason:` と `Restart Decision:` を log に残し、新しい run_name で 0 からやり直します。

#### 4.5 monitor

host 側で worker 状態や GPU 利用状況を見たい場合は、`RuntimeMonitor` を使います。
ただし monitor は evidence そのものではなく、run の観測補助です。
正式な evidence は最終的に JSON、JSONL、report、note へ落とします。

### 5. 結果レポート

run 後は、必ず結果を report と note に整理します。
批判的レビューの観点は [experiment-critical-review.md](/workspace/documents/experiment-critical-review.md) を正本にします。

最低限残すものは次です。

- `summary.json`
- `cases.jsonl`
- report へのリンク
- `Result Summary:`
- `Quantitative Summary:`
- `Comparison Table:`
- `Critical Review:`

report 本文は次の構成を基本にします。

- `Question and Context`
- `Protocol`
- `Results`
- `Discussion`
- `Limitations`
- `Reproducibility Record`
- `Artifacts and Carry-Over`
- `Critical Review`
- `Conclusion`

結果レポートでは、少なくとも次を分けます。

- 観測事実
  - `Results`
- その意味と比較
  - `Discussion`
- まだ言えないこと
  - `Limitations` と `Critical Review`

carry-over のルールは次です。

- 実行ごとの生成物は `experiments/<topic>/result/<run_name>/` に残す
- 1 回の実験 report は `experiments/report/<run_name>.md` に残す
- 複数 run をまたぐ知見だけを `notes/` へ持ち上げる
- partial run は診断用とし、正式な report の正本にしない

## 3. コード改造を伴う反復ワークフロー

実験を行いながらコードを改造する必要がある場合は、単発の

- 実装
- 実行
- 感想

では終わらせず、毎回の実験で結果とレポートを生成し、サブエージェントによる批判的レビューを挟んで反復します。

標準ループは次です。

1. `manager`
   - 今回の `Question:`、`Comparison Target:`、`Stop Condition:` を固定する。
1. `implementer`
   - コード変更を入れる。
1. `change_reviewer`
   - code diff を批判的にレビューする。
   - 数学的妥当性や報告内容も確認する場合は [experiment-critical-review.md](/workspace/documents/experiment-critical-review.md) の `Mathematical Validity` と `As Reported` を使う。
1. `implementer`
   - review を反映し、静的チェックを通す。
1. `experimenter`
   - 同じ protocol で fresh run を実行する。
1. `experimenter`
   - `summary.json`、`cases.jsonl`、report を生成する。`notes/` を使う場合は対応する experiment note も生成する。
1. `experiment_reviewer`
   - report と結果の読み方を批判的にレビューする。
   - [experiment-critical-review.md](/workspace/documents/experiment-critical-review.md) を使って、math validity、evidence sufficiency、figure validity、overclaim を確認する。
1. `implementer`
   - 必要な修正を入れる。
1. 4-8 を終了条件まで反復する。
1. `final_reviewer`
   - 最終 code と最終 claim を独立にレビューする。
1. `verifier`
   - gate を実行する。

この workflow の要点は次です。

- 毎回の実験で、結果とレポートを必ず生成する
- code review と report review を分ける
- 同じ protocol で再実行し、都合のよい subset に逃げない
- 修正のたびに静的チェックを挟む
- 良い結果だけでなく、失敗例、悪化例、未解決点も同じ note に残す

### 3.1 各サイクルで必ず残すもの

各 iteration で最低限残すものは次です。

- `Change:`
- `Expected Effect:`
- `Validation Plan:`
- 静的チェック結果
- 実行コマンド
- `result/<run_name>/` の所在
- report の所在
- 置き場と命名規則の変更有無
- `Critical Review:`
- `Decision:`
- `Next Idea:`

### 3.2 反復を止めてよい条件

反復は、少なくとも次のどれかを満たしたときに止めます。

- 事前に決めた `Stop Condition:` を満たした
- 追加変更を入れても改善が見えず、`Critical Review:` でそれを説明できる
- fairness を保った比較で、現時点の claim と limitation が十分に整理できた
- それ以上の実験が、別 topic や別 branch に分けるべき新しい問いへ変わった

## 4. 個別 README の位置づけ

`experiments/<topic>/README.md` は引き続き必要です。
ただし、役割はこの文書と分けます。

- この文書
  - 実験全般の標準手順
- topic README
  - その実験固有の目的、入力、CLI、`result/<run_name>/` の置き場、`experiments/report/<run_name>.md` の置き場、run_name 規則、既知の注意点

個別 README は「そのモジュールや実験をどう使うか」を書き、
この文書は「repo で実験をどう進めるか」を書く、という分担にします。

## 5. References

ローカルで読める索引は次です。

- [experiment_workflow/README.md](/workspace/references/experiment_workflow/README.md)
- [generative_ai/README.md](/workspace/references/generative_ai/README.md)

### 実験手順・再現性

- [Sandve et al. (2013), Ten Simple Rules for Reproducible Computational Research](/workspace/references/experiment_workflow/Sandve_2013_Ten_Simple_Rules_for_Reproducible_Computational_Research.pdf)
- [Wilson et al. (2014), Best Practices for Scientific Computing](/workspace/references/experiment_workflow/Wilson_2014_Best_Practices_for_Scientific_Computing.pdf)
- [Wilson et al. (2017), Good Enough Practices in Scientific Computing](/workspace/references/experiment_workflow/Wilson_2017_Good_Enough_Practices_in_Scientific_Computing.pdf)
- [Nature, Guidance on Reproducibility for Papers Using Computational Tools](/workspace/references/experiment_workflow/Nature_Guidance_on_Reproducibility_for_Papers_Using_Computational_Tools.pdf)
- [Bartz-Beielstein et al. (2020), Benchmarking in Optimization: Best Practice and Open Issues](/workspace/references/experiment_workflow/Bartz-Beielstein_2020_Benchmarking_in_Optimization_Best_Practice_and_Open_Issues.pdf)

### 批判的レビュー・図表

- [Minocher et al. (2023), Implementing Code Review in the Scientific Workflow](/workspace/references/experiment_workflow/Minocher_2023_Implementing_Code_Review_in_the_Scientific_Workflow.html)
- [Tiwari et al. (2021), Reproducibility in Systems Biology Modelling](/workspace/references/experiment_workflow/Tiwari_2021_Reproducibility_in_Systems_Biology_Modelling.pdf)
- [Rougier et al. (2014), Ten Simple Rules for Better Figures](/workspace/references/experiment_workflow/Rougier_2014_Ten_Simple_Rules_for_Better_Figures.pdf)

### 生成AIの活用

- [Rethinking the AI Scientist: Interactive Multi-Agent Workflows for Scientific Discovery](/workspace/references/generative_ai/Rethinking_the_AI_Scientist_Interactive_Multi-Agent_Workflows_for_Scientific_Discovery.pdf)
- [Towards Scientific Discovery with Generative AI: Progress, Opportunities and Challenges](/workspace/references/generative_ai/Towards_Scientific_Discovery_with_Generative_AI_Progress_Opportunities_and_Challenges.pdf)
- [Wu et al. (2025), Automated Literature Research and Review-Generation Method Based on Large Language Models](/workspace/references/generative_ai/Wu_2025_Automated_Literature_Research_and_Review_Generation_Method_Based_on_LLMs.html)
- [OpenReviewer: A Specialized Large Language Model for Generating Critical Scientific Paper Reviews](/workspace/references/generative_ai/OpenReviewer_A_Specialized_Large_Language_Model_for_Generating_Critical_Scientific_Paper_Reviews.pdf)
