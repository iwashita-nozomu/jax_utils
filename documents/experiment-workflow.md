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
  - final JSON と note に何を残すか。少なくとも時間、成功率、failure kind、主要誤差を含めます。
- `Stop Condition:`
  - smoke で止めるのか、verified まで進めるのか、正式な比較表や report まで必要なのか。
- `Fairness Notes:`
  - 同じ case set、同じ timeout、同じ hardware、同じ seed policy、同じ allocator 方針をどこまで維持するか。

次に、branch と worktree を分けます。

- コード改造用
  - `work/<topic>-<date>` branch を使います。
- 長時間 run 用
  - `results/<topic>` branch を使います。
- worktree log
  - `notes/worktrees/worktree_<topic>_YYYY-MM-DD.md` に開始時点の問い、比較対象、run 方針を書きます。

準備段階で確認するものは次です。

- topic の `README.md`
- 直近の experiment note
- final JSON / JSONL schema
- `git status --short`

`Interpretation:`
準備段階の目的は、今回の run が debug なのか、verified なのか、正式比較なのかを曖昧にしないことです。

### 2. 実験コードの実装

実験コードは、「問いと比較」を表現する薄い層として実装します。
process 管理や GPU 割当は runner 側の責務であり、実験 script 側に重複実装しません。

推奨構成は次です。

- `README.md`
  - 問い、比較対象、標準コマンド、出力先、carry-over 方針を書く。
- `cases.py`
  - case 定義、difficulty range、resource estimate を置く。
- `runner_config.py`
  - `smoke`, `small`, `verified`, `medium` のような size preset と timeout policy を置く。
- `results_aggregator.py`
  - JSONL から final JSON を作る。
- `run_*.py`
  - orchestration と CLI に集中させる。
- `results/`
  - JSON、JSONL、HTML、SVG、ログなどの生成物を置く。

`experiment_runner` を使う場合の入口は次です。

- 一般的な実験
  - `StandardWorker`
  - `StandardScheduler` または `StandardFullResourceScheduler`
  - `StandardRunner`
- host 側で pid / worker slot を強く観測したい実験
  - `build_worker_slots()`
  - `run_cases_with_subprocess_scheduler()`
  - 必要なら `RuntimeMonitor`

実装時にやらないことは次です。

- 実験 script 内で独自の mini-runner を書く
- GPU slot 管理を script 側で持つ
- `CUDA_VISIBLE_DEVICES` や `XLA_*` を script 側で直接組み立てる
- partial run を前提にした resume protocol を作る
- ad hoc な result path 命名を増やす

### 3. 実験コードの静的チェック

長時間 run の前に、少なくとも静的に次を確認します。

- `pyright`
  - 実験コード、`experiment_runner` 利用部分、集計コードの型整合を確認する。
- `ruff check`
  - import、未使用変数、到達不能コード、雑な例外処理を早めに落とす。
- CLI help
  - `python run_*.py --help` が通ることを確認する。
- import path
  - top-level import と package path が壊れていないことを確認する。
- 出力 schema
  - final JSON に必要な key が揃うよう、集計コードを静的に読んでおく。

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
- final JSON が生成される
- report 再生成の入口が成立する

#### 4.2 verified

本番に近い backend と env で、worker 数を絞って narrow run を行います。
ここで確認するのは次です。

- GPU visibility
- allocator 設定
- worker slot / timeout の挙動
- failure kind の記録
- final JSON と JSONL の整合

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

#### 4.4 long run の原則

- 長時間 run は `results/<topic>` worktree で行います。
- run は 1 つの run_id と 1 つの出力先に閉じた fresh 実行として扱います。
- case ごとの JSONL は progress 記録と failure 診断のために保存します。
- partial run は診断材料としては残してよいですが、正規の再開点にはしません。
- 止まった run は `Stop Reason:` と `Restart Decision:` を log に残し、新しい run_id で 0 からやり直します。

#### 4.5 monitor

host 側で worker 状態や GPU 利用状況を見たい場合は、`RuntimeMonitor` を使います。
ただし monitor は evidence そのものではなく、run の観測補助です。
正式な evidence は最終的に JSON、JSONL、report、note へ落とします。

### 5. 結果レポート

run 後は、必ず結果を report と note に整理します。

最低限残すものは次です。

- final JSON
- raw JSONL
- report 再生成コマンド
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

carry-over の原則は次です。

- raw JSON、HTML、SVG、ログ本体は results branch に残す
- `main` には note と必要最小限の final JSON だけを持ち帰る
- `main` に持ち帰る JSON は完走した fresh run の final JSON を原則とする
- partial run の JSON は carry-over の正本にしない

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
1. `implementer`
   - review を反映し、静的チェックを通す。
1. `experimenter`
   - 同じ protocol で fresh run を実行する。
1. `experimenter`
   - final JSON、raw JSONL、report、experiment note を生成する。
1. `experiment_reviewer`
   - report と結果の読み方を批判的にレビューする。
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
- final JSON / raw JSONL の所在
- report の所在
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
  - その実験固有の問い、入力、CLI、出力、results branch 名、既知の注意点

個別 README は「そのモジュールや実験をどう使うか」を書き、
この文書は「repo で実験をどう進めるか」を書く、という分担にします。
