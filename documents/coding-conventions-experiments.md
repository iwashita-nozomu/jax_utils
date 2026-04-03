# 実験環境の運用規約

この文書は、`experiments/` 配下の実験コード・ベンチマーク・実験結果・実行環境の運用を対象にします。
長時間実行と生成物の管理を安定させ、`main` のコード編集と衝突させないことを目的にします。
研究の問い、数式、比較対象、逐次改造の記録方法は `documents/research-workflow.md` を正本とします。
準備、実装、静的チェック、実行、結果レポートを通した標準手順の統合入口は `documents/experiment-workflow.md` を参照してください。
批判的レビューの観点は `documents/experiment-critical-review.md` を参照してください。

## 1. 対象

- 対象は `experiments/` 配下の実験コードと benchmark コードです。
- 対象には、実験実行スクリプト、レポート生成スクリプト、結果保存ディレクトリを含みます。
- 実験コードそのものを `main` に置くことは許可します。ただし、生成物は code と分けて扱います。
- topic 固有の helper module は `experiments/` に置き、再利用する runtime は `python/experiment_runner/` に分けます。
- 実験コードは 1 回の fresh 実行で対象レンジを完走する前提で書き、resume や途中 run の継ぎ足しを正規運用にしません。

## 2. ディレクトリ構成

- 実験は `experiments/<topic>/` に置きます。
- 各実験ディレクトリには、少なくとも次の 4 つを置きます。
  - `README.md`
  - `cases.py`
  - `experimentcode.py`
  - `result/` ディレクトリ
- `result/` には `.gitkeep` を置き、空ディレクトリでも構造を保ちます。
- 実験コードは再実行可能なスクリプトとして保ち、手作業前提の手順を埋め込みません。
- 1 回の run は 1 つの `run_name` と 1 つの出力先に閉じた完全実行として扱います。
- benchmark は topic に近い `experiments/` 配下へ置き、グローバルな `python/benchmark/` は既定にしません。
- benchmark の短時間運用は `documents/conventions/python/20_benchmark_policy.md` を、topic ごとの配置規約は `documents/conventions/python/30_experiment_directory_structure.md` を参照します。

### repo 全体での置き場

- 実験コードと runtime 生成物の正本は `experiments/` 配下に置きます。
- 1 回の run に対応する experiment report の正本は `experiments/report/` に置きます。
- 複数 run をまたぐ要約や知見整理は `notes/experiments/` や `notes/themes/` に置きます。
- 詳細な作業ログが必要な場合に限って `notes/worktrees/` の使用を許可します。
- top-level の `reports/` は project-wide な review、automation、management report の置き場であり、topic ごとの experiment report の正本にはしません。

### 実験コードの体裁

- topic ごとに、少なくとも `README.md`、`cases.py`、`experimentcode.py`、`result/` を意識した構成にします。
- `experimentcode.py` は orchestration と CLI に集中させ、case 定義や設計意図を抱え込みません。
- その topic に固有の case 生成と difficulty 設計は `cases.py` に寄せます。
- 正式な比較プロトコルでは、次元、level、problem size のような ordered difficulty 軸を 1 ずつ上げる連続 sweep にします。
- 飛び飛びの点だけを回す coarse sweep は debug / smoke / 予備探索に限り、正式な比較プロトコルでは理由を明記します。
- 実験 README には、問い、比較対象、標準 run、結果出力先、`experiments/report/` への入口、命名規則を明記します。

## 3. worktree とブランチ

- 長時間実験でも、branch や worktree を必須にはしません。
- worktree の作成と削除の共通ルールは `documents/worktree-lifecycle.md` を参照します。
- 同じ branch で実験コード、結果、report を扱うことを既定とします。
- 別 branch / worktree の使用は、巨大生成物の退避、長時間 run の隔離、破壊的な試行の切り分けに限って許可します。
- 別 worktree を使う場合は、VS Code やホスト OS から見える共有パスに置き、既定では `./.worktrees/<name>` を使います。
- `experimenter` を使う場合は、`WORKTREE_SCOPE.md` に `## Runtime Output Directories` を明記します。
- 詳細な action log が必要な場合に限って `notes/worktrees/worktree_<topic>_YYYY-MM-DD.md` の使用を許可します。

## 4. 生成物の扱い

- 生成された JSON、HTML、SVG、ログは code と分けて管理します。
- 生成物は `result/<run_name>/` に集約し、ソースコードの隣へ散らしません。
- runtime 生成物の canonical な置き方は次を既定にします。
  - `result/<run_name>/summary.json`
  - `result/<run_name>/cases.jsonl`
  - `result/<run_name>/run.log`
  - 図を出力する場合は `result/<run_name>/figures/`
- 空ログや途中失敗だけを示すログは、継続的な参照価値がない限り commit しません。
- 途中停止した run の JSONL や partial 集計は診断用の補助出力であり、正規の再開点として使いません。
- 1 回の run に対応する人が読む experiment report は `experiments/report/<run_name>.md` を正本にします。

## 4.5 命名

- topic ディレクトリ名は `snake_case` にします。
- run_name は `<topic>_<variant>_<YYYYMMDDTHHMMSSZ>` にします。
  - `variant` には `cpu`, `gpu`, `smoke`, `verified` のような比較に必要な最小限の識別子だけを入れます。
- per-run の生成物は run_name を親ディレクトリ名に使います。
  - `result/<run_name>/summary.json`
  - `result/<run_name>/cases.jsonl`
  - `result/<run_name>/run.log`
- partial や diagnostic の出力は `_partial` や `_debug` を run_name の suffix に付けることを許可します。ただし、正式 report の正本として扱うことを禁止します。
- report は `experiments/report/<run_name>.md` を既定にします。
- naming rule は script の中だけに閉じず、対応する `experiments/<topic>/README.md` と必要な正本文書に必ず残します。

## 5. 実行スクリプト

- 実行スクリプトは CLI 引数で条件を変更できるようにします。
- 実行スクリプトは、指定した条件集合を 1 回で最後まで走り切る責務を持たせます。
- 実行前に、比較対象、主要 metric、仮定、停止条件を対応する note に明記します。
- benchmark は短時間の前後比較を目的とし、長時間の多条件 sweep は experiment に分けます。
- 少なくとも、次の切り替えは引数化します。
  - 次元レンジ
  - レベルレンジ
  - dtype
  - platform
  - timeout
- GPU 実験では、JAX の GPU メモリ先取りを無効化して実行します。
- GPU ごとのケース実行は逐次にし、同じ GPU で前ケースのプロセスが終了してから次ケースを開始します。
- 長時間実験では、case ごとに fresh process を立てて終了させ、メモリを次ケースへ持ち越しません。
- 実行結果は JSON として保存し、出力パスを最後に標準出力へ出します。
- 長時間実験では、最終 JSON に加えて case 単位の JSONL を逐次保存します。
- 実行スクリプトは、使う run_name と canonical な出力先を README と整合する形で決めます。
- JSONL は case 完了時点で追記し、競合を避ける排他制御を入れます。
- JSONL の役割は run 中の progress 記録と failure 診断であり、途中停止後の resume 入力にはしません。
- run が途中で止まった場合は、その run を完了扱いにせず、新しい run_name と新しい出力先で 0 から再実行します。
- 失敗ケースは握りつぶさず、GPU OOM、host OOM、timeout、worker 異常終了を区別できる範囲で記録します。

### runner 側に寄せる責務

- process の fresh 起動と終了
- GPU slot / worker slot の管理
- child へ渡す `environment_variables` の構築
- GPU 可視性と allocator 系 env の反映
- scheduler に基づく case の順序決定

### 実験コード側で重複実装しないこと

- 独自の mini-runner
- `Popen` ベースの独自 worker 管理
- `CUDA_VISIBLE_DEVICES` や `XLA_*` の直設定
- JAX / XLA env を if 文で場当たり的に組み立てること
- timeout、signal、native crash の parent-side cleanup を script 側で重複実装すること
- partial run を前提にした resume protocol
- その場限りの ad hoc output path 命名

JAX / XLA env が必要な場合は `jax_util.xla_env` を正本として使います。
child / parent diagnostics、timeout、強制停止、completion 欠落時の failure record は `experiment_runner` を正本として扱います。

### Spot Run 禁止

- 事前に定義した比較プロトコルに入っていない ad hoc 実行を、正式な experiment evidence に使いません。
- 1 case だけの単発実行、README にない subset 実行、途中停止 run のつまみ食いを benchmark 結果として扱いません。
- debug / smoke 用の単発実行は許可します。ただし、`Debug Run:` や `Smoke:` として明示し、`summary.json` や carry-over の正本として扱うことを禁止します。

## 6. レポートと再現性

- 可視化は実験後処理として分け、実験本体と同じスクリプトへ混ぜません。
- レポート生成スクリプトは、既存の結果 JSON だけで再生成できるようにします。
- レポート本文の構成、Abstract、Results / Discussion の書き分け、Figure / Table caption のルールは `documents/experiment-report-style.md` を正本とします。
- figure には軸名と単位を付け、scale が linear か log かも明示します。
- figure caption または本文に、`How to read:` に相当する読み取り方を最低 1 文入れます。
- 実験結果には、少なくとも条件、成功/失敗、主要な誤差指標、時間、メモリ指標を残します。
- 実験結果には、対応する worktree、branch、commit、実行スクリプトパスも残します。
- 実験結果には、対応する定式化、比較対象、採用した fairness 条件を note 側から辿れるようにします。
- 小さい確認実験と大規模実験は同じ形式の JSON へ保存し、後段の処理を共通化します。

### 集計と考察

- `summary.json` には、少なくとも case 数、成功率、failure kind ごとの件数、主要 metric の代表値を含めます。
- 平均だけでなく、中央値や最小 / 最大も残します。分位点が必要な場合は追加します。
- baseline 比較を行う実験では、差分または比率を note か `summary.json` のどちらかで必ず辿れるようにします。
- note 側には、`Quantitative Summary:` と `Comparison Table:` と `Critical Review:` を残して、数字、読み、反論可能性を分けます。
- report 側では、`Results` で数字を出し、`Discussion` で解釈と先行比較を書きます。1 段落の中で観測と推測を混ぜません。
- 結論には、根拠となる figure / table を本文中で明示的に張ります。
- 差を見る場合は linear、桁差や比を見る場合は log を使います。使い分け理由は読者に見える場所へ書きます。
- 改善を主張するときは、改善しなかった条件帯、failure 増加、memory 悪化のような反証側も同じ note で扱います。

## 7. `main` へ残すもの

- `main` には再生成可能な code、文書、最小限の雛形だけを残します。
- 実験そのものの code を `main` に置くことは許可します。
- 巨大な JSON、画像、ログは `main` へ常設しません。
- 実験の所在と 1 回ごとの report は `./experiments/` 配下に残します。
- 複数 run をまたぐ定性的な考察を `./notes/experiments/` に残すことは許可します。
- `main` に持ち帰る JSON は、対応する `result/<run_name>/summary.json` をそのまま辿れる形にします。
- `./notes/experiments/` のメモは実験ごとに分け、複数の unrelated な実験を 1 ファイルへ混ぜません。
- `./notes/experiments/` では、文献由来の内容に対象文献を明示し、`Idea:`、`Interpretation:`、`Consideration:` などで自前の発案や考察を区別します。
- `./notes/experiments/` に一度残した過去の実験メモ本文は書き換えません。
- 追加の知見や訂正は、既存本文を直さず `Addendum:` や `Correction:` として追記するか、新しい日付付きメモを作って参照します。
- `./notes/experiments/` のタイトルは日付ではなく実験トピックを主にし、日付は副情報として扱います。
- 実験メモには、最低でも source branch、source JSON/JSONL、対応する report と `result/<run_name>/summary.json`、主要観測、次の action を同じファイル内に残します。
- 削除した worktree の固有メモは `main` の `./notes/worktrees/` へ残し、worktree 自体を消しても判断の流れを追えるようにします。
- 日付依存の強い実験ログや途中判断を `main` の `./diary/` に残すことは許可します。
- `./notes/experiments/` や `./notes/themes/` から `summary.json` や図へ張るリンクは、できるだけ相対パスで書きます。
- topic ごとの experiment report を top-level `./reports/` へ増やさず、`./experiments/report/` を正本にします。
- 実験用 worktree の一時パスを、`main` 側 Markdown の恒久リンクとして使いません。
- `./diary/` は日付ファイルへ逐次追記する運用を基本とし、その日の流れが読めるように保ちます。
- `./diary/` でも、文献に基づく記述は出典を明示し、未確認の仮説は断定しません。
- `./diary/` の過去日の本文は書き換えません。補足や訂正は同じファイル末尾へ追記して残します。
- 実験環境の運用ルールを変えた場合は、この文書と `coding-conventions-project.md` を同時に更新します。
