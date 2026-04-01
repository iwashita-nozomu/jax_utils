# 実験環境の運用規約

この文書は、`experiments/` 配下の実験コード・ベンチマーク・実験結果・実行環境の運用を対象にします。
長時間実行と生成物の管理を安定させ、`main` のコード編集と衝突させないことを目的にします。
研究の問い、数式、比較対象、逐次改造の記録方法は `documents/research-workflow.md` を正本とします。

## 1. 対象

- 対象は `experiments/` 配下の実験コードと benchmark コードです。
- 対象には、実験実行スクリプト、レポート生成スクリプト、結果保存ディレクトリを含みます。
- 実験コードそのものは `main` に置いてよいですが、生成物は code と分けて扱います。
- topic 固有の helper module は `experiments/` に置き、再利用する runtime は `python/experiment_runner/` に分けます。
- 実験コードは 1 回の fresh 実行で対象レンジを完走する前提で書き、resume や途中 run の継ぎ足しを正規運用にしません。

## 2. ディレクトリ構成

- 単独で完結する実験は `experiments/<topic>/`、area ごとに束ねる必要がある実験は `experiments/<area>/<topic>/` の形で配置します。
- 各実験ディレクトリには、少なくとも次の 3 種類を分けて置きます。
  - 実行スクリプト
  - 結果整理・可視化スクリプト
  - `results/` ディレクトリ
- `results/` には `.gitkeep` を置き、空ディレクトリでも構造を保ちます。
- 実験コードは再実行可能なスクリプトとして保ち、手作業前提の手順を埋め込みません。
- 1 回の run は 1 つの run_id と 1 つの出力先に閉じた完全実行として扱います。
- benchmark は topic に近い `experiments/` 配下へ置き、グローバルな `python/benchmark/` は既定にしません。
- benchmark の短時間運用は `documents/conventions/python/20_benchmark_policy.md` を、topic ごとの配置規約は `documents/conventions/python/30_experiment_directory_structure.md` を参照します。

### 実験コードの体裁

- topic ごとに、少なくとも `README.md`、`cases.py`、`runner_config.py`、`results_aggregator.py`、`run_*.py`、`results/` を意識した構成にします。
- `run_*.py` は orchestration と CLI に集中させ、case 定義、resource estimate、集計ロジックを抱え込みません。
- その topic に固有の case 生成と difficulty 設計は `cases.py` に寄せます。
- 次元、level、problem size のような ordered difficulty 軸は、原則として 1 ずつ上げる連続 sweep を基本にします。
- 飛び飛びの点だけを回す coarse sweep は debug / smoke / 予備探索に限り、正式な比較プロトコルでは理由を明記します。
- size preset や timeout policy は `runner_config.py` に寄せます。
- final JSON 生成や summary 集計は `results_aggregator.py` に寄せます。
- 実験 README には、問い、比較対象、標準 run、結果出力先、carry-over 方針を明記します。

## 3. worktree とブランチ

- 長時間実験は、必ず専用の git worktree 上で実行します。
- worktree の作成と削除の共通ルールは `documents/worktree-lifecycle.md` を参照します。
- `main` の worktree では、コード編集・文書更新・通常テストのみを行います。
- 実験用 worktree は、VS Code やホスト OS から見える共有パスに置きます。
- 既定の配置は `./.worktrees/<name>` とします。
- 実験結果を保存するブランチは `results/<topic>` 形式を原則とします。
- 実験用 worktree は results ブランチに対応づけ、同じ worktree のまま branch を切り替えて使い回しません。
- 実験開始前に、`main` と対象の results ブランチの両方を `origin` と同期します。
- 実験開始前に、`main` worktree と対象 results worktree の両方が clean であることを確認します。
- `experimenter` を使う場合は、`WORKTREE_SCOPE.md` に `## Runtime Output Directories` を明記します。
- 実験固有のコード変更は、まず対応する results ブランチの worktree で行います。
- 実験ブランチ上で十分に検証した変更だけを、最後に `main` へ merge します。
- 実験固有ではない共通コード変更を `main` で先に行った場合だけ、results worktree へ `main` を取り込みます。
- worktree を削除する前に、その worktree にしか残っていない知見を `main` の `./notes/worktrees/` へ吸い出して整理します。
- 吸い出しでは、少なくとも branch 名、worktree の用途、関連結果、主要な観測、次の `Idea:` を残します。
- 実験 branch と results branch は、`main` の `./notes/branches/README.md` から参照できるようにし、関連する experiment note や worktree note への入口を置きます。
- 実験 worktree では、scope 更新、条件変更、run 開始/停止、final JSON 採用、統合判断を action log に逐次記録します。
- 実験 worktree では、中断理由と fresh run への切り替え判断も action log に逐次記録します。
- 実験 worktree では、問い、定式化、比較対象、各改造の狙いも action log または experiment note に逐次記録します。
- action log の既定位置は `notes/worktrees/worktree_<topic>_YYYY-MM-DD.md` とし、worktree 内でも同じ相対パスで下書きします。

## 4. 生成物の扱い

- 生成された JSON、HTML、SVG、ログは code と分けて管理します。
- 生成物は `results/` に集約し、ソースコードの隣へ散らしません。
- `latest.json` と `latest_report/` は、直近結果への入口として置いてよいです。
- `latest.json` は `main` ではノイズにしないよう扱い、results ブランチ側で必要に応じて更新します。
- 空ログや途中失敗だけを示すログは、継続的な参照価値がない限り commit しません。
- 途中停止した run の JSONL や partial 集計は診断用の補助出力であり、正規の再開点として使いません。

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
- JSONL は case 完了時点で追記し、競合を避ける排他制御を入れます。
- JSONL の役割は run 中の progress 記録と failure 診断であり、途中停止後の resume 入力にはしません。
- run が途中で止まった場合は、その run を完了扱いにせず、新しい run_id と新しい出力先で 0 から再実行します。
- 失敗ケースは握りつぶさず、GPU OOM、host OOM、timeout、worker 異常終了を区別できる範囲で記録します。
- 実験スクリプトの先頭には、対応する results ブランチ名をコメントで明記します。

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
- partial run を前提にした resume protocol
- その場限りの ad hoc output path 命名

### Spot Run 禁止

- 事前に定義した比較プロトコルに入っていない ad hoc 実行を、正式な experiment evidence に使いません。
- 1 case だけの単発実行、README にない subset 実行、途中停止 run のつまみ食いを benchmark 結果として扱いません。
- debug / smoke 用の単発実行は許可しますが、`Debug Run:` や `Smoke:` として明示し、final JSON や carry-over の正本にしません。

## 6. レポートと再現性

- 可視化は実験後処理として分け、実験本体と同じスクリプトへ混ぜません。
- レポート生成スクリプトは、既存の結果 JSON だけで再生成できるようにします。
- レポート本文の構成、Abstract、Results / Discussion の書き分け、Figure / Table caption の原則は `documents/experiment-report-style.md` を正本とします。
- figure には軸名と単位を付け、scale が linear か log かも明示します。
- figure caption または本文に、`How to read:` に相当する読み取り方を最低 1 文入れます。
- 実験結果には、少なくとも条件、成功/失敗、主要な誤差指標、時間、メモリ指標を残します。
- 実験結果には、対応する worktree、branch、commit、実行スクリプトパスも残します。
- 実験結果には、対応する定式化、比較対象、採用した fairness 条件を note 側から辿れるようにします。
- 小さい確認実験と大規模実験は同じ形式の JSON へ保存し、後段の処理を共通化します。

### 集計と考察

- final JSON には、少なくとも case 数、成功率、failure kind ごとの件数、主要 metric の代表値を含めます。
- 平均だけでなく、中央値や最小 / 最大、必要なら分位点も残します。
- baseline 比較を行う実験では、差分または比率を note か final JSON のどちらかで必ず辿れるようにします。
- note 側には、`Quantitative Summary:` と `Comparison Table:` と `Critical Review:` を残して、数字、読み、反論可能性を分けます。
- report 側では、`Results` で数字を出し、`Discussion` で解釈と先行比較を書きます。1 段落の中で観測と推測を混ぜません。
- 結論には、根拠となる figure / table を本文中で明示的に張ります。
- scale は、差を見るなら linear、桁差や比を見るなら log を原則とし、使い分け理由を読者に見える場所へ書きます。
- 改善を主張するときは、改善しなかった条件帯、failure 増加、memory 悪化のような反証側も同じ note で扱います。

## 7. `main` へ残すもの

- `main` には再生成可能な code、文書、最小限の雛形だけを残します。
- 実験そのものの code は `main` に置いてよいです。
- 巨大な JSON、画像、ログは `main` へ常設しません。
- 実験の所在、branch 名、定性的な考察は `main` の `./notes/experiments/` に残してよいです。
- ただし、後から図や集計を再生成できるよう、各 branch の最終結果として必要最小限の JSON は `main` の `./notes/experiments/results/` へ持ち帰ります。
- `main` に持ち帰る JSON は完走した run の final JSON を原則とし、途中停止 partial run は carry-over の正本にしません。
- `./notes/experiments/` のメモは、原則として実験ごとに分け、複数の unrelated な実験を 1 ファイルへ混ぜません。
- `./notes/experiments/` では、文献由来の内容に対象文献を明示し、`Idea:`、`Interpretation:`、`Consideration:` などで自前の発案や考察を区別します。
- `./notes/experiments/` に一度残した過去の実験メモ本文は、原則として書き換えません。
- 追加の知見や訂正は、既存本文を直さず `Addendum:` や `Correction:` として追記するか、新しい日付付きメモを作って参照します。
- `./notes/experiments/` のタイトルは日付ではなく実験トピックを主にし、日付は副情報として扱います。
- 実験メモには、最低でも source branch、source JSON/JSONL、`main` に持ち帰った final JSON、主要観測、次の action を同じファイル内に残します。
- 削除した worktree の固有メモは `main` の `./notes/worktrees/` へ残し、worktree 自体を消しても判断の流れを追えるようにします。
- 日付依存の強い実験ログや途中判断は `main` の `./diary/` に残してよいです。
- `./notes/experiments/` や `./notes/themes/` から final JSON や図へ張るリンクは、できるだけ相対パスで書きます。
- 実験用 worktree の一時パスを、`main` 側 Markdown の恒久リンクとして使いません。
- `./diary/` は日付ファイルへ逐次追記する運用を基本とし、その日の流れが読めるように保ちます。
- `./diary/` でも、文献に基づく記述は出典を明示し、未確認の仮説は断定しません。
- `./diary/` も過去日の本文は原則として書き換えず、補足や訂正は同じファイル末尾へ追記して残します。
- 実験環境の運用ルールを変えた場合は、この文書と `coding-conventions-project.md` を同時に更新します。
