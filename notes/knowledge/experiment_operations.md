# Experiment Operations

## 保存

- 長時間 run は JSONL を逐次保存する。
- final JSON だけに頼らない。
- child が終わったら completion record を返し、`failure_kind` を分類する。
- timeout、signal 終了、completion 欠落も parent 側で診断に寄せる。

## ケース順

- case ordering は実験設計の一部と考える。
- `dimension -> level -> dtype` は frontier を読みたいときに向く。
- `dtype -> level -> dimension` は途中停止時に比較が偏りやすい。
- 途中停止しても何が読めるかを先に考える。

## main への持ち帰り

- raw JSONL は results branch に残す。
- `main` には再集計できる final JSON を持ち帰る。
- note には source JSON と archived JSON の両方を書く。
- 外向けに残したい図は `notes/assets/` にコピーする。
- experiment-runner の変更は、テストとドキュメントを同時に main に統合する。
- reusable runtime は `python/experiment_runner/` に残し、topic 固有ロジックは `experiments/<topic>/` に残す。

## worktree の後始末

- 削除前に `notes/worktrees/` へ吸い出す。
- 少なくとも、用途、主要結果、分かったこと、次の案を書く。
- 過去の note 本文は書き換えず、補足が必要な場合は追記で補う。
- 実験中の一挙手一投足は action log として逐次残し、削除前に要約へ圧縮する。

## 監視

- `nvidia-smi`
- `ps` / `pgrep`
- JSONL の行数
- `failure_kind`
- `integrator_init_seconds`
- `process_rss_mb`
- 中断後の child 残骸

## よくある失敗

- 実験を止めたあと partial を作らずに放置する。
- final JSON を `main` に持ち帰らず、あとで図を再生成できなくなる。
- case ordering のせいで比較に必要な dtype がほとんど残らない。
- parent を止めたあと child worker を回収せず、次の GPU run を汚す。
- experiment code 側で env と process lifecycle を抱え込み、runner 共通部と責務がぶつかる。

## 運用チェックリスト

- ワークツリー作成前に `git fetch origin main` を実行する。
- `scripts/setup_worktree.sh` で作られた `WORKTREE_SCOPE.md` を埋める。
- `notes/worktrees/worktree_<topic>_YYYY-MM-DD.md` を action log の正本にする。
- 実験・実装を main に取り込む前に、関連ユニットテストをワークツリーで実行する。
- ドキュメント更新はコード変更と同時に持ち帰る。
- 依存が増える変更では `docker/requirements.txt` と `docker/Dockerfile` を同時に更新する。
- 中断した長時間 run を再開する前に、`ps` と `nvidia-smi` で child の残骸がないことを確認する。

## 関連

- benchmark との使い分けは [Benchmark vs Experiment](/workspace/notes/knowledge/benchmark_vs_experiment.md) を参照する。
- 正本の運用規約は [documents/coding-conventions-experiments.md](/workspace/documents/coding-conventions-experiments.md) を参照する。
