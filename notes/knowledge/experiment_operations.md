```markdown
# Experiment Operations

## 保存

- 長時間 run は JSONL を逐次保存する。
- final JSON だけに頼らない。
- child が終わったら明示的に completion record を返す。
- failure kind は `error`, `oom`, `worker_terminated`, `timeout` のように分ける。

## ケース順

- case ordering は実験設計の一部と考える。
- `dimension -> level -> dtype` は frontier を読みたいときに向く。
- `dtype -> level -> dimension` は途中停止時に比較が偏りやすい。
- 途中停止しても何が読めるかを先に考える。

## main への持ち帰り

- raw JSONL は results branch に残す。
- `main` には、再集計できる final JSON を持ち帰る。
- note には source JSON と archived JSON の両方を書く。
- 外向けに残したい図は `notes/assets/` にコピーする。

## worktree の後始末

- 削除前に `notes/worktrees/` へ吸い出す。
- 少なくとも、用途、主要結果、分かったこと、次の案を書く。
- 過去の note 本文は原則書き換えず、必要なら追記で補う。

## 監視

- `nvidia-smi`
- `ps` / `pgrep`
- JSONL の行数
- `failure_kind`
- `integrator_init_seconds`
- `process_rss_mb`

## よくある失敗

- 実験を止めたあと partial を作らずに放置する。
- final JSON を `main` に持ち帰らず、あとで図を再生成できなくなる。
- case ordering のせいで比較に必要な dtype がほとんど残らない。

## 2026-03-18: 当日の学び

- ワークツリー運用では `.worktrees/` 配下に複数作業を並行保持できるため、実装作業はワークツリー単位で管理する。
- experiment-runner の変更（`python/jax_util/experiment_runner/*`）はテストとドキュメントを同時に main に統合する必要がある。
- コンテナ依存の更新は `docker/requirements.txt` と `docker/Dockerfile` の両方を更新する運用が安全。

:
```
git push origin main
```

```
