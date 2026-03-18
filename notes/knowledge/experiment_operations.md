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

### 運用の具体化（チェックリスト）

- ワークツリーで作業開始前に `git fetch origin main` を実行し、`scripts/setup_worktree.sh` で作成される `WORKTREE_SCOPE.md` に必須参照を埋める。
- 実験・実装を main に取り込む際は以下を行う：
  - ユニットテストをワークツリーで実行（`python/tests/*`）
  - ドキュメント（`documents/`）の更新を同時に main へ統合
  - コンテナ依存が増える場合は `docker/requirements.txt` と `docker/Dockerfile` を同時に更新し、CIでビルド確認

### 参考コマンド

```
# ワークツリー作成（例）
bash scripts/setup_worktree.sh experiment-runner-generalization "GPU scheduler work"

# メインへ取り込む（例: cherry-pick）
git fetch origin main
git checkout main
git cherry-pick <commit-hash>
```

### メモ（今日の具体例）

- スコープテンプレート運用変更: `scripts/setup_worktree.sh` を heredoc 生成から `documents/WORKTREE_SCOPE_TEMPLATE.md` のコピー方式に変更（commit: `514b2f6`）。
- experiment-runner の統合では `8fc94e1` を cherry-pick → コンフリクトを手動で解消し、関連テストとドキュメントを `main` に追加（最終的に `1fe605c` 系で反映）。
