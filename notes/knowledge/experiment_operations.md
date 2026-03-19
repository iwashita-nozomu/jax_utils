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
# Experiment Operations

## 保存

- 長時間実行は JSONL を逐次保存する。
- final JSON のみへ依存しない。
- 実行ごとに完了レコードを作成し、failure_kind を分類する。

## ケース設計

- case ordering は実験設計の一部として意図的に決める。
- 途中停止でも有用な出力が得られる順序を優先する。

## main への持ち帰り

- raw JSONL は results ブランチで保管し、main には再集計可能な final JSON を持ち帰る。
- 出力図や公開資料は `notes/assets/` に格納する。
- ドキュメントとテストは同時に統合する。

## worktree の後始末

- 削除前に `notes/worktrees/` に要約（目的・主要結果・次案）を残す。
- 過去ノートは上書きせず追記で補う。

## 監視指標

- GPU・プロセス・メモリ（`nvidia-smi`, `ps`/`pgrep`, RSS）
- JSONL 行数、failure_kind、初期化時間など

## よくある失敗

- partial を残して放置する
- final を main に持ち帰らない
- 比較に必要な構成が失われる

## 運用チェックリスト

- ワークツリー作成前に最新の main を取得する
- ワークツリー内でユニットテストを実行する
- 依存が増える変更はコンテナ定義を同時に更新する
```
