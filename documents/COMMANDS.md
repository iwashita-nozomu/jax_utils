# 開発・解析コマンド集

この文書は、リポジトリで推奨するコマンド操作とワークフローをまとめた規約です。
すべての操作は原則ワークツリー/ブランチ上で行い、main への直接編集は避けてください。

## 前提
- Docker 開発コンテナ内で作業することを想定しています。
- 主要ツール: `ruff`, `black`, `pyright`, `pytest`。
- 解析結果は `reports/static-analysis/` に保存してください。

## 基本コマンド

### 依存関係のインストール

```bash
# 開発依存をローカル環境へ
pip install -e ".[dev]"
# Docker 環境で必要なツールは docker/requirements.txt を参照してインストール
pip install -r docker/requirements.txt
```

### 静的解析（ローカル実行）

```bash
# ruff（スタイル・簡易自動修正）
ruff check .
# ruff を JSON で出力
ruff --format json . > reports/static-analysis/ruff.json

# pyright（型チェック）
pyright > reports/static-analysis/pyright.txt || true
```

### 自動修正（安全な変更のみ）

- 原則: 自動修正は新しいブランチで行い、影響を限定する。

```bash
# 新しいブランチを作成
git switch -c fix/ruff-imports-$(date +%Y%m%d)

# ruff の自動修正（インポート整理など安全な修正）
ruff --fix --select I001 .

# フォーマット
black --line-length 100 .

# 解析レポート保存
ruff --format json . > reports/static-analysis/ruff_after_fix_run.json
pyright > reports/static-analysis/pyright_after_fix_run.txt || true

# コミット・プッシュ
git add -A
git commit -m "fix(ruff): import organization and formatting"
git push -u origin HEAD
```

### 安全ファイル抽出（競合回避）

- 他ブランチとの重複変更を避けるため、まずは対象ファイルが他ブランチで変更されていないか確認します。

```bash
# ruff レポートからファイル一覧取得
jq -r '.[].filename' reports/static-analysis/ruff.json | sort -u > /tmp/ruff_files.txt

# 各ブランチが main と比較して変更しているファイル一覧を作成
git for-each-ref --format='%(refname:short)' refs/heads refs/remotes > /tmp/branches.txt
while read -r br; do
  git diff --name-only origin/main.."$br" | sort -u > "/tmp/files_${br//\//_}.txt"
done < /tmp/branches.txt

# 安全ファイル = どのブランチの変更リストにも現れないファイル
comm -23 /tmp/ruff_files.txt <(sort -m /tmp/files_* | sort -u) > /tmp/safe_files.txt
```

### コンフリクト検出と報告

- 重複しているファイルを一覧化して `reviews/CONFLICT_REPORT.md` を生成します。

```bash
# 既知の解析対象ファイルと全ブランチの差分を突き合わせてレポートを作る
# （詳細は scripts/ にあるサンプルスクリプトを参照）
```

## テスト実行

```bash
# 全テスト
pytest -q
# JUnit 形式で保存
pytest --junitxml=reports/test-results/junit.xml
```

- GPU に依存するテストは通常 CI から分離してください。代替として `-m "not gpu"` で除外できます。

## PR 作成とレポート添付

- 自動修正ブランチは PR を作成し、ボディに変更点のサマリと `reports/` へのリンクを含めること。
- PR のテンプレートは `reviews/` に用意されたテンプレートを利用してください。

## ワークツリー作成スクリプト

`./scripts/setup_worktree.sh` を使ってワークツリーを作成し、作業スコープを `WORKTREE_SCOPE.md` に記述してください。

```bash
# 例
scripts/setup_worktree.sh work/my-task-20260319
# そのワークツリーのルートに WORKTREE_SCOPE.md が作成されます
```

## CI の再現（ローカルでの一括実行）

```bash
# 一括で静的解析・型チェック・テストを実行してレポートを保存する例
ruff --format json . > reports/static-analysis/ruff.json || true
pyright > reports/static-analysis/pyright.txt || true
pytest --junitxml=reports/test-results/junit.xml || true
```

## ドキュメント更新ルール

- 変更がプロセスやツールに関する場合、`documents/` に追記してください。
- 操作手順の変更は `documents/COMMANDS.md` を更新し、`README.md` に参照を残すこと。

## タスク実行フロー — コマンドファースト方針

すべての修正タスクはまず「再利用可能なコマンド（スクリプト）を作成して動かす」ことを原則とします。これにより自動化、再現性、エージェントによる実行が容易になります。

推奨フロー:

1. 目的と影響範囲を特定し、`task.md` または `reviews/` に短い実行要件を記載する。
2. 小さなコマンド（`scripts/ci/` か `scripts/` 以下）を作成してローカルで実行可能にする。コマンドは引数で動作を切り替えられるようにする（例: `--dry-run`, `--commit`）。
3. コマンドを実行して出力（レポート・変更差分）を確認する。自動修正はまず `--dry-run`/非コミットで検証する。
4. コマンドの使い方（引数、出力場所、事前条件）を `scripts/ci/README.md` または `documents/COMMANDS.md` に追記する。
5. ドキュメント整備後、CI ワークフローやエージェントが呼び出せる形で公開する（`.github/workflows` から呼ぶ、または reusable workflow とする）。

ポリシー（必須）:

- すべての操作はまずローカル/ワークツリーで再現可能であること。
- 自動コミット・push はオプションとし、デフォルトでは `--dry-run` 相当で検証すること。
- コマンドは idempotent を目指す（何度実行しても破壊的にならない）。
- 新しいコマンドを作ったら、必ず `documents/COMMANDS.md` と `scripts/ci/README.md` に使用例を追記する。

この運用により、新しいエージェントはコマンドを直接呼び出して作業を再現・検証できるようになります。

## 推奨ワークフロー（詳細手順）

以下は `auto-fix` を行う場合の具体的手順です。worktree 運用規約に従い、まずは `work/*` ワークツリーや `origin/main` を同期・確認してください（参照: `documents/worktree-lifecycle.md`）。

前提: `origin/main` が最新であること、作業ワークツリーが clean であること。

1) ベースワークツリー準備

```bash
# main を最新化
git fetch origin
git switch main
git pull --ff-only origin main

# 新しい作業ブランチを作る（作業はワークツリーで）
scripts/setup_worktree.sh work/fix-ruff-YYYYMMDD
cd .worktrees/work/fix-ruff-YYYYMMDD
```

2) 解析レポート作成（読み取り専用）

```bash
# レポートを作る（ローカルでの再現）
scripts/ci/run_static_checks.sh reports/static-analysis

# 生成物例
# - reports/static-analysis/ruff.json
# - reports/static-analysis/pyright.txt
# - reports/test-results/junit.xml
```

3) 安全ファイル抽出（他ブランチとの競合回避）

```bash
# safe ファイル一覧を作る
python3 scripts/ci/safe_file_extractor.py
# 出力: /tmp/safe_files.txt, reports/static-analysis/safe_files.txt
```

4) 自動修正の dry-run（まず検証）

```bash
# dry-run: 実際にはコミットせずに変更差分を確認
cp /tmp/safe_files.txt /tmp/safe_files_dry.txt
while read -r f; do
  echo "ruff --fix $f (dry-run)"
  # 実行して差分を作るが commit しない
  ruff --fix "$f" || true
done < /tmp/safe_files_dry.txt

# フォーマット
black --line-length 100 $(sed 's@^/workspace/@@' /tmp/safe_files_dry.txt)

# レビュー: git diff を見て問題がないか確認する
git status --porcelain
git --no-pager diff

# ロールバック（dry-run の場合）
git restore --staged . || true
git restore . || true
```

5) 自動修正の適用（コミット）

```bash
# 安全と判断したら実行（--commit オプション）
scripts/ci/safe_fix.sh --commit

# これにより小さなコミットが作られ、ブランチに push される
```

6) PR 作成とレビュー

- PR ボディには `reviews/PRIORITIZED_FIXES.md` や `reports/static-analysis/*` へのリンクを含めること。
- `reviewer` が `pyright`/`pytest` を確認し、問題なければ `integrator` が統合する。

## 失敗時の扱い（要約）

- `ruff`/`pyright` のエラー: レポートを `reports/` に保存し、PR にログを添付して担当者に差し戻す。
- `pytest` の失敗: 自動マージ禁止。GPU 依存は `integration/gpu` に分離。
- 自動スクリプトが途中で失敗した場合は、`reports/logs/<job>-<ts>.log` を作成して `auditor` に通知する。

## コマンドの説明テンプレート（新規コマンド作成時に必ず記載）

- 命名: `scripts/<scope>/<name>.sh` または `scripts/<scope>/<name>.py`
- 概要: 何をするか一文で説明
- 引数: 主要なフラグと挙動（`--dry-run`, `--commit`, `--report-dir` 等）
- 入出力: 入力ファイル/期待する出力ファイルを明記
- 事前条件: 必要な前提（main 同期、worktree の clean 等）
- 失敗時の対応: 失敗したらどのログを確認するか、誰に報告するか

新しいコマンドを作ったら、このテンプレートに従って `scripts/` 側と `documents/COMMANDS.md` の双方に説明を追加してください。

---

作成日: 2026-03-19
