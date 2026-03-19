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

---

作成日: 2026-03-19
