# jax_util

> JAX 向けユーティリティ・線形代数・最適化ソルバー統合ライブラリ

______________________________________________________________________

## 🚀 クイックスタート（5分）

### 1. 環境構築

```bash
# 環境変数・ワークツリー自動設定
bash scripts/setup_worktree.sh my-feature-name "機能説明"

# ワークツリーに移動
cd .worktrees/my-feature-name
```

**⚠️ 注意**: ローカル仮想環境（`.venv`, `venv` など）の作成は **禁止** です。Docker 環境を使用してください。詳細は [documents/coding-conventions-project.md](./documents/coding-conventions-project.md#3-docker%E7%92%B0%E5%A2%83%E3%81%AE%E6%96%B9%E9%87%9D) を参照。

## 2. 最初のコミット

````bash
# スコープ確認・編集
vim WORKTREE_SCOPE.md
git add WORKTREE_SCOPE.md && git commit -m "chore(worktree): initialize scope"
```text

## 3. 実装テスト

```bash
# 単体テスト + 型チェック + リント統一実行
make ci
# または
bash scripts/ci/run_all_checks.sh
```yaml

📖 **詳細:** [QUICK_START.md](./QUICK_START.md) / [ワークツリー完全ガイド](./documents/worktree-lifecycle.md)

______________________________________________________________________

## 📚 ドキュメント（目的別逆引き）

### 👤 初心者向け

- **[セットアップ＆ワークフロー](./QUICK_START.md)** — 最初の 30 分
- **[ワークツリー規約](./documents/WORKTREE_SCOPE_TEMPLATE.md)** — スコープ定義テンプレート
- **[チェックリスト（8フロー）](./documents/FILE_CHECKLIST_OPERATIONS.md)** — 新規/テスト/ドキュメント各業務手順

### 👨‍💻 実装者向け

- **[コーディング規約](./documents/conventions/README.md)** — 言語別・共通規約
- **[Python 実装ガイド](./documents/coding-conventions-python.md)** — 型・関数・設計方針
- **[テスト規約](./documents/coding-conventions-testing.md)** — ユニット・統合テスト

### 🛠️ ツール・スクリプト

- **[ツール一覧](./documents/tools/README.md)** — 20+ スクリプト・使用方法
- **[ツール詳細リファレンス](./documents/tools/TOOLS_DIRECTORY.md)** — コマンド・オプション・フロー図

### 🔧 レビュー・運用者向け

- **[レビュー手順](./documents/REVIEW_PROCESS.md)** — 統合方針・検査項目
- **[チーム調整](./documents/AGENTS_COORDINATION.md)** — エージェント・ロール・手順
- **[常設エージェントチーム](./agents/README.md)** — 意図理解・調査・編集・独立レビュー・基盤管理を含む共通チーム構成
- **[タスク別 workflow 集](./agents/TASK_WORKFLOWS.md)** — 10 個の想定タスクと重複整理済み workflow family

### 🎓 設計・アーキテクチャ

- **[プロジェクト設計](./documents/design/jax_util/README.md)** — モジュール依存図
- **[ワークツリー構想図](./documents/conventions/README.md#%E5%9B%B3)** — 共通図表

### 📓 実験・ナレッジ

- **[実験メモ・結果](./notes/README.md)** — Smolyak・ベンチマーク・チューニング
- **[開発ログ](./diary/README.md)** — 日付別の試行・発見

### ❓ トラブルシューティング

- **[よくある問題](./documents/TROUBLESHOOTING.md)** — エラー＆解決策

______________________________________________________________________

## 🎯 よく使うコマンド

| 用途                      | コマンド                                         | 所要時間 |
| ------------------------- | ------------------------------------------------ | -------- |
| **CI 実行（全チェック）** | `make ci`                                        | 30～60秒 |
| **クイック CI**           | `make ci-quick`                                  | 10秒     |
| **Markdown 形式修正**     | `mdformat documents/`                            | 10秒     |
| **マークダウンリント**    | `python scripts/tools/check_markdown_lint.py .`  | 10秒     |
| **Markdown lint + fix**   | `python scripts/tools/fix_markdown_headers.py .` | 15秒     |
| **Dev 環境セットアップ**  | `make dev-setup`                                 | 1～2分   |
| **ツール一覧表示**        | `make tools-help`                                | 5秒      |

______________________________________________________________________

## 📖 ドキュメント構成（階層図）

```text
README.md (ここ) ← 👈 入口
  └─ 📚 documents/README.md
      ├─ 新規開発者向け
      │  ├─ conventions/ (規約体系)
      │  ├─ worktree-lifecycle.md
      │  └─ coding-conventions-testing.md
      ├─ 実装ガイド
      │  ├─ tools/README.md
      │  └─ FILE_CHECKLIST_OPERATIONS.md
      ├─ 設計・仕様
      ├─ 運用
      └─ よくある問題 (NEW)
  └─ 🛠️ scripts/README.md (ツール導入)
  └─ 📓 notes/README.md (実験ナレッジ)
  └─ 👥 .github/AGENTS.md (チーム運用)
  └─ 🤖 agents/README.md (常設チーム定義)
  └─ 🗂️ agents/TASK_WORKFLOWS.md (タスク workflow カタログ)
```text

**↑ 「📚」をクリック** → 各層の README.md で詳細ドキュメントへ

______________________________________________________________________

## 🔗 ワンクリックリンク集

### 急いでいる場合

- [5分で環境構築](./QUICK_START.md)
- [CI を実行するだけ](./scripts/ci/README.md)
- [よくある質問](./documents/TROUBLESHOOTING.md)

### ツール・スクリプト

- [整体テンプレート](./documents/WORKTREE_SCOPE_TEMPLATE.md)
- [マークダウン修正ツール](./documents/tools/README.md#markdown-%E3%83%84%E3%83%BC%E3%83%AB)
- [テスト・ログ保存](./scripts/run_pytest_with_logs.sh)

### 規約・ガイド

- [Python コーディング規約](./documents/coding-conventions-python.md)
- [テスト規約](./documents/coding-conventions-testing.md)
- [Markdown 記法統一](./documents/coding-conventions.md#markdown-%E6%9B%B8%E5%BC%8F%E4%BF%AE%E6%AD%A3%E3%83%AB%E3%83%BC%E3%83%AB)
- [常設エージェントチーム](./agents/README.md)
- [タスク別 workflow 集](./agents/TASK_WORKFLOWS.md)

### 実装開始の前に

1. [ワークツリー規約](./documents/WORKTREE_SCOPE_TEMPLATE.md) を読む
1. `.worktrees/*/WORKTREE_SCOPE.md` を編集
1. `make ci` でテスト実行

______________________________________________________________________

## 🔄 主要フロー（チェックリスト）

| フロー                  | ファイル                                                                                                               | 所要時間 |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------- | -------- |
| **1. 新規開発ブランチ** | [チェック1](./documents/FILE_CHECKLIST_OPERATIONS.md#%E3%83%81%E3%82%A7%E3%83%83%E3%82%AF%E3%83%AA%E3%82%B9%E3%83%881) | 20分     |
| **2. 実装テスト**       | [チェック3](./documents/FILE_CHECKLIST_OPERATIONS.md#%E3%83%81%E3%82%A7%E3%83%83%E3%82%AF%E3%83%AA%E3%82%B9%E3%83%883) | 30分     |
| **3. ドキュメント更新** | [チェック4](./documents/FILE_CHECKLIST_OPERATIONS.md#%E3%83%81%E3%82%A7%E3%83%83%E3%82%AF%E3%83%AA%E3%82%B9%E3%83%884) | 15分     |
| **4. 統合（main へ）**  | [チェック∞](./documents/FILE_CHECKLIST_OPERATIONS.md)                                                                  | 10分     |

詳細は [FILE_CHECKLIST_OPERATIONS.md](./documents/FILE_CHECKLIST_OPERATIONS.md) を参照

______________________________________________________________________

## 💡 開発フロー（図解）

```text
git clone → setup_worktree → WORKTREE_SCOPE 編集
   ↓
実装＆テスト (make ci) → コミット
   ↓
ドキュメント更新 (mdformat) → コミット
   ↓
統合ブランチへ merge → GitHub Actions
```yaml

______________________________________________________________________

## 📊 プロジェクト情報

| 項目             | 値                       |
| ---------------- | ------------------------ |
| **言語**         | Python 3.9+              |
| **主要依存**     | JAX, Equinox, scipy      |
| **テスト**       | pytest, pyright, ruff    |
| **マークダウン** | mdformat, markdownlint   |
| **ドキュメント** | 119 ファイル（構造化中） |

______________________________________________________________________

## 🆘 困ったら

1. **[よくある質問](./documents/TROUBLESHOOTING.md)** を先に確認
1. **[ツール一覧](./documents/tools/README.md)** で解決ツールを探す
1. **[ワークフロー手順](./documents/FILE_CHECKLIST_OPERATIONS.md)** で対応フローを確認
1. **[規約](./documents/conventions/README.md)** で実装・テスト方法を確認

______________________________________________________________________

## 📞 その他

- **Development:** `pip install -e ".[dev]"` → `pytest` → `ruff check .`
- **HLO Analysis:** `python scripts/hlo/summarize_hlo_jsonl.py <file.jsonl>`
- **Notes:** [notes/README.md](./notes/README.md) 参照
````
