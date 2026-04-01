# jax_util

> JAX 向けユーティリティ・線形代数・最適化ソルバー統合ライブラリ

**あなたは誰ですか？** → 下のボタンをクリック

______________________________________________________________________

## 🎯 あなたの役割で選ぶ（3回のクリックで到達可能）

### 📚 **ドキュメント・ガイドが必要**

👉 [documents/README.md](./documents/README.md)

- 初心者セットアップ（環境・テスト・規約）
- 実装ガイド（Python・テスト・デザイン）
- レビュー・運用手順
- よくある質問

### 🛠️ **ツール・スクリプトを使いたい**

👉 [scripts/README.md](./scripts/README.md)

- ツール一覧（20+ スクリプト）
- 使用方法・オプション
- CI/CD コマンド
- トラブルシューティング

### 📓 **実験・ナレッジ・メモを見たい**

👉 [notes/README.md](./notes/README.md)

- 実験レポート・結果
- 試行錯誤のメモ
- チューニング記録
- 開発ログ

### 🤖 **エージェント（AI）として利用**

👉 [agents/README.md](./agents/README.md) + [agents/TASK_WORKFLOWS.md](./agents/TASK_WORKFLOWS.md)

- チームロール・権限
- タスク別 workflow
- 通信規約・handoff
- 50 個のタスク完全ガイド（📖 [AGENT_TASK_MAP.md](./documents/AGENT_TASK_MAP.md)）

### 🎓 **設計・アーキテクチャを深掘り**

👉 [documents/design/README.md](./documents/design/README.md)

- ソルバー・最適化アルゴリズム
- 型システム・API 設計
- モジュール依存関係

______________________________________________________________________

## ⚡ クイックコマンド（よく使うやつ）

```bash
# 環境セットアップ（1回目）
bash scripts/setup_worktree.sh work/my-feature-YYYYMMDD
cd .worktrees/work-my-feature-YYYYMMDD

# テスト実行（毎回）
make ci                         # 全テスト（30～60秒）
make ci-quick                   # 軽量テスト（10秒）

# ドキュメント修正
mdformat documents/             # Markdown 書式修正

# ツール実行例
python scripts/check_convention_consistency.py  # 規約検証
python scripts/docker_dependency_validator.py   # Docker 依存確認
```

⚠️ **注意**: ローカル仮想環境（`.venv`, `venv`）は禁止。Docker を使ってください。

______________________________________________________________________

## 📋 ディレクトリ構成（サイト マップ）

```
/workspace
├─ README.md ← 👈 ここ（入口）
│
├─ 📚 documents/
│   ├─ README.md ← 【第2層】全ドキュメント・ガイド入口
│   ├─ conventions/ ← 規約（Python・C++・共通）
│   ├─ design/ ← アーキテクチャ・API・設計
│   ├─ tools/ ← ツール・スクリプト詳細
│   ├─ QUICK_START.md
│   ├─ REVIEW_PROCESS.md
│   └─ worktree-lifecycle.md
│
├─ 🛠️ scripts/
│   ├─ README.md ← 【第2層】ツール・スクリプト入口
│   ├─ agent_tools/ ← エージェント向け helper
│   ├─ ci/ ← CI/CD スクリプト
│   ├─ tools/ ← 検証・チェック ツール
│   └─ *.sh ← 補助スクリプト
│
├─ 📓 notes/
│   ├─ README.md ← 【第2層】実験・ナレッジ入口
│   ├─ experiments/ ← 実験レポート
│   ├─ worktrees/ ← worktree 削除予定メモ
│   └─ knowledge/ ← 試行錯誤の記録
│
├─ 🤖 agents/
│   ├─ README.md ← 【第2層】エージェント チーム入口
│   ├─ agents_config.json ← チーム正本（ロール・権限）
│   ├─ COMMUNICATION_PROTOCOL.md
│   ├─ TASK_WORKFLOWS.md ← タスク workflow 集
│   ├─ task_catalog.yaml
│   └─ templates/ ← エージェント出力テンプレート
│
├─ 📖 python/
│   ├─ jax_util/ ← メインライブラリ
│   ├─ experiment_runner/ ← 実験実行基盤
│   └─ tests/ ← テストスイート
│
├─ 🧪 experiments/
│   ├─ functional/ ← 機能テスト・検証
│   └─ smolyak_experiment/ ← Smolyak グリッド実験
│
└─ 📝 その他
    ├─ docker/ ← Docker 環境・依存
    ├─ reviews/ ← レビュー報告・進捗レポート
    ├─ diary/ ← 日付別開発ログ
    ├─ Makefile ← タスク自動化
    └─ task.md ← TO-DO（Phase 別）
```

**👉 3 階層ナビゲーション例:**
- 第 1 層: 👆 上の「あなたの役割で選ぶ」を選択
- 第 2 層: documents/README.md や scripts/README.md で詳細選択
- 第 3 層: 実際のドキュメント・ガイドを参照

______________________________________________________________________

## 📊 プロジェクト情報

| 項目             | 値                                 |
| ---------------- | ---------------------------------- |
| **言語**         | Python 3.10+, C++17           |
| **主要依存**     | JAX, Equinox, scipy, numpy    |
| **テスト**       | pytest, pyright, ruff         |
| **ドキュメント** | Markdown (gdformat)           |
| **自動化**       | Makefile, GitHub Actions, CI  |
| **ドキュメント数** | 130+ ファイル（構造化）        |
| **スクリプト**   | 20+ （解析・検証・自動化）     |

______________________________________________________________________

## 🆘 困ったら

**該当する状況を選んでクリック:**

| 状況 | 参照先 |
|------|--------|
| **セットアップできない** | [QUICK_START.md](./QUICK_START.md) / [worktree-lifecycle.md](./documents/worktree-lifecycle.md) |
| **テストが失敗する** | [documents/TROUBLESHOOTING.md](./documents/TROUBLESHOOTING.md) / [coding-conventions-testing.md](./documents/coding-conventions-testing.md) |
| **型エラーが出ている** | [coding-conventions-python.md](./documents/coding-conventions-python.md#型チェッカの活用) |
| **ドキュメント修正方法** | [`mdformat` の使い方](./documents/coding-conventions.md#markdown-書式修正ルール) |
| **エージェント向けガイド** | [agents/USER_GUIDE_JA.md](./agents/USER_GUIDE_JA.md) 🇯🇵 |
| **ツールが見つからない** | [scripts/README.md](./scripts/README.md) / [documents/tools/README.md](./documents/tools/README.md) |
| **Worktree・ブランチ管理** | [documents/worktree-lifecycle.md](./documents/worktree-lifecycle.md) / [documents/BRANCH_SCOPE.md](./documents/BRANCH_SCOPE.md) |
| **レビュー・統合手順** | [documents/REVIEW_PROCESS.md](./documents/REVIEW_PROCESS.md) / [agents/COMMUNICATION_PROTOCOL.md](./agents/COMMUNICATION_PROTOCOL.md) |

______________________________________________________________________

## 🎬 次のステップ

### **初心者向け**
1. [QUICK_START.md](./QUICK_START.md) を 5 分で読む
2. `bash scripts/setup_worktree.sh work/my-feature-YYYYMMDD` を実行
3. `make ci` でテストが通るか確認
4. [documents/coding-conventions-python.md](./documents/coding-conventions-python.md) を読む

### **実装者向け**
1. 実装対象を [task.md](./task.md) から選ぶ
2. `documents/conventions/` で規約確認
3. [scripts/ci/](./scripts/ci/) で CI 実行方法確認
4. ワークツリーで実装 → テスト → コミット

### **レビュー・運用者向け**
1. [documents/REVIEW_PROCESS.md](./documents/REVIEW_PROCESS.md) を読む
2. [agents/README.md](./agents/README.md) でチームロール確認
3. [agents/COMMUNICATION_PROTOCOL.md](./agents/COMMUNICATION_PROTOCOL.md) で通信規約確認

### **エージェント（AI）向け**
→ **[agents/USER_GUIDE_JA.md](./agents/USER_GUIDE_JA.md)** 🇯🇵 から始めてください

______________________________________________________________________

**最後に更新：** 2026-04-01  
**スポンサー:** jax_util development team  
**言語:** 日本語 • [English](./README_EN.md) (未翻訳)

1. **[ツール一覧](./documents/tools/README.md)** で解決ツールを探す
1. **[ワークフロー手順](./documents/FILE_CHECKLIST_OPERATIONS.md)** で対応フローを確認
1. **[規約](./documents/conventions/README.md)** で実装・テスト方法を確認

______________________________________________________________________

## 📞 その他

- **Development:** `pip install -e ".[dev]"` → `pytest` → `ruff check .`
- **HLO Analysis:** `python scripts/hlo/summarize_hlo_jsonl.py <file.jsonl>`
- **Notes:** [notes/README.md](./notes/README.md) 参照
