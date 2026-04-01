# 📚 ドキュメント ハブ

> jax_util プロジェクトの全階層・全分野向けドキュメント入口

`documents/` には正本だけを残します。履歴用の proposal / report / summary / `.bak` は置かず、過去状態は Git 履歴で追います。

______________________________________________________________________

## 🎯 目的別ナビゲーション

### 👤 **初心者向け（最初の 1 時間で読むべき）**

| 項目                 | 場所                                             | 読む時間 | 目的                   |
| -------------------- | ------------------------------------------------ | -------- | ---------------------- |
| **環境セットアップ** | [セットアップガイド](./worktree-lifecycle.md)    | 20分     | ワークツリー・環境作成 |
| **最初の実装**       | [コーディング規約](./conventions/README.md)      | 15分     | Python 実装ルール      |
| **作業フロー**       | [チェックリスト](./FILE_CHECKLIST_OPERATIONS.md) | 15分     | 8 種類の作業手順       |
| **テスト方法**       | [テスト規約](./coding-conventions-testing.md)    | 10分     | ユニット・統合テスト   |

**👉 推奨順序：** セットアップ → コーディング規約 → チェックリスト → テスト

______________________________________________________________________

### 👨‍💻 **実装者向け（プロジェクトの全体像）**

#### 規約・ガイドライン

- **[コーディング規約 (全言語)](./conventions/README.md)** — Python・C++ 共通 + 言語別ガイド
- **[Python 実装ガイド](./coding-conventions-python.md)** — 型注釈・関数・モジュール構成
- **[テスト規約](./coding-conventions-testing.md)** — ユニット・統合・エッジケーステスト
- **[Markdown 記法](./coding-conventions.md)** — ドキュメント統一・図解方法

#### 設計・アーキテクチャ

- **[プロジェクト設計](./design/README.md)** — モジュール依存・API 構造
<!-- コンポーネント詳細ドキュメントは Week 2 で作成予定 -->
<!-- - **[型システム・基本型](./type-aliases.md)** — 型エイリアス・Protocol 定義 -->
<!-- - **[基本コンポーネント](./design/base_components.md)** — LinOp・linearize・adjoint -->

#### ツール・スクリプト

- **[ツール一覧](./tools/README.md)** — Markdown・テスト・解析ツール
- **[ツール詳細リファレンス](./tools/TOOLS_DIRECTORY.md)** — コマンド・オプション・フロー

______________________________________________________________________

### 🔧 **レビュー・運用向け**

- **[レビュー手順](./REVIEW_PROCESS.md)** — PR フロー・チェックリスト・承認基準
  - 🆕 **レビュー履歴管理** — 最新のみ保持、古いレビューは git 履歴に移動
  - コマンド例: `git rm reviews/*.md && git commit`
  - 復元方法: `git show HEAD~n:reviews/FILENAME.md`
- **[チーム調整](./AGENTS_COORDINATION.md)** — ロール・責任・コミュニケーション
- **[タスク別 workflow 集](../agents/TASK_WORKFLOWS.md)** — 10 個の想定タスクと整理済み workflow family
- **[ワークツリー管理](./WORKTREE_SCOPE_TEMPLATE.md)** — スコープ・ブランチ管理

______________________________________________________________________

### 🎓 **設計詳細（深掘り参照）**

#### API・ソルバー設計

- **[Solvers 設計](./design/jax_util/README.md)** — シンプル KKT・PDIPM・LOBPCG
- **[補助設計文書](./design/)** — 実験・拡張提案

#### 実験・試行段階

- **[実験の標準手順](./experiment-workflow.md)** — 準備・実装・静的チェック・実行・結果レポートを 1 ページに統合した入口
- **[実験の批判的レビュー手順](./experiment-critical-review.md)** — 数学的妥当性、文献接続、図表、evidence sufficiency を点検する review 正本
- **[研究・実験改造 workflow](./research-workflow.md)** — 数式・比較対象・逐次改造・branch 反映の正本
- **[実験レポートの書き方](./experiment-report-style.md)** — IMRaD を repo 向けに寄せた report 構成と figure / table / abstract のルール
- **[実験環境](./coding-conventions-experiments.md)** — HLO 採取・ベンチマーク
- **[実験手順 reference 索引](../references/experiment_workflow/README.md)** — 実験手順、再現性、図表、benchmark、批判的 review の参照文献
- **[生成 AI reference 索引](../references/generative_ai/README.md)** — multi-agent workflow、LLM review 支援、文献レビュー自動化の参照文献
- **[Benchmark 方針](./conventions/python/20_benchmark_policy.md)** — benchmark と experiment の境界
- **[実験ディレクトリ構成](./conventions/python/30_experiment_directory_structure.md)** — `experiments/` と `experiment_runner` の責務分担
- **[レビュー文書](./coding-conventions-reviews.md)** — AI レビュー・解析結果

______________________________________________________________________

### 🤖 **AI エージェントワークフロー ガイド**

> Claude Code vs GitHub Copilot vs Cursor vs GPT Codex — ツール選択・活用方法・カスタマイズ機能

- **[AI エージェント ワークフロー比較](./AI_AGENT_WORKFLOWS_COMPARISON.md)** — 各エージェントの特徴・ワークフロー・推奨用途
  - Claude Code（Cursor での活用）
  - GitHub Copilot Pro/Pro+ エージェント  
  - OpenAI Codex（GitHub 統合）
  - ハイブリッド構成のベストプラクティス
  - **Skills 以外のカスタマイズ機能** — 7 つの拡張レイヤー解説

- **[AI エージェント カスタマイズ リサーチ 2026](./AI_AGENT_CUSTOMIZATION_RESEARCH_2026.md)** — 最新機能・実装例・統合テンプレート
  - **Custom Instructions** — `.github/copilot-instructions.md`、`.claude/instructions/`
  - **Prompt Files** — 再利用テンプレート・{{variable}} 対応
  - **MCP Servers** — PostgreSQL/MongoDB/REST/Git 連携
  - **Hooks & Lifecycle** — pre-gen, post-gen, validation 等
  - **Custom Agents** — Code Reviewer, Experiment Executor など
  - **Environment & Secrets Management** — credential 安全管理
  - **Subagents & Task Distribution** — Sequential/Parallel/Branching ワークフロー
  - **CLI エージェント・コマンドラインリファレンス** — Claude CLI / Copilot CLI / Cursor CLI / GitHub CLI

---

### 🛠️ **Skills ライブラリ — プロジェクト標準ワークフロー**

> 既存ワークフロー（実験・コードレビュー・テスト）を Skill 化し、エージェントと連携

- **[Skills 統合計画](../.github/SKILLS_INTEGRATION_PLAN.md)** — 全体 Skill 体系・実装ロードマップ
  - Skill 1: `static-check` — 型・テスト・Docker 検証
  - Skill 2: `code-review` — コード品質多層検証（A/B/C層）
  - Skill 3: `run-experiment` — 実験実行・結果生成（5段階）
  - Skill 4: `critical-review` — 実験学術的妥当性検証（Code/Results/Math）
  - Skill 5: `research-workflow` — 長期研究・反復管理

- **[Skills ライブラリ](../.github/skills/README.md)** — 全 5 Skill の使用ガイド・統合マップ
  - 各 Skill の README + Template + Script
  - CLI / GitHub Actions での実行方法
  - 実装状況・次のステップ

______________________________________________________________________

### 🤖 **エージェントタスク完全ガイド（NEW）**

> 50 通りのタスク・5 つの実装パス・完全実行可能性検証

#### 📖 ドキュメント群

| ドキュメント                                             | 対象     | 内容                           | 読む時間 |
| -------------------------------------------------------- | -------- | ------------------------------ | -------- |
| **[📋 索引ガイド](./AGENT_TASK_INDEX_GUIDE.md)**         | 全員     | 3 つドキュメントの役割・使い方 | 15 分    |
| **[🎯 タスクマップ](./AGENT_TASK_MAP.md)**               | 全員     | 50 個のタスク詳細 + 5 Path ◎   | 50-60 分 |
| **[✅ 検証レポート](./AGENT_TASK_VALIDATION_REPORT.md)** | QA・管理 | リソース・動線・テスト結果 ✓   | 40-50 分 |
| **[📝 準備チェックシート](./AGENT_TASK_CHECKLIST.md)**   | 管理者   | 8 Phase + ロールアウト計画     | 30-40 分 |

#### 🎯 Path 別ガイド

- **[Path A: 初心者オンボーディング](./AGENT_TASK_MAP.md#%F0%9F%93%A6-%E5%88%9D%E5%BF%83%E8%80%85%E5%90%91%E3%81%91task-1-10)** — Task 1-10 (1.5-2h)
- **[Path B: 実装者フロー](./AGENT_TASK_MAP.md#%F0%9F%92%BB-%E5%AE%9F%E8%A3%85%E8%80%85%E5%90%91%E3%81%91task-11-20)** — Task 11-20 (2-3h)
- **[Path C: アーキテクチャ・レビュー](./AGENT_TASK_MAP.md#%F0%9F%8E%93-%E3%83%86%E3%83%83%E3%82%AF%E3%83%AA%E3%83%BC%E3%83%89%E5%90%91%E3%81%91task-21-30)** — Task 21-30 (2.5-3.5h)
- **[Path D: 運用・管理](./AGENT_TASK_MAP.md#%F0%9F%8F%A2-%E3%83%97%E3%83%AD%E3%82%B8%E3%82%A7%E3%82%AF%E3%83%88%E7%AE%A1%E7%90%86%E5%90%91%E3%81%91task-31-40)** — Task 31-40 (2-3h)
- **[Path E: 自動化 AI](./AGENT_TASK_MAP.md#%F0%9F%A4%96-ai%E3%82%A8%E3%83%BC%E3%82%B8%E3%82%A7%E3%83%B3%E3%83%88%E5%90%91%E3%81%91task-41-50)** — Task 41-50 (2.5-3h)

#### 📊 統計

- **50 個タスク** × 5 ユーザー層 = **すべての役割をカバー** ✓
- **47 ドキュメント参考** (100% 検証済み) ✓
- **29 スクリプト・ツール** (90% 実装済み、10% 将来作成) ✓
- **推定学習時間**: 25-30 時間（全 Path 完了）

#### 🚀 クイックスタート

1. **[📋 索引ガイド](./AGENT_TASK_INDEX_GUIDE.md)** を 15 分で読む
1. 自分の Path を [🎯 タスクマップ](./AGENT_TASK_MAP.md) から選択
1. **Task 1** から開始
1. 疑問があれば [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) または [✅ 検証レポート](./AGENT_TASK_VALIDATION_REPORT.md) 参照

______________________________________________________________________

### ❓ **困ったとき**

- **[よくある質問](./TROUBLESHOOTING.md)** — エラー・解決策・ベストプラクティス
- **[ツール一覧](./tools/README.md)** → 対応ツールで解決
- **[チェックリスト](./FILE_CHECKLIST_OPERATIONS.md)** → 作業フロー手順確認
- **[エージェントタスク検証](./AGENT_TASK_VALIDATION_REPORT.md#%F0%9F%8E%AF-%E3%83%AA%E3%82%BD%E3%83%BC%E3%82%B9%E4%B8%8D%E8%B6%B3%E3%82%A8%E3%83%AA%E3%82%A2)** → 実装可能性確認

______________________________________________________________________

## 📑 **ドキュメント 階層構成**

```text
documents/ (ここ) ← 📍 ハブ

 📋 規約・方針（最初に読む）
  ├─ conventions/README.md ← コーディング規約の主入口
  ├─ conventions/python/ (Python ガイド詳細)
  ├─ coding-conventions.md (共通ルール)
  ├─ coding-conventions-testing.md (テスト)
	  ├─ coding-conventions-python.md (Python 詳細)
	  ├─ coding-conventions-experiments.md (実験)
	  ├─ experiment-workflow.md (実験の標準手順)
	  ├─ experiment-critical-review.md (実験の批判的レビュー)
	  └─ research-workflow.md (研究設計・比較・改造)
	  └─ experiment-report-style.md (実験レポート体裁)

 🏛️ 設計・アーキテクチャ
  ├─ design/README.md (設計 入口)
  ├─ design/base_components.md (型・Protocol)
  ├─ design/jax_util/README.md (ソルバー)
  └─ type-aliases.md (型定義)

 🛠️ ツール・スクリプト
  ├─ tools/README.md (ツール概要)
  ├─ tools/TOOLS_DIRECTORY.md (詳細リファレンス)
  └─ tools/check_markdown_lint.py (ツール実行体)

 📊 運用・管理
  ├─ FILE_CHECKLIST_OPERATIONS.md (作業チェックリスト)
  ├─ REVIEW_PROCESS.md (レビュー手順)
  ├─ AGENTS_COORDINATION.md (チーム・ロール)
  ├─ WORKTREE_SCOPE_TEMPLATE.md (スコープ テンプレート)
  └─ worktree-lifecycle.md (ワークツリー ライフサイクル)

 ❓ ヘルプ・参考
   ├─ TROUBLESHOOTING.md ← エラー解決
   └─ reviews/ (AI レビュー・解析)
```

______________________________________________________________________

## 🚀 **クイックアクセス**

### よくある作業

| 作業              | 参照先                                                                                                             | 時間 |
| ----------------- | ------------------------------------------------------------------------------------------------------------------ | ---- |
| 新規ブランチ作成  | [チェックリスト1](./FILE_CHECKLIST_OPERATIONS.md#%E3%83%81%E3%82%A7%E3%83%83%E3%82%AF%E3%83%AA%E3%82%B9%E3%83%881) | 20分 |
| Python コード実装 | [Python 規約](./coding-conventions-python.md)                                                                      | 15分 |
| テスト作成        | [テスト規約](./coding-conventions-testing.md)                                                                      | 10分 |
| ドキュメント追加  | [MD 記法](./coding-conventions.md#markdown-%E6%9B%B8%E5%BC%8F%E4%BF%AE%E6%AD%A3%E3%83%AB%E3%83%BC%E3%83%AB)        | 10分 |
| レビュー実施      | [レビュー手順](./REVIEW_PROCESS.md)                                                                                | 30分 |
| Markdown リント   | [ツール](./tools/README.md#markdown-%E3%83%84%E3%83%BC%E3%83%AB)                                                   | 5分  |

______________________________________________________________________

## 📝 **追記・更新ルール**

### 規約を追加する場合

1. **汎用 / 共通ルール** → `conventions/common/` に追記
1. **Python 固有** → `conventions/python/` に追記
1. **テスト・実験・レビュー** → `coding-conventions-*.md` に追記
1. **実験の標準手順・反復改造フロー** → `experiment-workflow.md` に追記
1. **実験の批判的レビュー観点** → `experiment-critical-review.md` に追記
1. **研究設計・比較実験・改造手順** → `research-workflow.md` に追記
1. **実験レポートの構成・体裁** → `experiment-report-style.md` に追記

### 設計仕様を追加する場合

1. **基本型・Protocol** → `design/base_components.md` 更新
1. **ソルバー・API** → `design/jax_util/*.md` に新規追記
1. **実験段階** → `design/` 直下に独立ファイル作成

### 文書管理の指針

- ✅ テーマごと独立ファイル化 → 参照の連鎖を避ける
- ✅ 各ファイルは「それ単体で読める」状態維持
- ✅ リンク集（このファイル）で逆引きできる
- ✅ Markdown は `mdformat`・`markdownlint` 統一

______________________________________________________________________

## 🔗 **外部リンク**

- **最初に読む：** [root/README.md](../README.md) → クイックスタート
- **コマンド参考：** [QUICK_START.md](../QUICK_START.md)
- **プロジェクト情報：** [pyproject.toml](../pyproject.toml)

______________________________________________________________________

## 📋 **現在の構成（トリプル層）**

### 1. 一般的なコーディング方針

- `coding-conventions.md` （共通・Markdown）
- `coding-conventions-python.md` （Python 詳細）
- `coding-conventions-testing.md` （テスト）
- `coding-conventions-experiments.md` （実験）
- `coding-conventions-reviews.md` （レビュー）
- `conventions/` （言語別ガイド）
- **対象:** 命名・型注釈・JAX 運用・テスト・ログなど共通ルール

### 2. ベースコンポーネント設計

- `design/base_components.md`
- `type-aliases.md`
- **対象:** `LinOp`・Protocol・型エイリアス

### 3. 安定 API 詳細設計

- `design/` （ソルバー・最適化・HLO）
- **対象:** サブモジュール単位の API 仕様
