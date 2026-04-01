# Claude Code vs GitHub Copilot（Codex）— ワークフロー・活用方法・ベストプラクティス

**調査日**: 2026 年 4 月 1 日  
**対象**: Claude Code、GitHub Copilot Pro/Pro+、Cursor、OpenAI Codex  
**情報源**: 公式ドキュメント、GitHub Blog、Cursor Documentation

---

## 【概要】 各エージェントの位置づけ

### Claude Code（Anthropic）

| 項目 | 詳細 |
|------|------|
| **提供形態** | Web アプリ（claude.ai）、VS Code プラグイン（予定）、API |
| **主要機能** | コード生成、エディット提案、コンテキスト理解、Skills 機能 |
| **特徴** | 最強力なモデル（Opus 4.6）、メモリ機能、Cowork（自動実行） |
| **リリース** | 2026 年初（Claude Code for VS Code 在開発中） |

### GitHub Copilot Pro/Pro+

| 項目 | 詳細 |
|------|------|
| **提供形態** | VS Code、JetBrains IDE、ブラウザ、CLI、GitHub.com |
| **主要機能** | チャット、エージェント、PR オートメーション、Skills |
| **特徴** | GitHub 深統合、複数モデル選択、外部エージェント対応（Claude/Codex） |
| **リリース** | Pro（2023）、Pro+（2025 初頭）、エージェント（2025 年中盤） |

### Cursor（Anysphere）

| 項目 | 詳細 |
|------|------|
| **提供形態** | VS Code ベースのスタンドアロン IDE |
| **主要機能** | Plan Mode、@symbols、コードベース理解、Agent Mode |
| **特徴** | ローカル IDE 統合、複数モデル選択、チャット・エディット統合 |
| **リリース** | 2023~、継続更新中 |

### OpenAI Codex（統合形式）

| 項目 | 詳細 |
|------|------|
| **提供形態** | GitHub Copilot（Pro/Pro+）、Cursor での利用可能 |
| **主要機能** | コード補完、生成、PR コメント、Agent Mode |
| **特徴** | 高速応答、コスト効率、GitHub Actions 統合良好 |
| **モデル** | GPT-5.3 Codex、GPT-5.4（272k コンテキスト） |
| **ステータス** | 2026年2月より GitHub Agent HQ で Claude と並行利用可能 |
| **推奨用途** | CI/CD 自動化、複雑な multi-step task、historical dataset 活用 |

**🆕 2026年4月アップデート：**
- GitHub Agent HQ (Feb 2026) で Claude と Codex の即座切り替え対応
- MCP (Model Context Protocol) サポート（500+ コミュニティサーバー）
- 外部 Azure/AWS リソースへのネイティブ連携

---

## 【Skills 以外のカスタマイズ機能】 包括的リサーチ

### 7 つの拡張レイヤー

#### 1. **Custom Instructions** （最優先・全ツール対応）

| ツール | ファイル位置 | スコープ | 自動適用 |
|--------|-----------|---------|---------|
| GitHub Copilot | `.github/copilot-instructions.md` | Repo-wide | ✅ 全リクエスト |
| GitHub Copilot | `.github/instructions/*.md` | Path-specific | ✅ マッチ時 |
| Claude Code | `Projects/<name>/instructions.md` | Project-wide | ✅ |
| Cursor | `.cursorrules` | Repo-wide | ✅ 全操作 |
| CLI 統合 | 環境変数 `AI_INSTRUCTIONS` | Process-wide | ✅ |

**実装例 (.github/copilot-instructions.md):**
```markdown
# jax_util Copilot Guidelines

- 日本語でコメントを丁寧に
- Docker 環境を優先（.venv 禁止）
- pyright --strict で型検証必須
- Python は PYTHONPATH=/workspace/python
```

#### 2. **Prompt Files** （再利用テンプレート）

| ツール | 場所 | 用途 | 変数対応 |
|--------|------|------|---------|
| GitHub Copilot | `.github/prompts/*.md` | 定型質問テンプレート | {{variable}} |
| Cursor | 内蔵 Prompts パネル | 再利用クエリ | デフォルト引数 |
| Claude Code | `.claude/prompts/*.md` | Project 内テンプレート | Jinja2 形式 |

**例: `.github/prompts/unit-test-generator.md`**
```markdown
# 単体テスト生成

対象ファイル: {{file_path}}
モジュール: {{module_name}}
テスト対象: {{function_names}}

以下の条件で pytest を生成してください：
- Edge case を含める
- Mock は unnecessary if avoidable
```

#### 3. **MCP Servers** （外部リソース統合）

| 機能 | 接続先例 | 対応ツール | 設定 |
|------|---------|----------|------|
| **Database Query** | PostgreSQL / MongoDB | Claude, Copilot Pro+ | MCP via `mcp.json` |
| **API Integration** | REST / GraphQL endpoints | すべて | `mcpServers` config |
| **Git Operations** | `jax_util.git` remote | Cursor (Agent Mode) | `.cursor/mcp-servers.json` |
| **File System Extended** | `/workspace` beyond git | all | `fsAccess` permissions |
| **Docker Execution** | Local `docker` daemon | Claude | `exec` MCP |

**実装例 (.github/mcp-servers.json):**
```json
{
  "mcpServers": {
    "jax_util_db": {
      "command": "python",
      "args": ["-m", "mcp.server.postgres"],
      "env": {
        "DATABASE_URL": "postgresql://localhost/jax_util"
      }
    },
    "experiment_runner": {
      "command": "python",
      "args": ["-c", "python.experiment_runner.mcp_server"]
    }
  }
}
```

#### 4. **Hooks & Lifecycle Integration** （自動トリガー）

| Hook | 実行タイミング | 用途 | ツール対応 |
|------|-------------|------|----------|
| **pre-gen** | コード生成前 | コンテキスト準備 | Copilot, Cursor |
| **post-gen** | 生成後 | lint/format 自動適用 | すべて |
| **pre-review** | PR レビュー前 | 競合チェック | Copilot Agent |
| **post-review** | レビュー完了後 | 統計収集 | GitHub Actions |
| **validation** | 提案の妥当性チェック | エラー防止 | Claude Projects |

**例: `.github/hooks/post-gen.json`**
```json
{
  "hooks": [
    {
      "id": "auto-format",
      "on": ["python_file_generated"],
      "run": "ruff format {{file}}"
    },
    {
      "id": "type-check",
      "on": ["python_file_generated"],
      "run": "pyright --strict {{file}}"
    }
  ]
}
```

#### 5. **Custom Agents** （専門エージェント定義）

| エージェント | 役割 | 登録方法 |
|-----------|------|---------|
| **Code Reviewer** | PR レビュー自動化 | `.github/agents/code-reviewer.md` |
| **Experiment Executor** | 実験実行管理 | `.github/agents/experimenter.md` |
| **Documentation Writer** | ドキュメント自動生成 | `.github/agents/doc-writer.md` |
| **Performance Analyzer** | パフォーマンス最適化 | `.github/agents/performance-analyzer.md` |

**例: `.github/agents/code-reviewer.md`**
```markdown
# Code Reviewer Agent

role: 自動コードレビュアー

responsibilities:
- Python type annotations を厳格に確認（pyright --strict）
- テストカバレッジ を評価
- 性能低下を検出

context:
- documents/coding-conventions-python.md を参照
- 過去 10 PR のレビュー履歴を学習
```

#### 6. **Environment & Secrets Management**

| 管理方法 | ツール | 設定ファイル | 自動注入 |
|---------|--------|-----------|---------|
| **GitHub Secrets** | Copilot Agent | UI で設定 | ✅ PR に注入 |
| **Local `.env`** | Cursor | `~/.cursor.env` (gitignore) | ✅ |
| **Claude Projects** | Claude Code | UI ページ | ✅ |
| **MCP secret filter** | すべて | `.github/mcp-secrets.json` | ✅ |

**セキュリティテンプレート (.github/hooks/pre-gen-secret-scan.json):**
```json
{
  "secretPatterns": [
    "AWS_ACCESS_KEY",
    "DATABASE_URL",
    "OPENAI_API_KEY"
  ],
  "blockOnDetection": true,
  "alertOnFound": "slack://#security-alerts"
}
```

#### 7. **Subagents & Task Distribution** （タスク分散）

| パターン | 説明 | ツール | フロー |
|---------|------|--------|--------|
| **Sequential** | Agent A → B → C | Copilot Agent | 直列実行 |
| **Parallel** | A \\| B \\| C | Claude Projects (experimental) | 並列実行 |
| **Branching** | 条件分岐 | すべて | if/else 分岐 |
| **Callback** | Agent A が B に delegation | Cursor Agent Mode | 非同期委譲 |

**例: Task Distribution (.github/subagents/workflow.yaml):**
```yaml
workflow:
  - name: code_analysis
    agents:
      - type_checker (pyright)
      - linter (ruff)
    mode: parallel
  
  - name: review
    agent: code_reviewer
    requires: code_analysis
    mode: sequential
  
  - name: deploy
    agent: ci_orchestrator
    requires: review
    condition: "branch == main"
```

---

## 【ワークフロー比較表】 機能別サポート状況

### コンテキスト・カスタマイズ

| 機能 | Claude Code | GitHub Copilot | Cursor | Codex | 備考 |
|------|-------------|-----------------|--------|-------|------|
| **Skills 機能** | ✅ `.claude/skills/` | ✅ `.github/skills/` | ✅ `.cursorrules` | ❌ | 手順・スクリプト・リソース |
| **Custom Instructions** | ✅ `.claude/instructions/` | ✅ `.github/copilot-instructions.md` | ✅ `.cursorrules` | ✅ `.github/instructions/` | 標準・自動適用 context |
| **Prompt Files** | ✅ `.claude/prompts/` | ✅ `.github/prompts/` | ✅ Prompts パネル | ✅ | 再利用テンプレート |
| **MCP（Model Context Protocol）** | ✅ 完全対応 | ✅ Pro+ | ✅（計画中） | ✅ (Copilot HQ) | 外部システム統合 |
| **Copilot Spaces** | ✅ Projects | ✅ Pro+ | ❌ | N/A | 知識ベース統合 |
| **ローカル `.md` 参照** | ✅ | ✅ `@docs` | ✅ `@filename` | ✅ | プロンプト・規約・仕様 |
| **メモリ機能** | ✅ Claude Memory | ✅（開発中） | ❌ | N/A | コンテキスト引き継ぎ |
| **Subagents & Task Distribution** | ✅ parallel (experimental) | ✅ sequential | ✅ delegation | ✅ (Agent HQ) | マルチエージェント協調 |

### エージェント・自動化

| 機能 | Claude Code | GitHub Copilot | Cursor | Codex |
|------|-------------|-----------------|--------|-------|
| **Agent Mode** | ✅（開発中） | ✅ Pro+ | ✅ | ✅ (HQ) |
| **Custom Agents** | ✅ `.claude/agents/` | ✅ `.github/agents/` | ✅ | ✅ (HQ) |
| **PR 自動生成** | ❌ | ✅ エージェント | ❌ | ✅ |
| **GitHub Issues 割当** | ❌ | ✅ エージェント | ❌ | ✅ |
| **CLI 統合** | ✅（Claude CLI） | ✅ Copilot CLI | ❌ | ✅ |
| **Slack/Teams 統合** | ✅（検討中） | ✅ Pro+ | ❌ | ✅ |
| **外部エージェント対応** | ❌ | ✅（Claude/Codex） | ✅ (MCP) | N/A |
| **Hooks & Lifecycle** | ✅ | ✅ `.github/hooks/` | ✅ | ✅ |

### ファイル・リポジトリ操作

| 機能 | Claude Code | GitHub Copilot | Cursor |
|------|-------------|-----------------|--------|
| **コードベース理解** | ✅ 強力 | ✅ | ✅ 強力 |
| **マルチファイル編集** | ✅ | ✅（エージェント） | ✅ Composer |
| **Plan Mode** | ❌ | ❌ | ✅（特色） |
| **Diff 表示** | ✅ | ✅ | ✅ |
| **コミット生成** | ❌ | ✅（エージェント） | ❌ |

### モデル選択・カスタマイズ

| 項目 | Claude Code | GitHub Copilot | Cursor |
|------|-------------|-----------------|--------|
| **Anthropic Claude** | Claude 4.6 Opus/Sonnet/Haiku | ✅ Haiku 4.5 | ✅ Opus/Sonnet/Haiku |
| **OpenAI GPT** | ❌ | ✅ GPT-5.3/5.4 mini | ✅ GPT-5.3/5.4 |
| **Google Gemini** | ❌ | ❌ | ✅ Gemini 3.1 Pro |
| **xAI Grok** | ❌ | ❌ | ✅ Grok 4.20 |
| **他社モデル** | API 経由可 | ❌ | ✅（複数選択） |
| **自動選択** | ◯ 一部 | ✅ Pro+ | ✅（デフォルト） |

### コンテキストウィンドウ・制限

| 項目 | Claude Code | GitHub Copilot | Cursor |
|------|-------------|-----------------|--------|
| **入力トークン** | 200k（Opus/Sonnet） | 200k | 200-272k |
| **出力トークン** | 1M（Opus/Sonnet） | - | 1M |
| **月間リクエスト** | 無制限（サブスク） | Pro: 300 premium | 無制限（サブスク） |
| **レート制限** | 低い | 中程度 | 個別設定可 |

---

## 【使用例】 具体的な活用シーン

### シーン 1: 新規機能実装（JAX ユーティリティの追加）

#### Claude Code での流れ

```
User: JAX テンソル操作のユーティリティモジュールを実装したい

1. Claude Code チャットで意図説明
   → Opus 4.6 が仕様理解
   
2. Workspace に `.claude/skills/jax-conventions/` 配置
   - JAX coding style guide
   - pytest pattern
   - docstring format (NumPy style)

3. Skills 自動読込
   → Opus がスタイル・テストパターン適用

4. Cowork 機能で
   - ファイル自動操作
   - 単体テスト実行
   - 型チェック実行
```

**メリット**:
- ✅ 全体的なコンテキスト理解が深い
- ✅ Cowork で自動実行可能
- ✅ 메모리 機能で次回引き継ぎ

**制限**:
- ❌ GitHub PR 自動生成不可
- ❌ GitHub Actions 統合力弱い

---

#### GitHub Copilot での流れ

```
User: GitHub Issue で機能リクエスト作成

1. Issue に Copilot を @assign
   → Pro+ エージェントが Plan 作成
   
2. .github/skills/ に JAX convention folder
   - setup.sh, validate.py など
   
3. Copilot Spaces に
   - python/conventions/
   - tests/test-patterns/
   が見える化
   
4. Copilot がコード生成
   → PR 自動作成
   → CI/CD 実行
   → Feedback コメント
```

**メリット**:
- ✅ GitHub ネイティブ統合
- ✅ PR 自動生成・管理
- ✅ CI/CD トリガー連動
- ✅ Slack/Teams 通知

**制限**:
- ❌ ローカルコンテキスト逆参照弱い
- ❌ IDE 内チャットは限定的

---

#### Cursor での流れ

```
User: IDE で Plan Mode 開始

1. "@project-context" で jax_util/ 全体理解
   → Cursor がコードベース index

2. Plan モード：
   - 修正箇所 3 つ明示
   - 順序・依存関係表示
   - ユーザー確認待ち

3. Composer が実装
   - 複数ファイル同時操作
   - Type check 並行実行
   
4. IDE 内 Diff review
   → Local commit → Push
```

**メリット**:
- ✅ IDE 内ワークフロー完結
- ✅ Plan Mode で安全操作
- ✅ 複数モデル試行可能
- ✅ ローカルファイル直接操作

**制限**:
- ❌ GitHub automation 弱い
- ❌ CI/CD 統合なし

---

### シーン 2: バグ修正・デバッグ

#### Claude Code

```md
**ワークフロー:**

1. エラーログ + コード断片を提供
   → Opus が原因特定

2. `.claude/skills/debugging/` 
   - pytest fixtures
   - debug strategy

3. 修正コード生成
   → Cowork で自動テスト実行
     (ローカル pytest)

4. Memory に "Bug Pattern: tensor shape mismatch" 記録
   → 次回の同類バグで自動参照
```

**最適プロジェクト**: 社内ツール、ワンショット修正

---

#### GitHub Copilot

```md
**ワークフロー:**

1. Issue で bug report 作成
   → Copilot code review で分析

2. Copilot @mentions で指摘・提案
   → Draft PR 自動生成

3. GitHub Actions で自動テスト
   → 失敗時に Copilot エージェント再実行

4. マージ後
   → Release Notes 自動生成（オプション）
```

**最適プロジェクト**: OSS、大規模チーム

---

#### Cursor

```md
**ワークフロー:**

1. IDE エラーメッセージをコピー
   → Chat に @stacktrace 参照

2. Plan Mode で修正戦略提示

3. 修視 preview → approve

4. Local test run
   (IDE integration)

5. Commit & push
```

**最適プロジェクト**: 個人開発、社内プロジェクト

---

### シーン 3: コードレビュー・品質走查

#### Claude Code

```md
- ✅ コード段片の詳細分析
- ✅ Docstring 自動生成
- ❌ PR ウォッチャー機能なし
- ❌ GitHub integration 弱い
```

**用途**: ローカルレビュー、教育用

---

#### GitHub Copilot

```md
- ✅ PR コメント 自動生成
- ✅ @copilot-review コマンド
- ✅ Generic security check
- ✅ Performance suggestion

**フロー:**
PR → @copilot-review → 
Issues 自動作成 → Commit suggestion
```

**用途**: チーム開発、自動品質管理

---

#### Cursor

```md
- ✅ IDE 内 multi-file diff review
- ✅ Plan Mode で安全性確認
- ❌ GitHub PR workflow 非対応
```

**用途**: Local pre-commit review

---

---

## 【推奨】 jax_util プロジェクト向けの選択基準

### **シナリオ別推奨マトリックス**

| シナリオ | 推奨エージェント | 理由 | Skills 構成 |
|---------|-----------------|------|-----------|
| **単独開発 + ローカル完結** | **Cursor** | IDE 統合、Plan Mode | `.cursorrules` |
| **チーム開発 + GitHub 活用** | **GitHub Copilot Pro+** | PR auto-gen、CI/CD 連携 | `.github/skills/` |
| **ハイレベル分析 + カスタマイズ** | **Claude Code** | Opus 4.6、メモリ | `.claude/skills/` |
| **複数モデル試行** | **Cursor** | GPT-5.4、Claude、Gemini | 切り替え可能 |
| **スタートアップ・教育** | **Claude Code** | 無料トライアル、API 対応 | 簡易 `.md` |

### **jax_util 向けベストプラクティス**

#### **階層 1: 基本方針**

```markdown
# .github/copilot-instructions.md（現在のコード）
- Docker 環境を基本とする
- 自動 venv 構築禁止
- PYTHONPATH=/workspace/python
- mdformat 適用必須

↓ 拡張推奨方針:

# 【GitHub Copilot エージェント対応】

## Skills 配置
- .github/skills/
  - jax-numpy-convention/
    - style_guide.md
    - type_hints_pattern.py
  - experiment-runner/ (Docker base)
    - setup.sh
    
## Copilot Spaces
- docs/  → Python conventions
- python/ → Implementation reference

## Agent Task Template
- Issue: "Implement feature X"
  ↓ Plan (MCP + Skills aware)
  ↓ Code (Dockerfile sync)
  ↓ Test (Docker compose run)
  ↓ PR + Auto-test
```

#### **階層 2: ファイル構成例**

```
/workspace/
├── .github/
│   ├── copilot-instructions.md     ← 現存（保持）
│   ├── agents/
│   │   └── discussion.md           ← 追加推奨（agent handoff log）
│   └── skills/
│       ├── jax-conventions.zip     ← Skills
│       └── test-pattern.zip
├── .claude/
│   └── skills/                      ← Claude Code 用（オプション）
│       └── jax-analysis/
├── documents/
│   ├── coding-conventions-python.md  ← 現存（Skills 参照先）
│   └── experiment-runner.md
└── docker/
    ├── Dockerfile                  ← 単一真実（同期必須）
    ├── requirements.txt
    └── Makefile                    ← Agent task entry
```

#### **階層 3: Agent → エージェント指示の三層別方法**

| 対象 | 方法 | 例 |
|------|------|-----|
| **GitHub Copilot** | `.github/skills/` + Issue → @assign | "Implement sparse grid solver" → PR auto-gen |
| **Claude Code** | `.claude/skills/` + chat context | "JAX util refactor" → Memory 記録 |
| **Cursor** | `.cursorrules` + Chat → Plan Mode | IDE で local test まで完結 |

---

## 【ベストプラクティス】

### 1. **コンテキスト・ドキュメント戦略**

```markdown
## 推奨テンプレート

### GitHub エージェント向け（Copilot）
- **Copilot Spaces**
  - `docs/coding-conventions-python.md`
  - `documents/experiment-runner.md`
  参照可能化 ← 自動提示
  
- **Skills**
  - `.github/skills/jax-test-pattern.zip`
    - `validate.py` (e.g., type check)
    - `README.md` (usage doc)

### ローカル IDE 向け（Cursor）
- **`.cursorrules` ファイル**
  ```
  # JAX Utility Convention
  - Use JAX API only (no NumPy direct)
  - pytest fixtures in tests/fixtures/
  - Type hints required (Python 3.10+)
  - Docker run: make test
  ```

### Claude Code 向け
- **`.claude/instructions.md`（オプション）**
  - Project specific knowledge
  - Prefer Opus 4.6 for complex analysis
  - Cowork: Docker env assumed
```

---

### 2. **マルチエージェント協調戦略**

#### **レコメンデーション**: **Hybrid Approach**

```
┌─────────────────────────────────────────┐
│   Developer Local Workflow (IDE)        │
│   ↓ Cursor + Plan Mode                  │
│   └─ @project-context で機能 sketch    │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│   GitHub Workflow (Team)                │
│   Issue → Copilot Agent @assign        │
│   ↓ PR auto-gen + CI/CD run            │
│   ↓ Code review @copilot request       │
│   └─ Merge → Memory 記録               │
└─────────────┬───────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│   Analysis & Refinement (AI Depth)     │
│   ← Claude Code (Opus 4.6)             │
│   - Complex logic review                │
│   - Performance analysis               │
│   - Memory 記録（次回参照）           │
└─────────────────────────────────────────┘
```

#### **Handoff Protocol**（ワークスペース既存）

```yaml
# ワークスペース内で既に実装済み
# agents/COMMUNICATION_PROTOCOL.md 参照

Handoff パケット例:
  from: Cursor (Local Plan)
  to: GitHub Copilot Agent
  stage: "approved sketch → implementation"
  artifacts:
    - plan_doc.md
    - docker-requirements-delta.yml
  open_questions:
    - Should agent update docker/requirements.txt?
  repo_changes: none (yet)
  status: ready_for_copilot_agent
```

---

### 3. **エラー対応・リトライ戦略**

#### **トークン超過時**

| エージェント | 対応 |
|-------------|------|
| **Claude Code** | ✅ Opus 4.6 は 1M output OK、断続的に分割要求 |
| **GitHub Copilot Pro+** | ⚠️ Premium requests 上限あり → 月内使用量確認 |
| **Cursor** | ✅ 複数モデル切り替え（Haiku に落とす等） |

#### **コンテキストウィンドウ飽和**

```markdown
**リトライ戦略:**

1. **Cursor**: ファイル選別、@filename 指定
2. **GitHub Copilot**: Copilot Spaces で確認、segment 作成
3. **Claude Code**: Memory 活用、過去コンテキスト確認
```

#### **不正な実装結果**

```markdown
**検証フロー:**

1. Local Cursor で Plan Mode review
   → Issues enumerate

2. GitHub Copilot に feedback
   @copilot "Fix issue #A, #B"
   → PR update

3. CI/CD 失敗
   → Error log → Claude Code で根本分析
   → Memory に Pattern 記録
```

---

### 4. **ドキュメント・プロンプト最適化**

#### **各エージェント向けドキュメント作成の差**

| 項目 | GitHub Copilot | Cursor | Claude Code |
|------|-----------------|--------|-------------|
| **形式** | `.md` + YAML frontmatter | `.cursorrules` | `.md` + Chat context |
| **詳細度** | 要件・制約のみ | 具体的ルール | 戦略・背景も含む |
| **リファレンス** | Copilot Spaces リンク | @filename 参照 | Chat で説明 |
| **更新頻度** | 全 issue 共通 | project base | Session ごと |

#### **`.github/copilot-instructions.md` 拡張案**

```markdown
# GitHub Copilot Instructions

## 1. 環境と規約
（現在のセクション保持）

## 2. エージェント指示【新規】

### Skills（Copilot Agent 用）
- `.github/skills/jax-test-pattern.zip`
  - pytest fixture pattern
  - type check (pyright)

### Agent Task Template
issue の形式:
```
Title: [FEATURE] Feature Name

### Acceptance Criteria
- [ ] Docker test pass
- [ ] Type check pass
- [ ] Docstring complete

### Context
@copilot-spaces jax-conventions

### Assigned
@copilot
```

## 3. Cursor ユーザ向け
`.cursorrules` で reference
```

---

### 5. **情報源・リファレンス管理**

#### **推奨保管場所**

```
docs/
├── ai-agent-guide-2026.md     ← This document
├── copilot-skills-template/
│   ├── jax-test-pattern.md
│   └── validate.py
└── cursor-cursorrules         ← Cursor 用
```

---

## 【制限事項・デメリット】

### Claude Code

| 項目 | 制限 | 対策 |
|------|------|------|
| GitHub 統合 | ❌ PR 自動生成なし | GitHub Copilot と併用 |
| スケーリング | ⚠️ 単一セッション | Memory 機能で継続 |
| 安全性 | ✅ コンテキスト管理良好 | Roles の分離に注意 |
| コスト | $ Pro Max（~月 $200） | 小規模ユースは claude.ai 無料トライアル |

### GitHub Copilot

| 項目 | 制限 | 対策 |
|------|------|------|
| ローカル IDE 操作 | ⚠️ vs Code plugin のみ | Cursor で補完 |
| コンテキスト逆参照 | ⚠️ Copilot Spaces 設定必須 | `.md` reference 明示 |
| 複数モデル選択 | ⚠️ Pro+ 必要 | GPT-5 mini がデフォルト |
| MCP 設定 | 🔧 複雑 | Doc 参照、Marketplace CLI tool 活用 |

### Cursor

| 項目 | 制限 | 対策 |
|------|------|------|
| GitHub 自動化 | ❌ 非対応 | 手動 push + GitHub Actions |
| チーム共有 | ❌ `.cursorrules` は repo only | 社内 Wiki で規約共有 |
| API 提供 | ❌ IDE のみ | Claude API 活用 |
| コンテキスト管理 | ⚠️ IDE 離脱で消失 | `.md` backup 記録 |

---

## 【情報源・引用】

| 項目 | 情報源 | URL / 日付 |
|------|--------|----------|
| **GitHub Copilot** | GitHub Official Docs | https://docs.github.com/en/copilot （2026.04） |
| **Copilot Agents** | GitHub Blog | "Getting started with agentic workflows" （2026） |
| **Agent Skills** | GitHub Docs | https://docs.github.com/en/copilot/concepts/agents/about-agent-skills |
| **MCP Protocol** | GitHub Docs | https://docs.github.com/en/copilot/how-tos/provide-context/use-mcp |
| **Claude API** | Anthropic Platform | https://platform.claude.com/docs （2026.04） |
| **Claude Code** | Claude.ai | https://claude.com/product/claude-code |
| **Cursor Documentation** | Cursor Official | https://cursor.com/docs （2026-ja） |
| **Cursor Models** | Cursor Pricing | Claude 4.6 Opus/Sonnet 対応（2026） |
| **GitHub Copilot Plans** | GitHub Pricing | Free/Pro/Pro+ （2026 現在） |

---

## 【実装ロードマップ】 jax_util 向け推奨策

### **Stage 1: 基盤整備（1-2 週）**

- [ ] `.github/skills/jax-convention.zip` 作成
  - `coding-conventions-python.md` 参照
  - pytest fixture pattern
  
- [ ] `.cursorrules` 作成（Cursor ユーザ向け）

- [ ] `.github/agents/discussion.md` initiate
  - Handoff protocol template

### **Stage 2: エージェント試行（2-3 週）**

- [ ] GitHub Copilot Pro+ trial
  - 1 issue を @copilot-agent assignment
  - PR 自動生成確認
  - Skills 読込み確認

- [ ] Cursor + Plan Mode trial
  - 小規模機能で sketch → implementation

### **Stage 3: 統合（1 ヶ月）**

- [ ] Hybrid workflow 確定
  - **Local** (Cursor) → **GitHub** (Copilot) → **Analysis** (Claude)

- [ ] Agent task template 確定
  - Issue template に `.github/agents/` 反映

- [ ] Documentation 整備
  - USER_GUIDE_JA.md に Agent workflow 追記

---

## 【まとめ】

### **jax_util プロジェクト向け最終推奨**

```markdown
最適ハイブリッド構成（2026年現在）:

┌─────────────────────────┐
│  ローカル開発 (Cursor)  │
│  - Plan Mode            │
│  - IDE 統合             │
│  - 複数モデル試行       │
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│ GitHub 自動化 (Copilot) │
│ - PR auto-gen           │
│ - Skills ready          │
│ - CI/CD integration     │
└────────┬────────────────┘
         ↓
┌─────────────────────────┐
│ 高度分析 (Claude Code)  │
│ - Opus 4.6 active       │
│ - Memory 記録           │
│ - 複雑ロジック検出      │
└─────────────────────────┘
```

**Cost-Benefit 分析**:

| 構成 | 初期投資 | 学習コスト | 効果 | スケール |
|------|--------|----------|------|--------|
| **GitHub Copilot Pro+** | $39/月 | 低 | 中 | ⭐⭐⭐⭐ |
| **Cursor** | $20/月 | 低 | 高 | ⭐⭐⭐ |
| **Claude Code** | $20/月 | 中 | 高 | ⭐⭐⭐⭐ |
| **全部** | $79/月 | 中 | 最高 | ⭐⭐⭐⭐⭐ |

→ **推奨**: GitHub Copilot Pro+ + Cursor（個人） or Claude API（分析用）

---

**ノート**: 本リサーチは 2026 年 4 月時点の公式ドキュメントに基づいています。各サービスは継続的に更新されるため、最新情報は公式サイトで確認してください。
