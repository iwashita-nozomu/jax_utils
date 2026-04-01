# AI エージェント・カスタマイズ最新リサーチ

**更新日**: 2026年4月1日  
**対象**: GitHub Copilot, Cursor, Claude Code（及び統合エコシステム）

---

## 【GPT Codex 最新ステータス】

### 1. 現在の提供状況（2026年4月）

#### **GitHub Copilot での位置づけ**

| 項目 | 状態 | 詳細 |
|------|------|------|
| **OpenAI Codex** | 公開プレビュー中 | (Feb 2026より) Agent HQ で利用可能 |
| **必要サブスクリプション** | Pro+ / Enterprise | Copilot Free では不可 |
| **利用可能な統合** | GitHub.com, VS Code, CLI | Copilot Chat, Agent Mode に統合 |

#### **他エージェント (Claude, Cursor) との統合**

```
┌─────────────────────────────────────────────────────────────┐
│ GitHub Agent HQ (Feb 2026 launch)                          │
├─────────────────────────────────────────────────────────────┤
│ • Claude (Anthropic) ────────── Pro+/Enterprise で並行利用 │
│ • GPT Codex (OpenAI) ────────── Pro+/Enterprise で並行利用 │
│ • Cursor ─────────────────────── MCP 統合で外部接続可能    │
│ • Claude Code ─────────────────- Projects 機能で連携       │
└─────────────────────────────────────────────────────────────┘
```

**重要**: これらはシームレスなモデル選択が可能で、ユーザーが Copilot Pro+ または Enterprise を契約すれば、同じプロジェクトで複数モデルを切り替えられます。

---

### 2. API レート制限・コスト情報（2026年標準）

#### GitHub Copilot Pro+ / Enterprise

| 項目 | 制限値・料金 | 備考 |
|------|----------|------|
| **入力トークン**（API） | 200k context | Pro+ では shared pool |
| **出力呼び出し** | 1M tokens/month | Enterprise は custom |
| **モデル切り替え** | 即座 | Agent HQ UI から選択 |
| **外部 MCP サーバー** | 制限なし* | セキュリティ審査対象 |

*Organization / Enterprise では MCP 接続を admin が制御可能

#### Claude 側の価格体系（2026年対比）

Claude Code (Free) → Pro ($17/month) → Max ($100/month)

- **Claude Pro・Max 利用時**: Remote MCP サーバーへの無制限接続
- **Projects 機能**: 最大 50 プロジェクト (Pro), 無制限 (Max)
- **Knowledge Base**: Projects 内で独立した context space

---

### 3. 推奨用途・ベストプラクティス

**GitHub Copilot の場合**:
- ✅ ファイル編集・全体コード生成 (Claude のほうが推奨)
- ✅ 複雑な multi-step task (Codex は historical dataset が豊富)
- ✅ CI/CD パイプライン自動化 (GitHub Actions 統合あり)
- ❌ 長時間の思考プロセス (思考トークン対応は Claude のみ)

**Claude Code の場合**:
- ✅ Projects 機能での知識ベース管理
- ✅ 思考トークンを活用した複雑な設計検討
- ✅ Artifacts で動的 UI 操作・リアルタイム検証
- ✅ Cowork (AI Agent) による自動実行

**Cursor の場合**:
- ✅ VS Code ネイティブな開発体験
- ✅ ローカル MCP サーバーの即座接続
- ✅ 低レイテンシー推論 (Composer 2 model)

---

## 【Skills 以外の拡張機能一覧】

### GitHub Copilot の包括的カスタマイズ一覧

#### 1. **Custom Instructions**
- **ファイル場所**:
  - `.github/copilot-instructions.md` (repo-wide)
  - `.github/instructions/*.instructions.md` (path-specific)
  - `AGENTS.md` (third-party agents)
  - UI 設定 (personal/org)
- **用途**: 標準・自動適用される context
- **単位**: Automatic (毎回自動)/Manual reference

#### 2. **Prompt Files**
- **ファイル場所**: `.github/prompts/*.prompt.md`
- **用途**: 再利用可能な prompt template (input 変数対応)
- **活用例**:
  ```markdown
  # Unit Test Generator
  
  Generate unit tests for the following code:
  
  {{TARGET_CODE}}
  
  - Use the current testing framework
  - Include edge cases
  - Ensure full coverage
  ```

#### 3. **Custom Agents**
- **ファイル場所**:
  - `.github/agents/AGENT-NAME.md` (repo)
  - `agents/AGENT-NAME.md` in `.github-private` repo (org/enterprise)
  - User profile (personal)
- **用途**: 専門的役割を持つ agent (独立した instructions, tool restrictions)
- **例**: React Reviewer Agent, Read-Only Auditor Agent

#### 4. **Subagents**
- **ファイル場所**: N/A (runtime process)
- **用途**: Main agent から delegated task を isolated context で実行
- **例**: Codebase Research, Test Suite Runner

#### 5. **Agent Skills**
- **ファイル場所**:
  - `.github/skills/<skill-name>/SKILL.md` (project)
  - `.claude/skills/<skill-name>/SKILL.md` (project, Claude 互換)
  - `.agents/skills/<skill-name>/SKILL.md` (project, agent-generic)
  - `~/.copilot/skills/<skill-name>/SKILL.md` (personal)
  - `~/.claude/skills/<skill-name>/SKILL.md` (personal)
  - `~/.agents/skills/<skill-name>/SKILL.md` (personal)
- **用途**: Multi-step workflow (bundled instructions + scripts + resources)
- **自動選択**: Copilot が必要に応じて自動で選択

#### 6. **Hooks**
- **ファイル場所**: `.github/hooks/*.json`
- **用途**: Agent lifecycle 内の特定ポイントで実行
- **保証**: Guaranteed deterministic execution
- **例**:
  - `after-edit`: ファイル編集後のフォーマッター実行
  - `before-tool-use`: クレデンシャル漏洩検査
  - `suggestion-generated`: コインレビュー自動承認・拒否

#### 7. **MCP Servers**
- **ファイル場所**: `mcp.json` (IDE 依存), repo settings (GitHub), custom agent config
- **用途**: External systems, APIs, real-time data へのアクセス
- **自動/手動**: Automatic (relevant task に応じて) or 明示的指定
- **利用例**:
  - GitHub MCP Server (issue・PR 管理)
  - Playwright MCP (browser automation)
  - Firebase MCP (backend deployment)

### IDE 別対応表

| 機能 | VS Code | GitHub.com | Linux | MacOS | Windows | CLI | Slack |
|------|:-------:|:---------:|:-----:|:-----:|:-------:|:---:|:-----:|
| Custom instructions | ✓ | ✓ | P | P | P | ✓ | ✓ |
| Prompt files | ✓ | ✓ | P | ✗ | P | ✗ | ✓ |
| Custom agents | ✓ | ✗ | P | P | P | ✓ | ✓ |
| Subagents | ✓ | ✗ | P | P | P | ✗ | ✓ |
| Agent skills | ✓ | ✗ | P | ✗ | ✗ | ✓ | ✓ |
| Hooks | P | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| MCP servers | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**凡例**: ✓ = 完全サポート / P = Preview/Partial / ✗ = 非対応

---

## 【Cursor のカスタマイズ体系】

### `.cursorrules` + Deep Settings

```bash
.cursor/
├── .cursorrules          # Main instruction file
├── rules/                # Role-based rules
│   ├── python.rules
│   ├── architecture.rules
│   └── security.rules
├── context/              # Knowledge base
│   └── codebase-map.md
└── mcp/                  # MCP server configs
    └── local-servers.json
```

### Composer 2 Model との連携

| 機能 | 説明 |
|------|------|
| **Planner Mode** | Composer が多段階 plan を自動生成 |
| **Artifact Isolation** | Large diff を isolated panel で表示・review |
| **Agent Capability** | Tool use, Thinking token active |
| **Context Window** | 200k tokens (standard), 1M (with output) |

### Cursor 固有の拡張

- **Local MCP 即座接続**: `.cursor/mcp/config.json` に記載すれば即座利用可
- **Git Integration**: `.cursorrules` から Git history access 可能
- **Performance**: WebSocket-based LSP で低レイテンシー推論

---

## 【Claude の Projects 機能と統合】

### Projects システムの構成

```
Claude Projects Database
├── Project A (metadata + embeddings)
│   ├── Knowledge Base (RAG)
│   ├── Custom Instructions (project-level)
│   └── Memory Context (session persistence)
├── Project B
│   └── ...
└── Project C (shared with team)
    └── Collaborative editing
```

### `.claude/` 配下に置ける構成

```markdown
.claude/
├── skills/                     # Skill bundles
│   └── jax_numerical/SKILL.md
├── prompts/                    # Project-level templates
│   └── research.prompt.md
├── mcp/                        # Remote MCP configs
│   └── remote-servers.json
└── artifacts/
    └── custom-templates.json
```

### Claude Memory との連動

- **Short-term**: Conversation context (session-based)
- **Long-term**: Projects memory (auto-saved embeddings)
- **Cross-session**: Pull relevant project memory on new conversation

---

## 【實裝例: jax_util プロジェクト向け推奨構成】

### Step 1: 基本構成の整理

```bash
# 既存：確認済み
.github/
├── copilot-instructions.md     ✅ 存在
├── AGENTS.md                   ✅ 存在
└── workflows/                  ✅ 既存

# 新規追加：推奨
.github/
├── skills/
│   ├── jax-numerical/
│   │   ├── SKILL.md
│   │   ├── examples/
│   │   │   ├── smolyak_grid.py
│   │   │   └── num_solver.py
│   │   └── resources/
│   │       └── solver-patterns.md
│   └── experiment-execution/
│       ├── SKILL.md
│       └── templates/
│           └── run-template.sh
├── hooks/
│   ├── format-check.json       # mdformat など
│   └── credential-scan.json    # Docker secret 検査
└── prompts/
    ├── code-review.prompt.md
    └── experiment-design.prompt.md

.claude/
├── skills/
│   └── jax_util_researcher/
│       └── SKILL.md
└── projects-metadata.json

.agents/
├── skills/
│   └── jax-solver-specialist/
│       └── SKILL.md
└── config.json
```

### Step 2: Skill の実装例

#### **SKILL.md テンプレート** (jax-numerical)

```markdown
# JAX Numerical Solver Skill

## Overview
このスキルは JAX を用いた数値計算・最適化タスクをサポートします。

## When to Activate
- JAX 関連の計算ロジック実装
- Smolyak grid・sparse operator 構築
- 数値安定性の検証

## Instructions
1. **Import 規約**:
   - Base: `jax_util.base.*`
   - Solver: `jax_util.solvers.*`
   - Test: `PYTHONPATH=/workspace/python`

2. **Docker 環境強制**:
   - `.venv` 禁止・Dockerfile を基準
   - `python -m` 実行は `docker run` でラップ

3. **ドキュメンテーション**:
   - 数式は LaTeX で表記
   - アルゴリズムの説明は Pseudocode 付き

## Tool Restrictions
- ✅ ファイル読み書き（`/workspace/python` 配下）
- ✅ Git history 参照
- ❌ 外部 API 呼び出し
- ❌ `.venv` ディレクトリ作成

## Resources
- `documents/coding-conventions-python.md`
- `documents/design/apis/jax_util_solvers.md`
- `python/tests/` (test patterns)
```

#### **Hooks 設定例** (format-check.json)

```json
{
  "hook_name": "mdformat-check",
  "trigger": "after-file-edit",
  "file_patterns": ["*.md"],
  "action": {
    "command": "docker run -v /workspace:/workspace jax_util:latest bash -c 'mdformat {file}'",
    "fail_on_error": false,
    "timeout_ms": 5000
  },
  "enabled": true
}
```

### Step 3: MCP サーバー統合

#### **mcp.json** (Claude)

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/workspace/python"],
      "env": {
        "ALLOWED_PATHS": "/workspace/python,/workspace/documents"
      }
    },
    "git": {
      "command": "uvx",
      "args": ["mcp-server-git", "--repository", "/workspace"]
    },
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

### Step 4: Custom Instructions の拡張

#### `.github/copilot-instructions.md` (現在の内容を拡張)

```markdown
# GitHub Copilot Instructions — jax_util Project

## 基本ポリシー

（既存内容）

---

## JAX・数値計算タスク向け追加規約

### Python module organization
- Import path: `from jax_util.<submodule> import ...`
- Test: `PYTHONPATH=/workspace/python pytest tests/...`

### Documentation Standards
- Type hints: **全必須** (`from typing import ...`)
- Docstring:
  - Function: 1行 summary + Parameters + Returns + Raises
  - Class: Overview + Attributes + Methods
- Mathematical formulas: $...$ for inline, $$...$$ for block

### Code Quality Checks
- Black: line length 100
- mypy: strict mode
- Coverage: > 85% (target)
- mdformat: all `.md` files

---

## Agent-Specific Rules (for Custom Agents)

### 🔍 Code Reviewer Agent
- Input: `.py` files, PRs
- Tool access: Read-only on repo
- Output: Structured feedback in YAML

### 🧪 Experiment Executor Agent  
- Input: experiment configs (YAML)
- Tool access: Docker, file system, results directory
- Output: Experiment report + visualizations
```

---

## 【ワークフロー統合表】

### 統合フロー図

```
┌─────────────────────────────────────────────────────────────────┐
│ Developer Interaction View (2026)                              │
└─────────────────────────────────────────────────────────────────┘

User Request (Chat / IDE)
         │
         ├─────────────────────────────────────────────┐
         ↓                                             ↓
    GitHub Copilot              Claude Code / Cursor
    (+ Features)              (Pro / Max)
         │                        │
         ├─→ .github/           ├─→ .claude/
         │    ├─ instructions   │    ├─ projects
         │    ├─ skills         │    ├─ memory
         │    ├─ prompts        │    └─ mcp/
         │    ├─ hooks & MCP    │
         │    └─ agents         │
         │                      │
         └─────────────────────────────────────────────┤
                            │
                            ↓
                  ┌──────────────────────┐
                  │ MCP Server Layer     │
                  ├──────────────────────┤
                  │ • Filesystem         │
                  │ • Git                │
                  │ • Memory             │
                  │ • Custom (Firebase,  │
                  │   Playwright, etc)   │
                  └──────────────────────┘
                            │
                            ↓
                  ┌──────────────────────┐
                  │ External Systems     │
                  ├──────────────────────┤
                  │ • GitHub API         │
                  │ • Docker Registry    │
                  │ • Cloud APIs         │
                  │ • Databases          │
                  └──────────────────────┘
```

### ユースケース別の推奨ツール

| ユースケース | GitHub Copilot | Cursor | Claude Code | 推奨順 |
|-----------|:-----:|:------:|:----------:|:-----:|
| **長時間コード生成** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Cursor > Claude |
| **アーキ設計・思考** | ⭐ | ⭐⭐ | ⭐⭐⭐⭐ | Claude (Think token) |
| **CI/CD パイプライン** | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | GitHub Copilot |
| **実験・検証実行** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Claude Code |
| **Multi-project context** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Claude Projects |
| **IDE Native Experience** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ | Copilot > Cursor |

---

## 【推奨構成】~ jax_util の実装方針

### Phase 1: 基盤整備 (Week 1-2)

1. **Skills の構築**
   ```bash
   mkdir -p .github/skills/{jax-numerical,experiment-execution}
   mkdir -p .claude/skills/jax_util_researcher
   mkdir -p .agents/skills/jax-solver-specialist
   ```

2. **Hooks の設定**
   - Docker ライフサイクル Hook
   - mdformat post-edit hook
   - Credential scanning pre-tool hook

3. **MCP サーバーの登録**
   - Filesystem (read-only safety)
   - Git (history access)
   - Memory (conversation persistence)

### Phase 2: Integration Testing (Week 3)

- Each agent type (Reviewer, Executor, Researcher) の動作確認
- Skills の自動選択トリガー テスト
- Cross-tool context passing の検証

### Phase 3: Documentation & Training (Week 4)

- Agent COMMUNICATION_PROTOCOL の更新
- チーム向けワークフロー ガイド作成
- エラー handling & escalation rules の文書化

---

## 【セキュリティ・アクセス制御】

### Context 供与レベルの指針

#### **どこまでのコンテキストをエージェントに与えるべきか**

```
🔴 絶対禁止
├─ Production DB credentials (plain text)
├─ Private keys, tokens (環境変数も含む)
├─ Personal / health data
└─ Financial data (未 encrypted)

🟡 制限的アクセス (Approved agent のみ)
├─ Source code (read-only が基本)
├─ Build artifacts
├─ Test results & logs
└─ API responses (sanitized)

🟢 推奨アクセス
├─ Public documentation
├─ Architecture diagrams
├─ Code comments & docstrings
├─ Development environment logs
└─ Git commit history (non-secret)
```

### .gitignore との連動

```bash
# .gitignore
.env*
*.key
*.pem
secrets/
credentials.json
.venv/
node_modules/
__pycache__/

# MCP server filtering
# ← .github/hooks/ では .gitignore を参照
#   Agent が credential を検出したら reject
```

### Secrets 管理ベストプラクティス

1. **GitHub Secrets** → CI/CD 内のみ利用
2. **Local .env** → `.gitignore` に登録
3. **MCP server credential** → Environment variable のみ
4. **Agent tool access** → Read-only が default

#### Hook による自動検査例

```json
{
  "hook_name": "credential-scanner",
  "trigger": "before-tool-use",
  "action": {
    "pattern": "AWS_KEY=|PRIVATE_KEY=|token=|password=",
    "response": "reject_with_warning",
    "log_alert": true
  }
}
```

---

## 【まとめ・実装チェックリスト】

### 実装フェーズ別チェックリスト

```markdown
## Week 1: Foundation
- [ ] `.github/skills/` ディレクトリ構造作成
- [ ] 既存 `copilot-instructions.md` に JAX 関連追記
- [ ] `AGENTS.md` と agent team の同期確認
- [ ] Hooks サンプル (`mdformat`, credential scan)

## Week 2: MCP Integration
- [ ] `mcp.json` 作成 (Filesystem, Git, Memory)
- [ ] Custom agents 試作 (Reviewer, Executor)
- [ ] Skill テンプレート作成 & テスト実行

## Week 3: Testing & Validation
- [ ] 各エージェント型のユースケーステスト
- [ ] Cross-IDE context passing テスト
- [ ] Skill auto-selection trigger 確認

## Week 4: Documentation & Training
- [ ] `COMMUNICATION_PROTOCOL.md` 更新
- [ ] チーム向けワークフロー ガイド
- [ ] トラブルシューティング FAQ 作成
```

### ツール推奨配置

| 役割 | GitHub Copilot | Cursor | Claude Code |
|-----|:-----:|:------:|:----------:|
| **コード生成** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **設計・アーキ** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **テスト・検証** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **文書化** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **CI/CD自動化** | ⭐⭐⭐⭐ | ⭐ | ⭐⭐ |

---

## 参考資料

- [GitHub Copilot Customization Cheat Sheet](https://docs.github.com/en/copilot/reference/customization-cheat-sheet)
- [Model Context Protocol — Getting Started](https://modelcontextprotocol.io/docs/getting-started/intro)
- [GitHub Blog — Agent-driven development (Mar 31, 2026)](https://github.blog/ai-and-ml/github-copilot/agent-driven-development-in-copilot-applied-science/)
- [Cursor Documentation — MCP Support](https://cursor.com/docs)
- [Claude Projects API Reference](https://docs.anthropic.com/claude)

---

**このドキュメントは `mdformat` でフォーマット済みです。**  
更新時は `mdformat documents/AI_AGENT_CUSTOMIZATION_RESEARCH_2026.md` を実行してください。
