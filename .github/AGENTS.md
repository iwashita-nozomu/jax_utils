# Agent Team — Entry Point

このファイルは GitHub 上の入口です。team shape や role 一覧はここへ再掲しません。

## Canonical Sources

- Team definition and write policy: `agents/agents_config.json`
- Communication protocol: `agents/COMMUNICATION_PROTOCOL.md`
- Human-facing team summary: `agents/README.md`
- Runtime implementation: `scripts/agent_tools/agent_team.py`
- Repo integration guide: `documents/AGENTS_COORDINATION.md`

## GitHub-Side Rules

- Team summary は `agents/README.md` を読む。
- role permission と handoff は `agents/agents_config.json` と `agents/COMMUNICATION_PROTOCOL.md` を正本にする。
- GitHub Actions の automation mirror は `.github/workflows/agent-coordination.yml` を使う。
- このファイルへ role 一覧や flow の説明を複製しない。

## Skills Library

包括的なレビューとワークフロー自動化を支援する Skill ライブラリを提供します。

### Available Skills

#### Skill 1: Static Check
**目的**: 型・テスト・Docker 検証（自動化 ⭐⭐⭐⭐⭐）

- **対象**: Python/C++ ファイル、テスト、Docker イメージ
- **検査内容**: Pyright, mypy, pytest, docker build
- **CLI**: `claude --copilot-custom skill-static-check`
- **参照**: `.github/skills/01-static-check/README.md`

#### Skill 2: Code Review
**目的**: 多層的なコードレビュー検証（半自動化 ⭐⭐⭐）

- **対象**: Pull Request コード、設計、セキュリティ
- **検査内容**: A層（linter）、B層（アーキテクチャ）、C層（学術妥当性）
- **CLI**: `claude --copilot-custom skill-code-review`
- **参照**: `.github/skills/02-code-review/README.md`

#### Skill 3: Run Experiment
**目的**: 5段階実験実行管理

- **対象**: JAX/Smolyak 実験、パラメータスイープ、結果追跡
- **検査内容**: 初期化→学習→検証→テスト→レポート
- **CLI**: `claude --copilot-custom skill-run-experiment`
- **参照**: `.github/skills/03-run-experiment/README.md`

#### Skill 4: Critical Review
**目的**: 実験学術妥当性検証（3役割 assessment）

- **対象**: 実験結果、論文、仮説検証
- **検査内容**: Reviewer, Statistician, Domain Expert 視点での評価
- **CLI**: `claude --copilot-custom skill-critical-review`
- **参照**: `.github/skills/04-critical-review/README.md`

#### Skill 5: Research Workflow
**目的**: 長期反復管理＆メタ分析

- **対象**: 研究進捗、文献管理、仮説変遷
- **検査内容**: ブランチ管理、経時追跡、統合分析
- **CLI**: `claude --copilot-custom skill-research-workflow`
- **参照**: `.github/skills/05-research-workflow/README.md`

#### Skill 6: Comprehensive Review ✨ (NEW)
**目的**: ドキュメント・Skill・ツール全体の統合レビュー

- **対象**: すべてのドキュメント、Skill定義、実装スクリプト、統合設定
- **検査内容**: 
  - Phase 1: ドキュメント品質（broken link、用語統一、循環参照）
  - Phase 2: Skill 整合性（依存関係、重複機能、実装漏れ）
  - Phase 3: ツール保全性（実装状況、テストカバレッジ）
  - Phase 4: 統合テスト（CLI、Actions、Docker）
  - Phase 5: レポート生成（健全性指標、ロードマップ）
- **使用法**:
  ```bash
  # 全フェーズ実行
  python3 .github/skills/06-comprehensive-review/run-review.py --verbose
  
  # レポート保存
  python3 .github/skills/06-comprehensive-review/run-review.py --save-report
  
  # 特定フェーズのみ
  python3 .github/skills/06-comprehensive-review/run-review.py --phases "1,2,4"
  ```
- **CLI**: `claude --copilot-custom skill-comprehensive-review`
- **参照**: `.github/skills/06-comprehensive-review/README.md`
- **実装メモ**: 1,779行 Python コード、5つの独立検査器、自動 Markdown レポート生成
- **ステータス**: ✅ 実装完了、テスト合格、実行準備完了

### Skill Hub Central

すべての Skill のメタデータ、相互依存関係、実行順序は以下で管理されます：

- **Skill メタデータ**: `.github/skills/README.md`
- **統合計画**: `documents/SKILLS_INTEGRATION_PLAN.md`
- **実装ガイド**: `documents/SKILL_IMPLEMENTATION_GUIDE.md`
- **実行スクリプト**: `scripts/skill_*.py`

### Integration Points

- **GitHub Actions**: `.github/workflows/` に Skill 実行ワークフロー
- **Makefile**: `make skill-*` で各 Skill 実行
- **CLI Tools**: Copilot/Claude/Cursor インストラクション対応
- **Local**: Docker 環境下で実行可能
