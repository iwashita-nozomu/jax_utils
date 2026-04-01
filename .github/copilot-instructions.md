日本語でお願いします。
コメントアウトは丁寧に書いてください。
./pythonを対象にしてください。
./jupyterも対象にしてください。
複雑な分岐は避け，シンプルなコードを心がけてください。

# GitHub Copilot Instructions

## 環境と規約

./documentsにコーディング規約を置いています。規約に従ってください。
特に [documents/coding-conventions-project.md](../documents/coding-conventions-project.md) の「3. Docker環境の方針」セクションを必ず確認してください。

## Python環境

**⚠️ 仮想環境（.venv / venv / conda など）の構築は厳禁です。**

- Docker のBashが既定の実行環境です
- `.venv` / `.venv-*` などのディレクトリを作成してはいけません
- `python -m venv` / `virtualenv` / `conda` の使用は禁止です
- Dockerfile と docker/requirements.txt が単一真実です
- 足りないモジュールは Dockerfile に追記し、`pip install` でコンテナ内に導入してください
- 追加時は Dockerfile の更新を同時に行ってください

テスト・スクリプトは `PYTHONPATH=/workspace/python` を基本とします。

## ファイル操作

- `*/Archive/*` は現在使わないコードの保管場所です。新規実装や修正の対象外としてください

## ドキュメント

- mdの記法に注意してください
- 数式、図を積極的に活用してください
- md ファイル編集後は `mdformat` で書式を直してください

## Skills ライブラリ

プロジェクト全体を支援するカスタム Skill を複数提供しています。

### 利用可能な Skill

#### Skill 1: Static Check
静的型チェック・テスト・Docker イメージ検証

```bash
python3 .github/skills/01-static-check/run-check.py --verbose
```

#### Skill 2: Code Review
多層的コードレビュー（A層: linter, B層: アーキテクチャ, C層: 学術妥当性）

```bash
python3 .github/skills/02-code-review/run-review.py --phase "B"
```

#### Skill 3: Run Experiment
実験実行管理（初期化→学習→検証→テスト→レポート）

```bash
python3 .github/skills/03-run-experiment/run-exp.py --config config/experiment.yaml
```

#### Skill 4: Critical Review
実験学術妥当性検証（Reviewer/Statistician/Domain Expert 視点）

```bash
python3 .github/skills/04-critical-review/run-critical.py --result results/exp.json
```

#### Skill 5: Research Workflow
長期研究進捗管理＆メタ分析

```bash
python3 .github/skills/05-research-workflow/run-workflow.py --log diary/
```

#### Skill 6: Comprehensive Review ✨ (NEW)
ドキュメント・Skill・ツール全体の統合レビュー

- **Phase 1**: ドキュメント品質（broken link、用語統一、循環参照）
- **Phase 2**: Skill 整合性（依存関係、重複、実装漏れ）
- **Phase 3**: ツール保全性（実装状況、テストカバレッジ）
- **Phase 4**: 統合テスト（CLI、Actions、Docker）
- **Phase 5**: レポート生成（健全性指標、ロードマップ）

```bash
# 全フェーズ実行（詳細出力）
python3 .github/skills/06-comprehensive-review/run-review.py --verbose

# レポート保存
python3 .github/skills/06-comprehensive-review/run-review.py --save-report --output markdown

# 特定フェーズのみ
python3 .github/skills/06-comprehensive-review/run-review.py --phases "1,2,3"

# JSON 形式で出力
python3 .github/skills/06-comprehensive-review/run-review.py --output json
```

参照: `.github/skills/06-comprehensive-review/README.md`

### Skill Hub

すべての Skill メタデータ・実装・相互依存関係:

- **Hub**: `.github/skills/README.md`
- **統合計画**: `documents/SKILLS_INTEGRATION_PLAN.md`
- **実装ガイド**: `documents/SKILL_IMPLEMENTATION_GUIDE.md`

### Skill 実行時の共通オプション

- `--verbose` : 詳細出力
- `--output {json,markdown,text}` : 出力形式指定
- `--save-report` : レポートをファイルに保存
