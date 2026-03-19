# 本プロジェクト完全マッピング報告書

**作成日:** 2026-03-19\
**調査対象:** `/workspace` 全体\
**レポート範囲:** スクリプト・ツール、ワークフロー、構成ファイル、CI/CD パイプライン、規約体系

______________________________________________________________________

## セクション1: スクリプト・ツール完全リスト

### 1.1 scripts/ 直下のスクリプト（11個）

| #   | ファイル名                 | 言語 | 用途                                   | 実行方法                                                      | 入出力                                   |
| --- | -------------------------- | ---- | -------------------------------------- | ------------------------------------------------------------- | ---------------------------------------- |
| 1   | `git_config.sh`            | Bash | Git ユーザー・デフォルトブランチ設定   | `bash scripts/git_config.sh`                                  | 標準出力                                 |
| 2   | `git_init.sh`              | Bash | Git 初期化（`git_config.sh` 呼び出し） | `bash scripts/git_init.sh` / `make git_init`                  | 標準出力                                 |
| 3   | `git_repo_init.sh`         | Bash | Python パッケージディレクトリ作成      | `bash scripts/git_repo_init.sh`                               | `python/<package-name>/`                 |
| 4   | `view_conventions.sh`      | Bash | 規約ファイル表示・検索                 | `bash scripts/view_conventions.sh [keyword]`                  | コンソール表示                           |
| 5   | `read_conventions.sh`      | Bash | 規約ファイル一覧表示                   | `bash scripts/read_conventions.sh`                            | テーブル形式表示                         |
| 6   | `setup_worktree.sh`        | Bash | ワークツリー・ブランチ作成 ⚠️          | `bash scripts/setup_worktree.sh <branch> [説明]`              | `.worktrees/<branch>/`                   |
| 7   | `guide.sh`                 | Bash | 作業フロー全体ガイド表示               | `bash scripts/guide.sh`                                       | テキストガイド                           |
| 8   | `create_toml.sh`           | Bash | pyproject.toml テンプレート作成        | `bash scripts/create_toml.sh <pkg-name>`                      | `pyproject.toml`                         |
| 9   | `jsonl_to_md.sh`           | Bash | JSONL → Markdown 変換                  | `bash scripts/jsonl_to_md.sh <in> <out>`                      | Markdown ファイル                        |
| 10  | `extract_deps_from_svg.sh` | Bash | 依存関係抽出（Graphviz）               | `bash scripts/extract_deps_from_svg.sh deps.dot [--internal]` | コンソール出力                           |
| 11  | `run_pytest_with_logs.sh`  | Bash | pytest 実行・ログ保存                  | `bash scripts/run_pytest_with_logs.sh`                        | `python/tests/logs/[YYYYMMDD]-[HHMMSS]/` |

**前提/関連:**

- `git_config.sh` は `git_init.sh` より呼び出し
- `setup_worktree.sh` は `.github/copilot-instructions.md` で指定（ドキュメント作成対象）
- `run_pytest_with_logs.sh` は CI パイプラインで実行推奨

______________________________________________________________________

### 1.2 scripts/tools/ 配下（11個）

| #   | ファイル名                  | 言語   | 用途                                      | 実行方法                                                          | 入出力                                                                |
| --- | --------------------------- | ------ | ----------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------- |
| 1   | `create_worktree.sh`        | Bash   | ワークツリー・ブランチ作成（標準推奨） ⭐ | `bash scripts/tools/create_worktree.sh <branch> [path]`           | `.worktrees/<branch>/` + `WORKTREE_SCOPE.md`                          |
| 2   | `check_worktree_scopes.sh`  | Bash   | 全ワークツリーのスコープ検査              | `bash scripts/tools/check_worktree_scopes.sh`                     | `reports/worktree_scope_report.txt`                                   |
| 3   | `format_markdown.py`        | Python | Markdown ファイル正規化                   | `python scripts/tools/format_markdown.py [files...]`              | ファイル直接修正                                                      |
| 4   | `fix_markdown_docs.py`      | Python | Markdown ドキュメント修正                 | `python scripts/tools/fix_markdown_docs.py`                       | `reports/markdown_fixes_report.txt`                                   |
| 5   | `audit_and_fix_links.py`    | Python | 相対リンク検査・修正                      | `python scripts/tools/audit_and_fix_links.py`                     | `reports/link_audit.txt`                                              |
| 6   | `organize_designs.py`       | Python | 設計ファイル整理（サブモジュール別）      | `python scripts/tools/organize_designs.py [--dry-run]`            | `documents/design/<submodule>/`, `reports/design_organize_report.txt` |
| 7   | `create_design_template.py` | Python | 設計テンプレート生成                      | `python scripts/tools/create_design_template.py <module-path>`    | `<module>/design/template.md`                                         |
| 8   | `find_similar_designs.py`   | Python | 設計ファイル類似度検出                    | `python scripts/tools/find_similar_designs.py [--threshold 0.85]` | `reports/similar_designs_report.txt`                                  |
| 9   | `find_similar_documents.py` | Python | ドキュメント全般類似度検出                | `python scripts/tools/find_similar_documents.py`                  | `reports/similar_documents_report.txt`                                |
| 10  | `tfidf_similar_docs.py`     | Python | TF-IDF 高度な類似度分析                   | `python scripts/tools/tfidf_similar_docs.py [--output file]`      | `reports/tfidf_similar_documents_report.txt`                          |
| 11  | `find_redundant_designs.py` | Python | 完全一致する設計と判定、削除              | `python scripts/tools/find_redundant_designs.py [--delete]`       | `reports/redundant_designs.txt`                                       |

**前提/関連:**

- `create_worktree.sh` は `setup_worktree.sh` より先進版（推奨）
- `check_worktree_scopes.sh` は CI で定期実行推奨
- ドキュメント処理スクリプトは `.github/workflows/` で活用可能
- `organize_designs.py` / `create_design_template.py` はセット運用

______________________________________________________________________

### 1.3 scripts/hlo/ 配下（1個）

| #   | ファイル名               | 言語   | 用途                     | 実行方法                                                             | 入出力                 |
| --- | ------------------------ | ------ | ------------------------ | -------------------------------------------------------------------- | ---------------------- |
| 1   | `summarize_hlo_jsonl.py` | Python | HLO JSONL ログ集計・分析 | `python scripts/hlo/summarize_hlo_jsonl.py <input.jsonl> [--top 50]` | コンソール出力（統計） |

**前提/関連:**

- 入力: `jax_util.hlo.dump_hlo_jsonl()` で生成した JSONL ファイル
- 出力: レコード数、タグ出現回数、HLO op 頻度分析

______________________________________________________________________

### 1.4 合計統計

```
scripts/直下:           11個
scripts/tools/:         11個
scripts/hlo/:           1個
─────────────────────────────
合計:                   23個

言語別内訳:
  Bash: 14個 (61%)
  Python: 9個 (39%)

用途別分類:
  Git・ブランチ管理:    3個
  ドキュメント処理:     5個
  ワークツリー管理:     2個
  設計ファイル整理:     7個
  テスト実行:           1個
  HLO 分析:             1個
  ユーティリティ:       3個
  その他:               1個
```

______________________________________________________________________

## セクション2: .github/ CI/CD パイプライン・構成

### 2.1 GitHub Actions ワークフロー

| ファイル                                   | トリガー                | ジョブ群                                | 主処理                      | 状態                |
| ------------------------------------------ | ----------------------- | --------------------------------------- | --------------------------- | ------------------- |
| `.github/workflows/ci.yml`                 | push (main, master), PR | `test`                                  | pytest + pyright            | 実装済み（基本）    |
| `.github/workflows/agent-coordination.yml` | `workflow_dispatch`     | `coordinator`, `reviewer`, `integrator` | 静的解析・テスト + ログ集計 | テンプレート状態 ⚠️ |

### 2.2 CI/CD パイプラインの詳細

#### 2.2.1 `ci.yml` （実装済み・実行中）

```yaml
トリガー: push/PR on main/master

ジョブ: test (ubuntu-latest, Python 3.10)
  ステップ:
    1. checkout
    2. Python 3.10 setup
    3. システムパッケージ（graphviz）
    4. pip 依存インストール（docker/requirements.txt）
    5. パッケージインストール（python/）
    6. pytest 実行（python/tests/experiment_runner）

実行対象: experiment_runner テストのみ（限定的）
```

**問題点:**

- `experiment_runner` テストのみで、全テスト未実施
- `pyright`, `ruff` は CI に含まれていない
- `pytest` のログ保存なし

#### 2.2.2 `agent-coordination.yml` （テンプレート状態）

```yaml
トリガー: workflow_dispatch（手動実行）

ジョブチェーン:
  1. coordinator
     → タスク準備・通知（未実装）
  
  2. reviewer (needs: coordinator)
     → pyright / pytest 実行
     → 変更ファイル列挙
  
  3. integrator (needs: coordinator, reviewer)
     → 統合テスト実行
     → レポートアップロード（future）

状態: 実装骨子のみ（実際の統合処理未実装）
```

**用途:** エージェント間タスク分配・相互検証

______________________________________________________________________

### 2.3 .github/ 配下の全ファイル

| ファイル                           | 内容                                    | 担当             |
| ---------------------------------- | --------------------------------------- | ---------------- |
| `AGENTS.md`                        | エージェント運用指針（4つの必須ロール） | 運用指針         |
| `copilot-instructions.md`          | GitHub Copilot 実装指示書               | コパイロット指示 |
| `agents/discussion.md`             | エージェント間議論ログテンプレート      | 議論記録         |
| `workflows/ci.yml`                 | 基本 CI（pytest）                       | CI/CD            |
| `workflows/agent-coordination.yml` | エージェント調整ワークフロー            | CI/CD            |

______________________________________________________________________

## セクション3: プロジェクトルート構成ファイル

### 3.1 主要ファイル一覧

| ファイル/ディレクトリ       | 用途                     | 内容概要                             |
| --------------------------- | ------------------------ | ------------------------------------ |
| **Makefile**                | ビルドターゲット         | `git_init` (1ターゲット)             |
| **pyproject.toml**          | Python パッケージ定義    | jax_util / Python 3.10+ / setuptools |
| **pyrightconfig.json**      | Pyright 設定参照         | `extends: ./pyproject.toml`          |
| **.markdownlint.json**      | Markdown リント設定      | MD003/004/013/029/030 ルール定義     |
| **docker/Dockerfile**       | コンテナイメージ定義     | CUDA 12.2 / Python 3.10 / GPU JAX    |
| **docker/requirements.txt** | 依存パッケージ           | pytest, scipy-stubs, optax, pyright  |
| **.gitignore**              | Git 除外パターン         | 仮想環境、キャッシュ、ログ           |
| **.devcontainer/**          | VS Code Dev Container    | コンテナ開発環境設定                 |
| **.vscode/**                | VS Code プロジェクト設定 | ワークスペース設定等                 |

### 3.2 Makefile の詳細

```makefile
定義内容:

  git_init:
    実行: ./scripts/git_init.sh
    機能: Git 初期化（ユーザー・ブランチ設定）
    
概観:
  - ターゲット数: 1個（最小限）
  - 拡張予定: CI ターゲット、テスト実行ターゲット、ドキュメント生成等
```

**問題点:** ターゲット数が少なく、開発効率化フローが Makefile 化されていない

### 3.3 pyproject.toml の詳細

```toml
[project]
  name: jax_util
  version: 0.1.0
  requires-python: >=3.10
  dependencies: []  # 無し（依存は Docker で管理）

[project.optional-dependencies]
  dev: pytest>=8, ruff>=0.5, scipy-stubs

[tool.setuptools]
  package-dir: {"": "python"}
  packages: find (python/ 配下自動検出)

[tool.ruff]
  line-length: 100
  target-version: py310
  lint.select: E, F, I

[tool.pyright]
  typeCheckingMode: strict
  include: ./python/jax_util
  exclude: archive, tests, __pycache__, build, dist
  extraPaths: ./python, ./python/jax_util
  stubPath: ./python/typings
  reportUnknown*: none  (型情報薄い外部ライブラリに配慮)

[tool.pytest.ini_options]
  testpaths: python/tests
```

### 3.4 docker/Dockerfile の詳細

```dockerfile
ベース: nvidia/cuda:12.2.0-devel-ubuntu22.04
Python: python3 (バージョン指定なし、システムデフォルト)

インストール:
  - ビルドツール: gcc, pkg-config, cmake
  - Python 環境: python3-pip, python3-venv
  - JAX: jax[cuda12] (コンテナ内 CUDA 使用)
  - その他: jaxtyping, typeguard, equinox, pytest, scipy, optax, pyright, pydeps

PYTHONPATH: /workspace/python
git safe.directory: /mnt/git/template.git, /mnt/git/jax_util.git

CMD: /bin/bash
```

**インストールモジュール:**

```
Core:
  - jax[cuda12] (JAX + CUDA Toolkit 12)
  - jaxtyping, equinox (JAX 関連)
  - optax (最適化)
  - scipy (数値計算)

Development:
  - pytest (テスト)
  - pyright (型チェック)
  - pydeps (依存関係分析)

Stubs/Type:
  - scipy-stubs, typeguard (型情報)
```

### 3.5 .markdownlint.json ルール

```json
規則:
  MD003: 一貫した記号スタイル
  MD004: 一貫した箇条書きスタイル
  MD029: 順序付きリスト番号（1 or ordered）
  MD030: リスト周辺空白（single/multi）
  
無効化:
  MD013: 行長制限（無効化）
```

______________________________________________________________________

## セクション4: 既存ツール整理ドキュメント

### 4.1 documents/tools/TOOLS_DIRECTORY.md の現状

**作成日:** 2026-03-19\
**状態:** 詳細整理済み（散在 → 構造化進行中）

**構成:**

1. ツール概説（20個のスクリプト分類）
1. 8つのツール分類セクション
1. ツール依存グラフ
1. フロー別ガイド（5フロー）
1. 整理方針・推奨変更（優先度3段階）
1. 次ステップ（5段階）

**記載範囲:**

- ✅ 各ツールの用途・実行方法・前提条件
- ✅ 副作用・エラー処理
- ✅ 関連ドキュメント参照
- ✅ 使用フロー別チェックリスト
- ⚠️ 優先度付き改善案不足（レビュー待ち）

**既知問題:**

1. `setup_worktree.sh` と `create_worktree.sh` の重複 → 統一推奨
1. `view_conventions.sh` / `read_conventions.sh` の役割曖昧
1. CI 統合ドキュメント欠落

______________________________________________________________________

### 4.2 documents/FILE_CHECKLIST_OPERATIONS.md の現状

**作成日:** 2026-03-19\
**状態:** チェックリスト体系完成

**構成:** 8個のチェックリスト

| #   | チェックリスト                   | 項目数    | 所要時間 | 対象       |
| --- | -------------------------------- | --------- | -------- | ---------- |
| 1   | 新規開発ブランチ開始             | 3ステップ | 15分     | 開発者     |
| 2   | Python パッケージ作成            | 3ステップ | 5分      | 開発者     |
| 3   | コード実装＆テスト               | 3ステップ | 15-30分  | 開発者     |
| 4   | ドキュメント更新                 | 4ステップ | 20-50分  | 開発者     |
| 5   | 設計ドキュメント整理             | 4ステップ | 30-60分  | メンテナー |
| 6   | ワークツリー完了＆クリーンアップ | 3ステップ | 10分     | 開発者     |
| 7   | ワークツリー規約遵守チェック     | 2ステップ | 10-30分  | 管理者     |
| 8   | ドキュメント品質チェック         | 3ステップ | 10分     | 管理者     |

**内容:**

- 全セクションに「確認項目」☑️ チェックボックス付き
- 実装コマンド例・説明 完備
- トラブルシューティング セクション付き

**特徴:**

- ✅ 粒度・フロー別分類が明確
- ✅ 所要時間が記載（計画立案用）
- ✅ 参考資料リンク完備
- ✅ 管理者向け 2 チェックリスト（規約検査・品質チェック）

______________________________________________________________________

### 4.3 documents/WORKFLOW_INVENTORY.md の現状

**作成日:** 2026-03-19\
**状態:** 自動化現状＆未自動化タスク整理

**自動化済みワークフロー（先行実装）:**

1. 静的チェック実行（`scripts/ci/run_static_checks.sh`）
1. 安全ファイル抽出（`scripts/ci/safe_file_extractor.py`）
1. 安全ファイル修正（`scripts/ci/safe_fix.sh`）
1. レポート収集（`scripts/ci/collect_reports.sh`）
1. 設計ファイル整理（`scripts/tools/organize_designs.py` 等）

**未自動化/部分自動化タスク:**

1. ブランチスコープの PR 警告（未実装）
1. ワークツリー `WORKTREE_SCOPE.md` チェック（有：`check_worktree_scopes.sh`、部分実装）
1. 設計ファイル類似度検出（有：`find_similar_designs.py`）
1. 設計移行の dry-run → PR 自動化（未実装）
1. 設計インデックス自動生成（未実装）

**優先度提案:**

| 優先度 | 項目                                       | 理由                                    |
| ------ | ------------------------------------------ | --------------------------------------- |
| 高     | ブランチスコープ PR 警告                   | 1 ブランチ = 1 サブモジュール規約の補強 |
| 高     | ワークツリー scope チェック + オーナー通知 | 規約遵守の自動化                        |
| 中     | 類似度検出ツール（完全版）                 | 重複排除の効率化                        |
| 中     | 設計移行 dry-run → PR 自動化               | フロー統合                              |
| 低     | インデックス自動生成                       | cron で十分                             |

______________________________________________________________________

## セクション5: 規約体系（コーディング規約・運用規約）

### 5.1 主要規約ドキュメント一覧

| ファイル                                  | 内容                             | 行数 | 対象           |
| ----------------------------------------- | -------------------------------- | ---- | -------------- |
| `documents/README.md`                     | ドキュメント体系全体             | TBD  | 全員           |
| `documents/coding-conventions-project.md` | プロジェクト全体運用規約         | 150+ | 全員           |
| `documents/WORKTREE_SCOPE_TEMPLATE.md`    | ワークツリースコープテンプレート | 40+  | 開発者・管理者 |
| `documents/BRANCH_SCOPE.md`               | ブランチスコープ定義             | 10+  | 設計書         |
| `documents/worktree-lifecycle.md`         | ワークツリー管理規約             | TBD  | 開発者・管理者 |
| `documents/AGENTS_COORDINATION.md`        | エージェント運用指針             | TBD  | 自動化         |
| `documents/coding-conventions-testing.md` | テスト規約                       | TBD  | テスト書者     |
| `documents/coding-conventions-logging.md` | ログ規約                         | TBD  | 実装者         |
| `documents/coding-conventions-reviews.md` | レビュー規約                     | TBD  | レビュアー     |
| `documents/conventions/python/`           | Python 規約（細則）              | 複数 | Python 開発者  |
| `documents/conventions/common/`           | 共通規約                         | 複数 | 全員           |

### 5.2 coding-conventions-project.md の主要内容

```
セクション:

1. 対象: Docker 環境・ドキュメント運用・サブモジュール位置づけ

2. 文書構造
   - documents/README.md が入口
   - base と安定サブモジュール API は design/apis/ に集約
   - レビュー報告は reviews/ に、メモは notes/, ログは diary/

3. Docker 環境方針
   - Dockerfile + requirements.txt を同期更新
   - 仮想環境使用禁止（venv/conda 対象外）
   - PYTHONPATH: ./python 基本、import: jax_util.*

4. サブモジュール位置づけ（7つ）
   - base, solvers, optimizers, hlo: 安定版
   - neuralnetwork: 実験段階
   - solvers/archive: 保管領域
   - tests, scripts, experiments: インフラ

5. 共通ルール
   - Scalar/Vector/Matrix 優先
   - Matrix: (n, batch) 原則
   - 線形作用素: @ (適用), * (合成)
   - 反復: jax.lax.scan/while_loop/fori_loop
   - ログ: DEBUG ガード内のみ

6. ドキュメント更新ルール
   - 実装変更 = 文書同時更新
   - 新規規約: conventions/ に追記
   - 正本は 1 つに定義（重複禁止）

7. タスク運用
   - task.md で進捗管理（粒度: ファイル単位）

8. ブランチ運用（重要）
   - 1 ブランチ = 1 サブモジュール（原則）
   - worktree = WORKTREE_SCOPE.md 必須
   - 実験結果 = results/<topic> ブランチに分離
   - Markdown: 相対パスのみ（絶対パス禁止）
```

______________________________________________________________________

## セクション6: 配置の問題点と散在状況

### 6.1 完全な問題点分析

| 問題                                   | 重大度 | 場所                                                              | 詳細                                                   | 影響範囲                   | 推奨対応                                      |
| -------------------------------------- | ------ | ----------------------------------------------------------------- | ------------------------------------------------------ | -------------------------- | --------------------------------------------- |
| **ワークツリー作成ツール重複**         | 🔴 高  | `scripts/setup_worktree.sh` vs `scripts/tools/create_worktree.sh` | 機能 90% 重複、引数体系が異なる                        | 開発フロー                 | 統一選択（推奨: `create_worktree.sh` 標準化） |
| **CI パイプライン不完全**              | 🔴 高  | `.github/workflows/ci.yml`                                        | `experiment_runner` テストのみ、`pyright`/`ruff` 未含  | CI 検証精度                | 全テスト + 静的解析を CI ターゲットに         |
| **エージェントワークフロー未実装**     | 🟡 中  | `.github/workflows/agent-coordination.yml`                        | テンプレート骨子状態、実処理欠落                       | 自動化効率                 | 実装完了（別タスク）                          |
| **スクリプト位置づけ混在**             | 🟡 中  | `scripts/` / `scripts/tools/`                                     | 「直下」と「tools/」の役割区分が曖昧                   | 新規スクリプト追加時の判断 | ディレクトリ運用指針明文化                    |
| **Makefile ターゲット最小限**          | 🟡 中  | `Makefile`                                                        | 1 ターゲット（git_init）のみ                           | 開発効率                   | テスト・ドキュメント生成等ターゲット追加      |
| **ドキュメント処理ツールと CI 未統合** | 🟡 中  | `scripts/tools/*.py` vs CI                                        | リンク監査・Markdown 整形が CI に含まれない            | ドキュメント品質           | CI に ドキュメント検査ステップ追加            |
| **設計ファイル整理フロー複雑**         | 🟡 中  | `organize_designs.py`, `find_*` 一族                              | 5 個のツール分散、統合 CLI 欠落                        | ツール使いやすさ           | ワンステップ整理コマンド（wrapper）           |
| **実験系スクリプト配置未定**           | 🟡 中  | `scripts/hlo/`                                                    | 1 個のみ、`experiments/` スクリプトがない              | 実験フロー                 | `scripts/experiments/` ディレクトリ検討       |
| **設定ファイル分散**                   | 🟢 低  | `.markdownlint.json`, `pyrightconfig.json` 等                     | Dockerfile+requirements.txt 需同期がドキュメント化欠落 | メンテナンス               | 設定更新時の SOP 作成                         |

### 6.2 スクリプト散在の具体例

```
ツール機能別分類ビュー（散在度分析）:

【ワークツリー管理】
  ✓ create_worktree.sh               (scripts/tools/)  ⭐ 推奨
  ✓ setup_worktree.sh                (scripts/)        ⚠️ 重複
  ✓ check_worktree_scopes.sh          (scripts/tools/)
  → 統一度: 低 / 分散度: 高 / 場所: 2 ディレクトリ

【ドキュメント処理】
  ✓ format_markdown.py               (scripts/tools/)
  ✓ fix_markdown_docs.py             (scripts/tools/)
  ✓ audit_and_fix_links.py           (scripts/tools/)
  → 統一度: 中 / 分散度: 低 / 場所: 1 ディレクトリ
  ⚠️ CI との未統合

【設計ファイル整理】
  ✓ organize_designs.py              (scripts/tools/)
  ✓ create_design_template.py        (scripts/tools/)
  ✓ find_similar_designs.py          (scripts/tools/)
  ✓ find_similar_documents.py        (scripts/tools/)
  ✓ tfidf_similar_docs.py            (scripts/tools/)
  ✓ find_redundant_designs.py        (scripts/tools/)
  → 統一度: 低 / 分散度: 高 / 場所: 1 ディレクトリ
  ⚠️ フロー統合なし（並列実行必須）

【Git・ブランチ】
  ✓ git_config.sh                    (scripts/)
  ✓ git_init.sh                      (scripts/)
  ✓ git_repo_init.sh                 (scripts/)
  → 統一度: 高 / 分散度: 低 / 場所: 1 ディレクトリ

【ユーティリティ・情報表示】
  ✓ view_conventions.sh              (scripts/)
  ✓ read_conventions.sh              (scripts/)
  ✓ guide.sh                         (scripts/)
  ✓ extract_deps_from_svg.sh         (scripts/)
  → 統一度: 低 / 分散度: 中 / 役割: 曖昧

【テスト・ログ】
  ✓ run_pytest_with_logs.sh          (scripts/)
  → 統一度: N/A / 分散度: N/A （唯一、独立）

【HLO 分析】
  ✓ summarize_hlo_jsonl.py           (scripts/hlo/)
  → 統一度: N/A / 分散度: N/A （独立ディレクトリ）
```

______________________________________________________________________

## セクション7: 業界標準パターン（簡潔版）

### 7.1 大型 Python プロジェクトのスクリプト配置慣例

**標準的なパターン：**

```
scripts/
├── README.md                    ← スクリプト使用ガイド
├── setup/                       ← 初期化・セットアップ
│   ├── init.sh                  (環境初期化)
│   └── bootstrap.sh             (依存インストール)
├── dev/                         ← 開発補助 (実装・テスト・デバッグ)
│   ├── test.sh
│   ├── lint.sh
│   └── format.sh
├── ci/                          ← CI スクリプト（ローカルでも実行可）
│   ├── run_checks.sh
│   ├── run_tests.sh
│   └── collect_reports.sh
├── tools/                       ← 補助ツール・メンテナンス
│   ├── analyze_deps.py
│   ├── organize_structures.py
│   └── generate_docs.py
└── experiments/                 ← 実験・分析スクリプト
    ├── run_experiment.sh
    └── analyze_results.py
```

**参考実例等:**

- JAX, TensorFlow: `tools/` に分析ツール、`scripts/` に実行スクリプト
- PyTorch: `tools/` に検証ツール、`setup.py` 中心
- scikit-learn: `maint_tools/` に管理ツール集合、構造化

**本プロジェクトの比較:**

- ✅ 長所: ツール用 `scripts/tools/` ディレクトリ分離（標準パターン従従）
- ⚠️ 短所: 直下に「混在」（setup + ユーティリティ）
- ⚠️ 短所: `ci/` ディレクトリがなく、CI スクリプト未統合

______________________________________________________________________

### 7.2 CI/CD ワークフロー整理の標準パターン

**業界標準の 3 層構造:**

```
Layer 1: ローカル開発（developers）
  ├─ make test          （CLI 実行）
  ├─ make lint
  ├─ make format
  └─ scripts/dev/なんか.sh

Layer 2: ローカル CI シミュレー（CI 担当者）
  ├─ scripts/ci/run_checks.sh   ←← CI と同じ処理の再現
  └─ scripts/ci/run_tests.sh

Layer 3: GitHub Actions（リモート自動実行）
  ├─ .github/workflows/ci.yml
  │   └─ scripts/ci/run_checks.sh を呼び出し（同じ処理）
  └─ .github/workflows/pr-checks.yml
      └─ 段階的テスト
```

**メリット:**

1. ローカル ↔ リモート の一貫性
1. CI スクリプト自体がバージョン管理される
1. 新 CI スクリプト追加時にローカルで検証可能

**参考事例:**

- Django: `.github/workflows/` から `scripts/runtests.py` を call
- CPython: Workflows から `Tools/` スクリプトを execution
- Kubernetes: `.github/workflows/` + `hack/` スクリプト統合

**本プロジェクトの比較:**

- ✅ 長所: `scripts/tools/` にドキュメント処理ツール集約
- ❌ 短所: `scripts/ci/` がなく、CI スクリプト未バージョン管理
- ❌ 短所: GitHub Actions から `scripts/` を呼び出していない（直接実行）

______________________________________________________________________

### 7.3 相対パス参照構造の推奨形式

**業界標準（絶対評価）:**

| パターン                        | 形式                            | 用途                              | 例                        |
| ------------------------------- | ------------------------------- | --------------------------------- | ------------------------- |
| **リポジトリ内（相対）**        | `../../path/to/file`            | Markdown, Config, Script 相互参照 | ✅ 推奨                   |
| **リポジトリ内（ルート相対）**  | `path/to/file`                  | CI, Docker 内実行                 | ✅ 推奨（context 明確時） |
| **リポジトリ外（絶対）**        | `/home/user/...`                | 避ける（環境依存）                | ❌ 禁止                   |
| **リポジトリ外（\$HOME, env）** | `$HOME/path`, `$WORKSPACE/path` | 環境変数経由は OK                 | ⚠️ 条件付き               |

**本プロジェクトのパターン:**

```bash
# ✅ 推奨: 相対パス
cd /workspace/scripts
bash tools/create_worktree.sh my-feature

# ✅ 推奨: ルート相対 (PYTHONPATH)
PYTHONPATH=./python python -m pytest python/tests/

# ⚠️ 現在の実装: 絶対パス混在
PYTHONPATH="/workspace/python"  # ← Dockerfile に記載
LOG_ROOT="/workspace/python/tests/logs"  # ← run_pytest_with_logs.sh

# 改善案:
PYTHONPATH="./python"  # ← reporoot からの相対
LOG_ROOT="./python/tests/logs"  # ← 同様
```

**Markdown リンク推奨:**

```markdown
✅ 推奨:
[ドキュメント](../../documents/README.md)
[API仕様](../design/apis/solvers.md)

❌ 非推奨:
[ドキュメント](/workspace/documents/README.md)
[API仕様](file:///home/user/workspace/documents/design/apis/solvers.md)
```

______________________________________________________________________

## セクション8: 推奨されるディレクトリ構成案

### 8.1 推奨構造（現在 → 改善後）

```diff

現在の構造:
─────────────────────────────────────────────────
scripts/
├── .gitkeep
├── create_toml.sh
├── extract_deps_from_svg.sh
├── git_config.sh
├── git_init.sh
├── git_repo_init.sh
├── guide.sh
├── jsonl_to_md.sh
├── read_conventions.sh
├── run_pytest_with_logs.sh
├── setup_worktree.sh              ⚠️ 重複
├── view_conventions.sh
├── hlo/
│   └── summarize_hlo_jsonl.py
└── tools/                         ← ツール集約
    ├── 11個ファイル


推奨構造（段階改善案 Phase 1）:
─────────────────────────────────────────────────
scripts/
├── README.md                      ← 🆕 スクリプト使用ガイド
├── setup/                         ← 🆕 初期化スクリプト
│   ├── git_config.sh
│   ├── git_init.sh
│   ├── git_repo_init.sh
│   └── create_toml.sh
├── dev/                           ← 🆕 開発補助スクリプト
│   ├── run_tests.sh               (run_pytest_with_logs.sh → rename)
│   ├── check_formats.sh           🆕
│   └── check_worktrees.sh         (tools/check_worktree_scopes.sh → move)
├── ci/                            ← 🆕 CI 統合スクリプト
│   ├── run_static_checks.sh       🆕
│   ├── run_all_checks.sh          🆕
│   └── collect_reports.sh         🆕
├── guide/                         ← 🆕 情報表示
│   ├── conventions.sh             (view_conventions.sh, read_conventions.sh → merge)
│   └── guide.sh
├── tools/                         ← 現在のディレクトリ（機能拡張）
│   ├── README.md                  ← 🆕 ツール・リポジトリ使用ガイド
│   ├── worktree/
│   │   ├── create_worktree.sh     (tools/ から move → コアツール)
│   │   └── list_worktrees.sh      🆕
│   ├── docs/
│   │   ├── format_markdown.py
│   │   ├── audit_links.py
│   │   ├── fix_docs.py
│   │   └── README.md              ← 🆕 ドキュメント ツール使用例
│   ├── design/
│   │   ├── organize_designs.py
│   │   ├── create_template.py
│   │   ├── find_similar.py
│   │   ├── find_redundant.py
│   │   ├── tfidf_similarity.py
│   │   └── README.md              ← 🆕 使用順序ガイド
│   └── analysis/
│       ├── extract_deps.sh
│       └── summarize_hlo.py
└── experiments/                   ← 🆕 実験スクリプト（今後）
    └── README.md


推奨構造（段階改善案 Phase 2）:
─────────────────────────────────────────────────
上記 Phase 1 + 以下:

scripts/
├── ci/
│   └── Makefile                   🆕 CI ターゲット定義
├── integration/                   🆕 複合フロー
│   ├── full_validation.sh
│   ├── clean_designs.sh
│   └── prepare_release.sh
└── docs/                          🆕 ドキュメント生成・検査
    ├── build_api_docs.sh
    └── check_consistency.sh

プロジェクト root Makefile 拡張:
Make targets:
  make setup              ← scripts/setup/*.sh
  make test               ← scripts/dev/run_tests.sh
  make lint               ← scripts/dev/check_formats.sh  
  make ci                 ← scripts/ci/run_all_checks.sh
  make clean-design       ← scripts/integration/clean_designs.sh
  make validate           ← scripts/integration/full_validation.sh
```

### 8.2 具体的な改善アクション

| Phase | 優先度 | 行動                                                                      | 所要工数      | 効果                    |
| ----- | ------ | ------------------------------------------------------------------------- | ------------- | ----------------------- |
| 1-1   | 🔴 高  | `setup_worktree.sh` を廃止 → `create_worktree.sh` を標準化                | 5min + テスト | ツール統一              |
| 1-2   | 🔴 高  | `scripts/ci/` ディレクトリ作成、CI スクリプト集約                         | 30分          | ローカル ↔ リモート同期 |
| 1-3   | 🔴 高  | `scripts/setup/` ディレクトリ作成、初期化スクリプト整理                   | 15分          | 関心ごとの分離          |
| 1-4   | 🟡 中  | `scripts/tools/` に README.md 追加（使用順序ガイド）                      | 20分          | ツール発見性向上        |
| 1-5   | 🟡 中  | `.github/workflows/ci.yml` を scripts/ci/ の実行に切り替え                | 15分          | ローカル再現性向上      |
| 2-1   | 🟡 中  | `scripts/guide/` → `view_conventions.sh` と `read_conventions.sh` を 統合 | 20分          | UI 統一                 |
| 2-2   | 🟡 中  | `scripts/tools/design/` 統合 wrapper 作成                                 | 30分          | ワンステップ整理        |
| 2-3   | 🟢 低  | `scripts/experiments/` ディレクトリ作成、実験スクリプト予約               | 10分          | 将来展開対応            |

### 8.3 Phase 1 実装ロードマップ（1-2 週間見積）

```
Week 1:
  [Day 1] 1-1 setup_worktree 統一（廃止 + テスト）
  [Day 2] 1-2 scripts/ci/ ディレクトリと CI スクリプト集約
  [Day 3] 1-3 scripts/setup/ ディレクトリ作成
  [Day 4] 1-4 ツール README 作成（scripts/tools/ + 各 subdir）

Week 2 (前半):
  [Day 5] 1-5 CI ワークフロー修正（scripts/ci/ 呼び出し）
          + Makefile ターゲット追加
  [Day 6] レビュー・テスト全体
  [Day 7] 本番反映への準備

Optional (Phase 2):
  + 2-1 conventions 統合
  + 2-2 design 統合 wrapper
  + 2-3 experiments 予約
```

______________________________________________________________________

## セクション9: サマリー＆推奨実装順序

### 9.1 本プロジェクトの総合評価

| 観点                   | 評価 | 理由                                                       |
| ---------------------- | ---- | ---------------------------------------------------------- |
| **スクリプト整理度**   | 7/10 | ツール集約は良好。重複・散在が残存                         |
| **CI/CD 整備度**       | 5/10 | 基本 CI は存在。全テスト + ドキュメント検査が欠落          |
| **ドキュメント充実度** | 9/10 | 規約・チェックリスト・ツール目録が詳細。実装指示明確       |
| **運用規約体系**       | 8/10 | WORKTREE_SCOPE, BRANCH_SCOPE, コンベンション整備。但し分散 |
| **相対パス参照度**     | 6/10 | Markdown は相対パス。Shell スクリプトは絶対パス混在        |
| **業界標準準拠度**     | 6/10 | 理想型への 60% 達成。Phase 1 改善で 80% へ                 |

### 9.2 ブロッカーと推奨優先順序

```
即実施（ブロッカー）:
  [1] setup_worktree.sh 廃止 → create_worktree.sh 統一
      理由: 開発者混乱、ドキュメント矛盾
      コスト: 5分 + テスト
      効果: ツール統一、心理的負荷軽減

短期（1-2 週間）:
  [2] scripts/ci/ ディレクトリ作成 + CI スクリプト統合
      理由: ローカル ↔ リモート同期、保守効率向上
      コスト: 30分
      効果: 高（ CI 再現性向上）

  [3] Makefile ターゲット追加（test, lint, ci）
      理由: 開発 UX 向上、フロー簡素化
      コスト: 15分
      効果: 中（開発効率 ↑）

中期（1 ヶ月）:
  [4] scripts/tools/ README.md + 各サブディレクトリの整理
      理由: ツール発見性・使い込みやすさ向上
      コスト: 60分
      効果: 中（新規ユーザーオンボーディング短縮）

  [5] CI に ドキュメント検査ステップ（lint, link audit）
      理由: ドキュメント品質保証の自動化
      コスト: 30分
      効果: 中（ドキュメント品質向上）

長期/オプション:
  [6] 設計ファイル整理統合 wrapper 作成
  [7] GitHub Actions agent-coordination.yml 完全実装
  [8] experiments スクリプト構成体系化
```

### 9.3 実装済み・要改善・未実装の整理表

| 機能                 | 状態          | 改善予定                        | 優先度 |
| -------------------- | ------------- | ------------------------------- | ------ |
| Git 初期化スクリプト | ✅ 完成       | —                               | —      |
| ワークツリー作成     | ⚠️ 重複       | 統一（create_worktree.sh 標準） | 🔴 高  |
| チェックリスト体系   | ✅ 完成       | —                               | —      |
| 規約ドキュメント     | ✅ 完成       | 散在解消（リンク整理）          | 🟡 中  |
| CI 基本テスト        | ✅ 存在       | 全テスト + 静的解析追加         | 🔴 高  |
| CI ドキュメント検査  | ❌ なし       | 実装（audit_links + format）    | 🟡 中  |
| Makefile             | ✅ 最小限     | ターゲット追加                  | 🟡 中  |
| スクリプト README    | ❌ なし       | 作成（全ディレクトリ）          | 🟡 中  |
| ツール統合 CLI       | ❌ なし       | 設計整理 wrapper 追加           | 🟢 低  |
| エージェント自動化   | ⚠️ スケルトン | 完全実装                        | 🟢 低  |
| 相対パス統一化       | ⚠️ 混在       | Bash スクリプト修正             | 🟡 中  |

______________________________________________________________________

## 附録A: ファイル・スクリプト参照一覧（全 23個）

### A.1 shells スクリプト（Bash, 14個）

1. [scripts/git_config.sh](scripts/git_config.sh)
1. [scripts/git_init.sh](scripts/git_init.sh)
1. [scripts/git_repo_init.sh](scripts/git_repo_init.sh)
1. [scripts/view_conventions.sh](scripts/view_conventions.sh)
1. [scripts/read_conventions.sh](scripts/read_conventions.sh)
1. [scripts/setup_worktree.sh](scripts/setup_worktree.sh) ⚠️ 廃止予定
1. [scripts/guide.sh](scripts/guide.sh)
1. [scripts/create_toml.sh](scripts/create_toml.sh)
1. [scripts/jsonl_to_md.sh](scripts/jsonl_to_md.sh)
1. [scripts/extract_deps_from_svg.sh](scripts/extract_deps_from_svg.sh)
1. [scripts/run_pytest_with_logs.sh](scripts/run_pytest_with_logs.sh)
1. [scripts/tools/create_worktree.sh](scripts/tools/create_worktree.sh) ⭐ 推奨
1. [scripts/tools/check_worktree_scopes.sh](scripts/tools/check_worktree_scopes.sh)

### A.2 Python スクリプト（9個）

1. [scripts/tools/format_markdown.py](scripts/tools/format_markdown.py)
1. [scripts/tools/fix_markdown_docs.py](scripts/tools/fix_markdown_docs.py)
1. [scripts/tools/audit_and_fix_links.py](scripts/tools/audit_and_fix_links.py)
1. [scripts/tools/organize_designs.py](scripts/tools/organize_designs.py)
1. [scripts/tools/create_design_template.py](scripts/tools/create_design_template.py)
1. [scripts/tools/find_similar_designs.py](scripts/tools/find_similar_designs.py)
1. [scripts/tools/find_similar_documents.py](scripts/tools/find_similar_documents.py)
1. [scripts/tools/tfidf_similar_docs.py](scripts/tools/tfidf_similar_docs.py)
1. [scripts/tools/find_redundant_designs.py](scripts/tools/find_redundant_designs.py)

### A.3 HLO (1個)

1. [scripts/hlo/summarize_hlo_jsonl.py](scripts/hlo/summarize_hlo_jsonl.py)

______________________________________________________________________

## 附録B: 設定ファイル参考情報

### B.1 Docker Dockerfile インストールモジュール完全リスト

```dockerfile
System:
  ca-certificates curl git
  build-essential pkg-config cmake
  python3 python3-pip python3-venv
  openssh-client graphviz

Python packages:
  jax[cuda12]
  jaxtyping typeguard equinox
  pytest scipy scipy-stubs
  optax
  pyright pydeps
```

### B.2 pyproject.toml セッティング

```toml
Type checking mode: strict
Include paths: ./python/jax_util (+ exclude: archive, tests, build, dist)
Stub path: ./python/typings
Report settings: Unknown types suppressed (外部ライブラリ配慮)

Ruff:
  Line length: 100
  Target: Python 3.10+
  Lint select: E (errors), F (flake8), I (isort)
```

### B.3 CI ワークフロー設定概要

```yaml
ci.yml:
  Trigger: push/PR on main, master
  Python: 3.10
  Matrix: 1 version
  Key steps: checkout, setup-python, install-deps, pytest

agent-coordination.yml:
  Trigger: workflow_dispatch
  Jobs: coordinator → reviewer → integrator (pipeline)
  State: Skeleton (未完全)
```

______________________________________________________________________

## 附録C: 互換性と相互依存グラフ

### C.1 スクリプト依存グラフ（簡略版）

```
┌─ git_config.sh
│
├─ git_init.sh ──┬─→ [プロジェクト初期化]
│                │
│                ├─→ setup_worktree.sh ⚠️

├─ create_worktree.sh ⭐ [標準ワークツリー作成]
│   └─→ WORKTREE_SCOPE_TEMPLATE.md

├─ [ドキュメント処理]
│   ├─→ format_markdown.py
│   ├─→ fix_markdown_docs.py
│   └─→ audit_and_fix_links.py

├─ [設計ファイル整理] ※順序推奨
│   ├─→ organize_designs.py
│   ├─→ find_similar_designs.py
│   ├─→ find_redundant_designs.py
│   └─→ tfidf_similar_docs.py

├─ run_pytest_with_logs.sh ──→ python/tests/logs/

└─ [情報表示・ユーティリティ]
    ├─→ view_conventions.sh
    ├─→ guide.sh
    └─→ extract_deps_from_svg.sh
```

______________________________________________________________________

## 結論

本プロジェクトは以下の強みを持ちます：

✅ **ドキュメント整備:**

- 規約・チェックリスト・ツール目録が詳細かつ実用的
- コーディング規約が厳密で一貫性がある

✅ **ツール群の充実:**

- 23個のスクリプト・ツールが、操作性・保守性を考慮して配置
- Python + Bash の混合利用で、タスク適性に合わせた実装

✅ **運用体系の完成度:**

- WORKTREE_SCOPE による スコープ管理
- チェックリスト形式の段階的手順書

**改善余地:**

⚠️ **ツール重複:**

- `setup_worktree.sh` と `create_worktree.sh` 統一が喫緊

⚠️ **CI/CD 未完全:**

- ドキュメント検査が CI に含まれない
- Agent Coordination が未実装

⚠️ **相対パス混在:**

- Bash スクリプトに絶対パスが散在

**推奨 Phase 1 実装項目（優先度順）:**

1. ワークツリーツール統一（5分）
1. scripts/ci/ 作成 + CI 統合（30分）
1. Makefile ターゲット追加（15分）
1. ツール README 作成（60分）
1. CI ドキュメント検査追加（30分）

**期待効果:**

- 開発フロー統一化 → 心理的負荷軽減
- ローカル ↔ リモート CI 同期 → 再現性向上
- ツール発見性向上 → 新規ユーザーオンボーディング短縮
- 業界標準に接近 → 80% 準拠まで改善

______________________________________________________________________

**レポート作成者:** GitHub Copilot\
**作成日:** 2026-03-19\
**対象期間:** プロジェクト全期間\
**次回レビュー日:** 2026-04-02（実装後 2 週間）
