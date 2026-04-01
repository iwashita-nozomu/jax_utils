# Skill: `static-check` — 基礎検証スキル

**目的**: Python/C++ コードの型注釈、Docstring、テスト、Docker 依存関係を検証

**対応ドキュメント**: 
- `documents/SKILL_IMPLEMENTATION_GUIDE.md` の A 層（基礎検証）
- `documents/coding-conventions-python.md`
- `documents/coding-conventions-testing.md`

---

## 概要

このスキルは、PR 作成前またはコード作成中に実行する自動チェック群です。以下を検証：

1. **Python 型チェック** (`pyright`) — 型注釈の正確性
2. **Python Lint** (`ruff`) — スタイル・危険なパターン
3. **テスト実行** (`pytest`) — 全テスト成功確認
4. **Docker 検証** (`check-docker.py`) — Dockerfile と requirements.txt の整合性
5. **Markdown フォーマット** (`mdformat`) — ドキュメント統一

---

## 使用方法

### CLI で実行

```bash
# フル検査（全チェック実行）
claude --skill static-check

# または
copilot --skill static-check

# 特定ファイルのみチェック
claude --skill static-check src/my_module.py

# JSON 出力（パース可能）
copilot --skill static-check --format json > results.json
```

### ローカルで直接実行

```bash
# Python チェック
bash .github/skills/01-static-check/check-python.sh

# テスト実行
bash .github/skills/01-static-check/check-tests.sh

# Docker 検証
python .github/skills/01-static-check/check-docker.py

# 全実行
bash .github/skills/01-static-check/run-all.sh
```

### GitHub Actions で自動実行

GitHub Actions パイプラインから自動的に実行されます（`.github/workflows/` を参照）

---

## スクリプト詳細

### 1. `check-python.sh` — Python 型・Lint チェック

```bash
bash .github/skills/01-static-check/check-python.sh [--verbose] [--report DIR]
```

実行内容：
- `pyright` — 型エラー検出
- `ruff check` — スタイル・危険パターン
- `black --check` — フォーマット確認

出力：
- stdout: チェック結果
- `--report DIR`: 結果を JSON で DIR に保存

---

### 2. `check-tests.sh` — テスト実行

```bash
bash .github/skills/01-static-check/check-tests.sh [--parallel] [--report DIR]
```

実行内容：
- `pytest` 全実行
- カバレッジ測定（オプション）

オプション：
- `--parallel`: 並列実行（デフォルト）
- `--report DIR`: 結果を JSON で保存

---

### 3. `check-docker.py` — Docker 依存関係検証

```bash
python .github/skills/01-static-check/check-docker.py [--verbose]
```

実行内容：
- Dockerfile モジュールスキャン
- `docker/requirements.txt` 解析
- インストール済みモジュール vs 参照モジュール 検証
- 不整合検出時は失敗コード 1 で終了

---

## チェックリスト

PR 作成前に以下を確認：

- [ ] `check-python.sh` が 0 で終了
- [ ] `check-tests.sh` が 0 で終了
- [ ] `check-docker.py` が 0 で終了
- [ ] ドキュメント変更時：`mdformat documents/` 実行
- [ ] Git status は clean （コミット前）

---

## よくある問題と対応

| 問題 | 原因 | 対応 |
|------|------|------|
| `pyright` エラー | 型注釈欠落 | 関数に `->` 型を追加 |
| `ruff` 警告 | スタイル不統一 | `ruff format` で修正 |
| テスト失敗 | 回帰 | テスト追加・修正 |
| Docker エラー | requirements.txt が古い | `pip freeze > docker/requirements.txt` |
| `mdformat` エラー | Markdown フォーマット不統一 | `mdformat documents/` 実行 |

---

## 実装状況

| 要素 | 状態 | 備考 |
|------|------|------|
| check-python.sh | ✅ 実装 | pyright, ruff, black 統合 |
| check-tests.sh | ✅ 実装 | pytest + coverage |
| check-docker.py | ✅ 実装 | 依存関係検証 |
| config.yaml | 🔄 開発中 | Skill 定義 |
| run-all.sh | 🔄 開発中 | 統合スクリプト |

---

## 参考資料

- **CLI リファレンス**: `documents/AI_AGENT_CUSTOMIZATION_RESEARCH_2026.md` - CLI エージェント・コマンドラインリファレンス
- **実装ガイド**: `documents/SKILL_IMPLEMENTATION_GUIDE.md`
- **統合計画**: `.github/SKILLS_INTEGRATION_PLAN.md`
