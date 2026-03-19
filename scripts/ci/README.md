# scripts/ci — CI スクリプト・ローカル実行ガイド

このディレクトリは、GitHub Actions で実行する CI チェックをローカルで実行できるスクリプトを置きます。

**目的:**

- 開発中に CI が失敗することを防ぐ
- リモート CI 実行前にバグを検出
- CI と開発環境の一貫性確保

______________________________________________________________________

## スクリプト一覧

### run_all_checks.sh（★推奨: 統合テスト）

**用途:** pytest + pyright + ruff を一括実行

**実行方法:**

```bash
# 標準実行（全チェック）
bash scripts/ci/run_all_checks.sh

# 高速モード（ruff をスキップ）
bash scripts/ci/run_all_checks.sh --quick

# 詳細出力
bash scripts/ci/run_all_checks.sh --verbose
```yaml

**実行内容:**

1. `pytest python/tests/` — ユニット・統合テスト
1. `pyright python/` — Python 型チェック
1. `ruff check python/` — リント・スタイルチェック

**戻り値:**

- `0`: すべて成功
- `1`: テスト失敗 または致命的エラー

**所要時間:** 30秒～2分（マシン依存）

______________________________________________________________________

## 開発ワークフロー

### 推奨される作業フロー

```bash
# 1. コード編集
vim python/jax_util/module/file.py

# 2. ローカルで CI チェック（失敗時は修正）
bash scripts/ci/run_all_checks.sh

# 3. 成功後、コミット
git add -A
git commit -m "feat: new feature"

# 4. リモート push
git push origin branch-name
```text

## GitHub Actions CI との関係

`.github/workflows/ci.yml` はこのスクリプトを呼び出します：

```yaml
- name: Run all checks
  run: bash scripts/ci/run_all_checks.sh
```yaml

**つまり:** ローカルで成功 = リモート CI でも成功（高確率）

______________________________________________________________________

## トラブルシューティング

### pytest 失敗時

```bash
# 失敗したテスト単体を再実行
pytest python/tests/test_module.py::TestClass::test_method -v

# 失敗情報を詳しく表示
pytest python/tests/ -v --tb=long
```yaml

詳細: [../../documents/FILE_CHECKLIST_OPERATIONS.md#チェックリスト3](../../documents/FILE_CHECKLIST_OPERATIONS.md)

## pyright 警告が多い場合

```bash
# 警告ファイルをフィルタ表示
pyright python/ 2>&1 | grep "error"

# 特定モジュールのみチェック
pyright python/jax_util/solvers/
```text

## ruff エラー修正

```bash
# 自動修正
ruff check --fix python/

# 修正内容確認
git diff python/
```text

## 依存パッケージが不足の場合

```bash
# 環境確認
python -c "import pytest, pyright, ruff; print('OK')"

# インストール
pip install -r docker/requirements.txt

# または Docker 使用
docker build -t jax-util -f docker/Dockerfile .
docker run --rm -v $(pwd):/workspace -w /workspace jax-util bash scripts/ci/run_all_checks.sh
```text

______________________________________________________________________

## 環境変数・カスタマイズ

### PYTHONPATH（自動設定）

スクリプスは自動的に以下を設定します：

```bash
export PYTHONPATH="/workspace/python:${PYTHONPATH:-}"
```yaml

参照: [documents/coding-conventions-project.md](../../documents/coding-conventions-project.md)

### ログ出力（未実装）

将来は以下のようにログ保存にも対応予定：

```bash
bash scripts/ci/run_all_checks.sh 2>&1 | tee logs/ci_$(date +%Y%m%d_%H%M%S).txt
```yaml

______________________________________________________________________

## 参考リンク

- [../../README.md](../../README.md) — プロジェクト概要
- [../README.md](../README.md) — スクリプト全体
- [../../documents/FILE_CHECKLIST_OPERATIONS.md](../../documents/FILE_CHECKLIST_OPERATIONS.md) — 作業別チェックリスト
- [../../.github/workflows/ci.yml](../../.github/workflows/ci.yml) — GitHub Actions ワークフロー

______________________________________________________________________

**最終更新:** 2026-03-19
