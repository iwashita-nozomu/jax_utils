# Skill 8: Experiment Initialization

## 概要

新規実験・プロジェクトの初期化と セットアップを自動化するスキル。

実験用ディレクトリ構造の作成、config ファイルの生成、
必要な依存関係の確認、テンプレートファイルの展開を行います。

## 主要機能

- **ディレクトリ構造生成**: 标準実験フォルダツリーを作成
- **Config テンプレート**: experiment.yaml・config.json を自動生成
- **依存関係チェック**: JAX・NumPy・その他の必須パッケージ確認
- **テンプレートファイル**: main.py・README を展開

## 依存関係

- Skill 1: Static Check

## 実行例

```bash
python3 .github/skills/08-experiment-initialization/run-init.py --exp-name my_experiment --config template
```

## パラメータ

- `--exp-name STRING` — 実験プロジェクト名（必須）
- `--config {template,full,minimal}` — テンプレートレベル（デフォルト: template）
- `--output DIRECTORY` — 出力ディレクトリ（デフォルト: ./experiments）
