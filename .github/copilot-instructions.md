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
