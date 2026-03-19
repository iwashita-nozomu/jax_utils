# エージェントチーム運用と役割分担

目的: 設計ファイル整理とテンプレート作成を安全かつ再現可能に実行するためのエージェントチーム編成。

役割:

- Coordinator: タスクを割り振り、実行ログを集める。人間とのエスカレーション担当。
- Mover Agent: `scripts/tools/organize_designs.py` を実行し、設計ファイルをサブモジュール別に整理する。
- Template Agent: `scripts/tools/create_design_template.py` を用いて、サブモジュールごとのテンプレートを生成する。
- Reviewer Agent: 整理結果をレビューし、`documents/design/<submodule>/README.md` に変更要約を残す。

実行フロー（短期）:

1. Coordinator が `organize` タスクを起動（dry-run 推奨）。
1. Mover Agent が `organize_designs.py` を実行してコピーを作成し、`reports/design_organize_report.txt` を出力。
1. Template Agent が検出されたサブモジュールごとに `create_design_template.py` を実行して `template.md` を作る。
1. Reviewer Agent がコピー先を確認、不要なら差し戻し。
1. Coordinator がコミットと PR を作成（ドキュメントだけ main へ反映）。

安全ルール:

- 自動で file 移動はコピーのみ。オリジナル削除は人間の承認後に実行。
- すべての出力は `reports/` に保存する。
