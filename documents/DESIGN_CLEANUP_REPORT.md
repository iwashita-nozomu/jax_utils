# 設計文書クリーンアップ報告

目的: 冗長・重複している設計文書を整理し、サブモジュール単位で設計が辿りやすい状態にする。

実施内容（要約）:

- `documents/design/apis/` 配下の API 設計を `documents/design/<submodule>/` に移行（コピー）しました。
- 完全一致の重複ファイルを検出・削除しました（詳細は `reports/redundant_designs.txt` を参照）。
- 設計テンプレートを `documents/design/<submodule>/template.md` に生成しました。
- 自動化スクリプトを追加し（`scripts/tools/organize_designs.py`、`scripts/tools/create_design_template.py`、`scripts/tools/find_redundant_designs.py`）、レポートを `reports/` に保存しています。

保留／要レビュー:

- 内容が類似しているが完全一致ではないファイル（要マニュアルレビュー）。

ナビゲーション方針（3 参照以内ルール）:

1. `documents/design/` ルートに必ず `README.md`（インデックス）を置く。
1. サブモジュール毎の設計は `documents/design/<submodule>/README.md` に一覧を置く。
1. 個別設計へのリンクは `documents/design/<submodule>/README.md` から直接たどれるようにする。

作業ログ / 重要な結果:

- 生成レポート: `reports/design_organize_report.txt`
- 重複削除報告: `reports/redundant_designs.txt`

推奨アクション（優先度順）:

1. 類似ファイルの手動レビュー（ファイル毎に担当者を割り当てる）。
1. `documents/design/README.md` と各 `documents/design/<submodule>/README.md` を整備して 3 参照以内ルールを満たす。
1. `scripts/tools/organize_designs.py` を定期実行する CI（週次）を追加し、dr y-run → レポート → 人間レビューフローを確立する。
1. 古い設計ファイルの完全削除は、レビュー完了後に実施する（現在は削除済みの完全重複のみ実行）。
