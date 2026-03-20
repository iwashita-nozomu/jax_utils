## PR 説明

何が変わりましたか？（1-2 行）

## 関連 Issue

closes #XXX

## 変更タイプ

- [ ] バグ修正
- [ ] 新機能
- [ ] ドキュメント更新
- [ ] リファクタリング
- [ ] テスト追加
- [ ] その他

## チェックリスト

### コード品質

- [ ] Python コード規約に従っている（`documents/coding-conventions-python.md`）
- [ ] 型注釈を追加した（関連ファイル）
- [ ] Docstring を追加した（モジュール / クラス / 関数）
- [ ] 複雑な分岐を避けシンプルに実装した
- [ ] `/python/Archive/*` には変更を加えていない

### テスト・検証

- [ ] ローカル CI 実行: `make ci`
- [ ] `pytest` が成功
- [ ] `pyright` -エラーなし
- [ ] `pydocstyle` - チェック済み
- [ ] `ruff check` (E, F, I, D, UP) - パスした

### ドキュメント

- [ ] 実装変更に対応する `documents/` ファイルを更新した
- [ ] リンク参照が正確（相対パス、3参照ルール内）
- [ ] Markdown ファイルを `mdformat` で整形した

### Docker 環境

- [ ] 仮想環境（`.venv` など）を作成していない
- [ ] 新規モジュール追加時、`Dockerfile` と `docker/requirements.txt` を同時更新した

### ワークツリー

- [ ] `WORKTREE_SCOPE.md` が設定されている（ワークツリー作業の場合）
- [ ] ブランチは指定scopeの範囲内

## レビュアーへのコメント

（特に注意が必要な箇所があればここに記述）

---

**CI チェック状態**: [![CI](https://github.com/org/jax-util/workflows/CI/badge.svg)](../../actions)
