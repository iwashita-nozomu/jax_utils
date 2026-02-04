# テストの書き方

## 要約
- テストは `python/tests/` に配置します。
- 詳細はテスト規約を参照します。

## 規約
- テストコードは `python/tests/` に配置します。
- 詳細は `../../coding-conventions-testing.md` を参照してください。
- `__main__` ブロックは **テスト側のみに限定**し、本体モジュールには置きません。
- 実行環境の設定は `python/jax_util/base/_env_value.py` に集約します。

## テスト分類
- `python/tests/base/`: 単体テスト（小規模・高速）。
- `python/tests/Algorithms/`: アルゴリズム系テスト（反復・数値解析）。
