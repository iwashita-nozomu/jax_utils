# ソルバー規約（共通）

この文書は、`python/jax_util/Algorithms/` 配下のソルバー系実装を対象にします。

## 1. 対象と目的
この文書は、`python/jax_util/Algorithms/` 配下のソルバー系実装を対象にします。

## 2. 戻り値の統一
- ソルバー系の戻り値は **`ans, state, info`** の順番に統一します。
- `state` を持たない場合でも、**空の構造体**を定義して返します。
- **API（関数名・引数・公開範囲）は現在の形を維持**し、戻り値の統一だけを行います。

## 3. `state` の方針
- `state` はウォームスタートや内部状態の保持に使います。
- 何も保持しない場合は、**`eqx.Module` の空クラス**を用意します。
- `dataclass` は使わず、`eqx.Module` に統一します。

## 4. `info` の方針
- `info` は辞書で返します。
- 反復回数・残差・相対誤差・収束判定を含めます。
- 例キー: `num_iter`, `res_norm`, `rel_res`, `converged`

## 5. ログ出力の方針
- **中間ログは `DEBUG` ガードの内側でのみ出力**します。
- ログは **JSONL（1 行 1 JSON）** に統一します。
- 推奨キーは次の通りです。
    - `case`: どのソルバーか（例: `pcg`, `minres`）
    - `source_file`: 実装ファイル（`__file__`）
    - `func`: 関数名（例: `pcg_solve`）
    - `event`: イベント名（例: `residuals`, `summary`, `return`）
    - `iter` / `step`: 反復番号
- **リターン直前に空行を 1 行**挿入し、ログの区切りを明確にします。
- 正解値の出力は不要です（テスト側で出力します）。

## 6. 例（イメージ）
- 戻り値の形だけを示します。

```
import equinox as eqx


class EmptyState(eqx.Module):
    """状態を持たないソルバー用の空構造体。"""


def solve(...):
    ans = ...
    state = EmptyState()
    info = {"num_iter": 0, "res_norm": 0.0, "converged": True}
    return ans, state, info
```
