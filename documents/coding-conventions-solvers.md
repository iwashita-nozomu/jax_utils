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
- 何も保持しない場合は、空の構造体を用意します。
- 例:
	- `@dataclass(frozen=True)` の空クラス
	- `eqx.Module` の空クラス

## 4. `info` の方針
- `info` は辞書で返します。
- 反復回数・残差・相対誤差・収束判定を含めます。
- 例キー: `num_iter`, `res_norm`, `rel_res`, `converged`

## 5. 例（イメージ）
- 戻り値の形だけを示します。

```
@dataclass(frozen=True)
class EmptyState:
    """状態を持たないソルバー用の空構造体。"""


def solve(...):
    ans = ...
    state = EmptyState()
    info = {"num_iter": 0, "res_norm": 0.0, "converged": True}
    return ans, state, info
```
