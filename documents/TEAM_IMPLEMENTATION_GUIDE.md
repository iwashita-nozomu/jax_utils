# 🎯 チーム実装ガイド - Code Quality 改善チェックリスト

## 📘 はじめに

このドキュメントは、現在の Python コード品質を改善するためのタスクを**実装可能な段階的チェックリスト**に分解した正本です。

**全体目標**: `69%` → `95%` の品質スコア改善（推定8-9時間の作業）

______________________________________________________________________

## 🚀 クイックスタート

### コマンド一覧

```bash
# 現在の品質スコアを確認
make ci

# docstring チェックのみ
pydocstyle python/jax_util/

# 個別ファイルの Docstring チェック
pydocstyle python/jax_util/solvers/pcg.py

# 自動修正（import 順序など）
ruff check --fix python/jax_util/

# 型チェック
pyright python/jax_util/
```

______________________________________________________________________

## 📋 フェーズ 1: モジュール Docstring（最優先）

**時間目安**: 30-40 分\
**難度**: ⭐ 簡単\
**影響度**: 🔴 極大（docstring スコア 27% → 45%）

### ✅ 物理的なチェックリスト

**対象**: モジュール先頭に `"""..."""` 形式の Docstring がないファイル（28 個）

#### GROUP 1: BASE モジュール（5個）- 最優先

- [ ] `python/jax_util/base/__init__.py`

  - [ ] モジュール Docstring を追加
  - [ ] `from X import *` を具体化（[ガイド参照](#__all__-%E5%AE%9A%E7%BE%A9%E3%82%AC%E3%82%A4%E3%83%89)）
  - [ ] `__all__` を定義

- [ ] `python/jax_util/base/_env_value.py`

  - [ ] モジュール Docstring を追加

- [ ] `python/jax_util/base/linearoperator.py`

  - [ ] モジュール Docstring を追加
  - [ ] `from X import *` を具体化

- [ ] `python/jax_util/base/nonlinearoperator.py`

  - [ ] モジュール Docstring を追加
  - [ ] `from X import *` を具体化
  - [ ] `__all__` を定義

- [ ] `python/jax_util/base/protocols.py`

  - [ ] モジュール Docstring を追加

#### GROUP 2: SOLVERS モジュール（7個）- 高優先

- [ ] `python/jax_util/solvers/__init__.py`

  - [ ] モジュール Docstring を追加

- [ ] `python/jax_util/solvers/_check_mv_operator.py`

  - [ ] モジュール Docstring を追加

- [ ] `python/jax_util/solvers/_test_jax.py`

  - [ ] モジュール Docstring を追加
  - [ ] `__all__` を定義

- [ ] `python/jax_util/solvers/kkt_solver.py`

  - [ ] モジュール Docstring を追加
  - [ ] `from X import *` を具体化

- [ ] `python/jax_util/solvers/matrix_util.py`

  - [ ] モジュール Docstring を追加
  - [ ] `__all__` を定義

- [ ] `python/jax_util/solvers/pcg.py`

  - [ ] モジュール Docstring を追加
  - [ ] `__all__` を定義

- [ ] `python/jax_util/solvers/slq.py`

  - [ ] モジュール Docstring を追加

#### GROUP 3: FUNCTIONAL モジュール（5個）- 中優先

- [ ] `python/jax_util/functional/__init__.py`

  - [ ] モジュール Docstring を追加

- [ ] `python/jax_util/functional/integrate.py`

  - [ ] モジュール Docstring を追加

- [ ] `python/jax_util/functional/monte_carlo.py`

  - [ ] モジュール Docstring を追加

- [ ] `python/jax_util/functional/protocols.py`

  - [ ] モジュール Docstring を追加

- [ ] `python/jax_util/functional/smolyak.py`

  - [ ] モジュール Docstring を追加

#### GROUP 4: NEURALNETWORK モジュール（6個）- 中優先

- [ ] `python/jax_util/neuralnetwork/__init__.py`

  - [ ] モジュール Docstring を追加

- [ ] `python/jax_util/neuralnetwork/layer_utils.py`

  - [ ] モジュール Docstring を追加

- [ ] `python/jax_util/neuralnetwork/neuralnetwork.py`

  - [ ] モジュール Docstring を追加
  - [ ] `__all__` を定義

- [ ] `python/jax_util/neuralnetwork/protocols.py`

  - [ ] モジュール Docstring を追加

- [ ] `python/jax_util/neuralnetwork/sequential_train.py`

  - [ ] モジュール Docstring を追加

- [ ] `python/jax_util/neuralnetwork/train.py`

  - [ ] モジュール Docstring を追加

#### GROUP 5: その他のモジュール（5個）- 低優先

- [ ] `python/jax_util/__init__.py`

  - [ ] モジュール Docstring を追加

- [ ] `python/jax_util/core.py`

  - [ ] モジュール Docstring を追加

- [ ] `python/jax_util/pdipm.py`

  - [ ] モジュール Docstring を追加

- [ ] `python/jax_util/lobpcg.py`

  - [ ] モジュール Docstring を追加

- [ ] `python/jax_util/_linop_utils.py`

  - [ ] モジュール Docstring を追加

### 🔍 検証方法

```bash
# フェーズ 1 の完成度をチェック
pydocstyle python/jax_util/ 2>&1 | grep -c "^python" || echo "✅ すべて修正完了"

# 個別ファイルの検証
pydocstyle python/jax_util/base/linearoperator.py
```

______________________________________________________________________

## 📋 フェーズ 2: クラス・関数 Docstring

**時間目安**: 3-4 時間\
**難度**: ⭐⭐ 標準\
**影響度**: 🟠 大（docstring スコア 45% → 85%+）

### 対象ファイルと修正項目数

| ファイル                            | クラス数 | 関数数 | 優先度 |
| ----------------------------------- | -------- | ------ | ------ |
| `base/protocols.py`                 | 7        | 0      | 🔴 高  |
| `functional/protocols.py`           | 6        | 0      | 🔴 高  |
| `solvers/` (全)                     | 8        | 12     | 🔴 高  |
| `neuralnetwork/layer_utils.py`      | 7        | 3      | 🟠 中  |
| `neuralnetwork/sequential_train.py` | 2        | 1      | 🟠 中  |
| `functional/monte_carlo.py`         | 1        | 0      | 🟡 低  |

### チェックリスト例

```python
# ❌ 修正前
class PCGState(eqx.Module):
    x: Vector
    r: Vector
    p: Vector

def pcg_solve(A, b, x0=None, rtol=1e-5):
    ...

# ✅ 修正後
class PCGState(eqx.Module):
    """PCG ソルバーの計算状態。
    
    共役勾配法の各イテレーション状態を保持します。
    
    Attributes:
        x: 現在の近似解
        r: 現在の残差
        p: 共役方向
    """
    x: Vector
    r: Vector
    p: Vector

def pcg_solve(
    A: LinearOperator,
    b: Vector,
    x0: Vector | None = None,
    rtol: Scalar = 1e-5,
) -> tuple[Vector, int, dict[str, Any]]:
    """PCG ソルバーを実行。
    
    対称正定値行列システム Ax = b を解きます。
    
    Args:
        A: 対称正定値線形作用素
        b: 右辺ベクトル
        x0: 初期近似解
        rtol: 相対誤差許容度
        
    Returns:
        tuple: (x, n_iter, info)
        
    See Also:
        slq_solve: SLQ ソルバー
    """
    ...
```

______________________________________________________________________

## 🏷️ フェーズ 3: `__all__` 定義と Import 整理

**時間目安**: 1-2 時間\
**難度**: ⭐ 簡単\
**影響度**: 🟡 中（`__all__` スコア 82% → 100%）

### 対象ファイル一覧

#### A) `__all__` が未定義（7ファイル）

- [ ] `python/jax_util/solvers/_test_jax.py`

  - [ ] 公開シンボルを特定（テスト関数？）
  - [ ] `__all__ = [...]` を追加

- [ ] `python/jax_util/solvers/matrix_util.py`

  - [ ] 公開関数を列挙
  - [ ] `__all__` を追加（例: `__all__ = ["tensor_to_matrix", "matrix_to_tensor"]`）

- [ ] `python/jax_util/solvers/pcg.py`

  - [ ] 公開 API を列挙（PCGState, pcg_solve など）

- [ ] `python/jax_util/neuralnetwork/neuralnetwork.py`

  - [ ] 公開クラス・関数を特定

- [ ] `python/jax_util/base/nonlinearoperator.py`

  - [ ] 公開クラスを特定

- [ ] `python/jax_util/base/protocols.py`

  - [ ] Protocol クラスを列挙

- [ ] `python/jax_util/functional/protocols.py`

  - [ ] Protocol クラスを列挙

#### B) Import ワイルドカード（`from X import *`）の修正（4ファイル）

- [ ] `python/jax_util/base/__init__.py`

  - [ ] ❌ Change from: `from .base import *`
  - [ ] ✅ To: `from .base import [explicit_symbols]`

- [ ] `python/jax_util/base/linearoperator.py`

  - [ ] ❌ Change from: `from X import *`
  - [ ] ✅ To: `from X import A, B, C`

- [ ] `python/jax_util/base/nonlinearoperator.py`

  - [ ] ❌ Change from: `from X import *`
  - [ ] ✅ To: 明示的な import

- [ ] `python/jax_util/solvers/kkt_solver.py`

  - [ ] ❌ Change from: `from X import *`
  - [ ] ✅ To: 明示的な import

### 検証方法

```bash
# 自動チェックと修正
ruff check --fix python/jax_util/

# Manual verification
grep -r "from .* import \*" python/jax_util/
# 上記が何も出力しなければ OK
```

______________________________________________________________________

## 📊 進捗トラッキング表

**全体進捗**: `__/_` 完了

```
フェーズ 1: モジュール Docstring
├─ GROUP 1 (BASE): [ ] 0/5
├─ GROUP 2 (SOLVERS): [ ] 0/7
├─ GROUP 3 (FUNCTIONAL): [ ] 0/5
├─ GROUP 4 (NEURALNETWORK): [ ] 0/6
└─ GROUP 5 (Others): [ ] 0/5

フェーズ 2: クラス・関数 Docstring
├─ Protocols: [ ] 0/13
├─ Solvers: [ ] 0/20
├─ Neural Network: [ ] 0/10
└─ Functional: [ ] 0/7

フェーズ 3: `__all__` と Import
├─ `__all__` 追加: [ ] 0/7
└─ wildcard import の修正: [ ] 0/4

品質スコア推移：
69% → フェーズ1後: 45% → フェーズ2後: 85% → フェーズ3後: 95%
```

______________________________________________________________________

## 🔗 参考資料

### ドキュメント

- [📝 Docstring 実装ガイド](./DOCSTRING_GUIDE.md)
- [📄 コーディング規約 - Python](./coding-conventions-python.md)
- [📚 ドキュメントハブ](./README.md)

### コマンド

```bash
# CI 全テスト実行
make ci

# 高速 CI（Ruff skip）
make ci-quick

# 個別ツール実行
pydocstyle python/jax_util/
pyright python/jax_util/
ruff check python/jax_util/
```

### ガイドライン

- **Docstring テンプレート**: [coding-conventions-python.md](./coding-conventions-python.md#-docstring-%E3%83%86%E3%83%B3%E3%83%97%E3%83%AC%E3%83%BC%E3%83%88)
- **`__all__` ルール**: [DOCSTRING_GUIDE.md](./DOCSTRING_GUIDE.md#-__all__-%E5%AE%9A%E7%BE%A9%E3%82%AC%E3%82%A4%E3%83%89)

______________________________________________________________________

## 📈 期待される改善

| メトリクス       | 現在    | 目標    | 改善度    |
| ---------------- | ------- | ------- | --------- |
| Docstring 完全性 | 27%     | 95%     | +68pp     |
| `__all__` 定義   | 82%     | 100%    | +18pp     |
| Import 管理      | 89%     | 100%    | +11pp     |
| **全体スコア**   | **69%** | **95%** | **+26pp** |

______________________________________________________________________

## ⏱️ タイムライン

**推奨**: 1-2 週間で段階的に実施

- **Week 1 Monday**: フェーズ 1（30-40 分）
- **Week 1 Wednesday**: フェーズ 2（3-4 時間）
- **Week 2 Monday**: フェーズ 3（1-2 時間）
- **Week 2 Wednesday**: CI パイプラインで最終検証

______________________________________________________________________

## 🎯 成功の定義

```bash
# CI が完全に成功
make ci
# → ✅ pytest 成功
# → ✅ pyright 成功
# → ✅ pydocstyle 成功
# → ✅ ruff (D,E,F,I,UP) 成功
```

**品質スコア**: `95%` 以上を達成

______________________________________________________________________

**作成日**: 2026-03-20\
**対象チーム**: Python 開発者\
**推定グループ作業時間**: 6-8 時間
