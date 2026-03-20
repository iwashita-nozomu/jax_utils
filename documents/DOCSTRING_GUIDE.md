# Docstring ガイドと `__all__` 統一ルール

## 概要

このドキュメントは、コードレビューレポート（2026-03-20）で指摘された **docstring 欠落** と **`__all__` 定義の不統一** を改善するためのガイドです。

---

## 📝 Docstring 実装ガイド

### なぜ docstring が重要なのか？

- **自動ドキュメント生成**: Sphinx や MkDocs で自動抽出
- **IDE 統合**: ホバー表示で関数説明が即座に確認できる
- **保守性向上**: 実装者の意図が明確になる
- **規約遵守**: コーディング規約で明示的に要求

**規約**: [documents/coding-conventions-python.md](./coding-conventions-python.md)

---

## 📋 実装パターン

### 1. モジュール Docstring（**必須**）

**ファイル先頭に配置**

```python
"""module_name パッケージの概要（1行）。

このモジュールは[責務]を担当します。
- 主要機能1
- 主要機能2

公開インターフェース:
    PublicClass: [説明]
    public_function: [説明]

参考資料:
    - documents/coding-conventions-python.md
    - [リンク]
    
実装例:
    >>> from jax_util.module_name import PublicClass
    >>> obj = PublicClass(...)
"""

from __future__ import annotations

# ... import 以下のコード ...
```

**チェックリスト**:
- [ ] モジュールの目的が明確
- [ ] 公開シンボルがリストアップされている
- [ ] 参考資料・リンクが明示

---

### 2. クラス Docstring（**必須**）

```python
class LinearOperator(eqx.Module):
    """線形演算子の抽象クラス。
    
    行列・線形変換を表現する基本クラス。
    `shape` プロパティで次元情報を提供し、
    `__matmul__` と `__mul__` で演算をサポートします。
    
    Attributes:
        shape (tuple): 演算子の形状 (m, n)
        
    See Also:
        NonLinearOperator: 非線形演算子
        
    Example:
        >>> class MyOp(LinearOperator):
        ...     def __init__(self, mv):
        ...         self.mv = mv
        ...     @property
        ...     def shape(self):
        ...         return (n, n)
        ...     def __matmul__(self, x):
        ...         return self.mv(x)
        
        >>> A = MyOp(my_matrix_vector_product)
        >>> result = A @ vector
    """
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """演算子の形状 (m, n)。"""
        ...
```

**チェックリスト**:
- [ ] クラスの役割が1行で明確
- [ ] 重要なメソッドやプロパティを記載
- [ ] 使用例が含まれている

---

### 3. 関数 Docstring（**必須**）

```python
def linear_solve(
    A: LinearOperator, 
    b: Vector, 
    x0: Optional[Vector] = None
) -> Tuple[Vector, int, Dict[str, Any]]:
    """線形方程式 Ax = b を解く。
    
    前処理付き CG 法で疎行列の線形方程式を解きます。
    行列 A は対称正定値である必要があります。
    
    Args:
        A: 対称正定値行列演算子
        b: 右辺ベクトル (shape=(n,))
        x0: 初期推定値。省略時は零ベクトル
        
    Returns:
        解ベクトル x (shape=(n,))
        反復回数 (int)
        メタデータ dict
        
    Raises:
        ValueError: A が正方行列でない
        ValueError: b の次元が A と一致しない
        
    Example:
        >>> import jax.numpy as jnp
        >>> from jax_util.solvers import linear_solve
        >>> 
        >>> A = LinearOperator(jax_matvec_product, shape=(100, 100))
        >>> b = jnp.ones(100)
        >>> x, iterations, info = linear_solve(A, b)
        >>> print(f"収束: {iterations} 反復")
        
    Notes:
        - 数値安定性確保のため正規化を内部で実施
        - 予条件行列を指定可能（将来）
    """
    ...
```

**チェックリスト**:
- [ ] 関数の目的が1文で説明されている
- [ ] Args に全パラメータが記載
- [ ] Returns が戻り値の型と意味を説明
- [ ] Raises で例外を記載
- [ ] Example で実行可能なコード例

---

## 🏷️ `__all__` 定義ガイド

### 目的

- **公開 API の明示**: ユーザーが `from module import *` で何を得るか明確にする
- **型チェッカーの支援**: Pylance が `__all__` から公開シンボルを推論
- **ドキュメント生成**: Sphinx がこれを元に API ドキュメント作成

### 実装パターン

**基本形式**:

```python
__all__ = [
    "PublicClass",
    "public_function",
    "PUBLIC_CONSTANT",
]
```

**ルール**:
- 大文字・小文字を正確に
- アルファベット順でソート（推奨）
- プライベート（`_` prefix）は含めない
- 型チェッカー用のシンボルも含める

### 悪い例 ❌

```python
# NG: __all__ がない
def public_func():
    pass

def _private_func():
    pass

# NG: プライベート関数を公開
__all__ = ["public_func", "_private_func"]

# NG: from X import *
from . import some_module as *
```

### 良い例 ✅

```python
"""エクスポート構造を明示。"""

from typing import Protocol

class Operator(Protocol):
    """演算子プロトコル。"""
    pass

class MyClass:
    """パブリッククラス。"""
    pass

def my_function():
    """パブリッシック関数。"""
    pass

# 公開インターフェースを明示
__all__ = [
    "MyClass",
    "Operator",
    "my_function",
]
```

---

## 📊 チェックリスト

### 新規ファイル作成時

- [ ] モジュール docstring を先頭に追加
- [ ] `__all__` を定義（公開シンボルがある場合）
- [ ] すべてのパブリッククラスに docstring を追加
- [ ] すべてのパブリック関数に docstring を追加
- [ ] `from X import *` を使わない

### 既存ファイル修正時

- [ ] 修正対象のクラス/関数に docstring を追加
- [ ] 例があれば Example セクションを追加
- [ ] `__all__` が存在する場合、修正内容を反映
- [ ] `pyright --outputjson` で型チェック実行

---

## 🔧 自動チェック

### ローカル実行

```bash
# docstring チェック（pydocstyle）
pydocstyle python/jax_util/

# スタイル + import + docstring チェック（ruff）
ruff check python/jax_util/ --select D

# 型チェック
pyright python/jax_util/
```

### CI/CD パイプライン

```bash
# make ci で実行
make ci

# または手動実行
bash scripts/ci/run_all_checks.sh
```

---

## 📈 改善の段階

### フェーズ 1（今週）

- [ ] モジュール docstring をすべてのファイルに追加
- [ ] `__all__` を未定義ファイルに追加

**所要時間**: 1-2 時間

### フェーズ 2（来週）

- [ ] 既存クラスに docstring を追加
- [ ] 既存関数に docstring を追加
- [ ] Example セクションを充実

**所要時間**: 4-6 時間

### フェーズ 3（完成）

- [ ] Sphinx でドキュメント生成テスト
- [ ] CI パイプラインに自動チェック統合
- [ ] チームレビュー

---

## 参考資料

- [Google Python Style Guide - Comments and Docstrings](https://google.github.io/styleguide/pyguide.html)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [NumPy Docstring Standard](https://numpydoc.readthedocs.io/en/latest/format.html)
- [documents/coding-conventions-python.md](./coding-conventions-python.md)
- [CODE_REVIEW_REPORT_2026-03-20.md](./CODE_REVIEW_REPORT_2026-03-20.md)

---

**作成日**: 2026-03-20  
**対象**: Python コードベース全体（`python/jax_util/`）  
**関連**: コードレビューレポート優先度 P1-P2
