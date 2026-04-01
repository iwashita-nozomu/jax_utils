# スキル実装ガイド（v3.3）

**目的**: `.code-review-SKILL.md` と `.infrastructure-review-SKILL.md` の実装方法を体系化

---

## 1. スキル体系図

```
┌─────────────────────────────────────────────────────────────┐
│ 総合コード品質レビュー                                       │
└─────────────────────────────────────────────────────────────┘
         │
         ├─────────────────────────────────────────────────────────┐
         │                                                           │
    ┌────┴──────────┐                                  ┌────────────┴──────────┐
    │ code-review   │                                  │ infrastructure-review │
    └────┬──────────┘                                  └───────────┬───────────┘
         │                                                         │
    ┌────┴─────────────────────────────────────────────────────┐  │
    │ A層 基礎検証                                              │  │
    ├────────────────────────────────────────────────────────┤  │
    │ 1. Python コード - 型注釈・Docstring・テスト           │  │
    │ 2. C++ コード - ヘッダガード・メモリ管理               │  │
    │ 3. ドキュメント - 形式・一貫性・参照性                 │  │
    └────┬─────────────────────────────────────────────────┘  │
         │                                                      │
    ┌────┴─────────────────────────────────────────────────┐  │  1. Docker
    │ B層 深度検証                                          │  │  2. requirements.txt
    ├────────────────────────────────────────────────────┤  │  3. CI/CD
    │ 4. Python テスト・アーキテクチャ                    │  │
    │ 5. C++ スタイル・ベストプラクティス                 │  │
    └────┬─────────────────────────────────────────────┘  │
         │                                                │
    ┌────┴─────────────────────────────────────────────┐  │
    │ C層 統合検証                                      │  │
    ├────────────────────────────────────────────────┤  │
    │ 6. プロジェクト規約との整合性                  │  │
    │ 7. 規約ファイル矛盾検出                        │  │
    │ 8. 実装⟷Doc⟷テストの三点セット検証            │  │
    └────┬─────────────────────────────────────────┘  │
         │                                             │
    ┌────┴─────────────────────────────────────────┐  │
    │ D層 プロセス                                 │  │
    ├────────────────────────────────────────────┤  │
    │ 9. レビュー実施フロー・チェックリスト     │  │
    │ 10. よくある問題と対応                      │  │
    └────┬─────────────────────────────────────┘  │
         │                                        │
    ┌────┴─────────────────────────────────────┐  │
    │ E層 報告書テンプレート                    │  │
    ├────────────────────────────────────────┤  │
    │ 11. 標準編（Python/C++/Doc）           │  │
    │ 12. 実験編（JAX・Smolyak等）           │  │
    └────┬─────────────────────────────────┘  │
         │                                    │
    ┌────┴─────────────────────────────────┐  │
    │ F層 参考資料                          │  │
    ├────────────────────────────────────┤  │
    │ 13. ツール・スクリプト実装例       │  │
    │ 14. チェックリスト詳細             │  │
    └────┬─────────────────────────────┘  │
         │                                │
    ┌────┴─────────────────────────────┐  │
    │ G層 特殊環境                      │  │
    ├────────────────────────────────┤  │
    │ 15. 実験コード レビュー         │  │
    │ 16. 実験ワークツリー・レポート │  │
    │ 17. 実験ログ・エクスペリメント │  │
    └────────────────────────────────┘  │
                                         │
                                         └─→ 統合実施
```

---

## 2. スキル適用の流れ

### Phase 1: コード作成中（開発者）

**対応スキル**: A層（基礎検証）+ 各言語固有チェック

```
1. 型注釈を付与 [code-review 1.1]
2. Docstring を記述 [code-review 1.1]
3. テストを作成 [code-review 1.2]
4. 実装を完成させる [code-review 1.3]
5. ローカルでチェック実行
   bash ./scripts/run_comprehensive_review.sh --report
```

### Phase 2: PR 作成前（開発者）

**対応スキル**: B層（深度検証）+ C層（統合検証）

```
1. セクション 11 の「PR 作成前」チェックリストを実行
2. Docker ビルド・テスト [infrastructure-review 1-2]
3. 規約との矛盾がないか確認 [code-review 6-7]
```

### Phase 3: PR レビュー（レビュアー）

**対応スキル**: A層 → B層 → C層 → セクション 11（報告書）

```
1. A層 基礎検証: ファイル単位のチェック
2. B層 深度検証: 関数・モジュール単位の検証
3. C層 統合検証: プロジェクト全体との整合性
4. セクション 11 テンプレートで報告書を作成
5. 改善について議論・承認判定
```

### Phase 4: Merge 前（Maintainer）

**対応スキル**: セクション 9（実施フロー）+ 自動チェック群

```
1. CI/CD パイプライン成功確認
2. 自動チェック群（pyright, pytest, triplet）成功確認
3. セクション 9 の「Merge 時フロー」を実行
```

---

## 3. 各スキルセクションの位置付け

| レイヤー | セクション | 主体 | 実施タイミング | 自動化可能性 |
|---------|-----------|------|--------------|----------|
| **A層** | 1-3 | 開発 + CI | 作成中 + Merge | ⭐⭐⭐⭐⭐ |
| **B層** | 4-5 | レビュアー | PR レビュー | ⭐⭐⭐⭐ |
| **C層** | 6-8 | レビュアー | PR レビュー | ⭐⭐⭐ |
| **D層** | 9-10 | Maintainer | Merge フロー | ⭐⭐ |
| **E層** | 11-12 | レビュアー | レビュー完了後 | ⭐ |
| **Infrastructure** | 1-7 | DevOps + CI | PR 作成前 + Merge | ⭐⭐⭐⭐ |

---

## 4. ツール群の活用

### 4.1 自動化ツール群

| ツール | 対応セクション | 実行タイミング |
|--------|--------------|-------------|
| **pyright** | A層(1.1) | CI, ローカル |
| **ruff** | A層(1.1) | CI, ローカル |
| **pytest** | A層(1.2) | CI, ローカル |
| **check_doc_test_triplet.py** | C層(8) | CI, PR レビュー |
| **check_convention_consistency.py** | C層(6-7) | CI, PR レビュー |
| **docker_dependency_validator.py** | Infrastructure(2) | CI, PR 前 |
| **requirement_sync_validator.py** | Infrastructure(2) | CI, PR 前 |
| **mdformat** | A層(3) | CI, ローカル |

### 4.2 実行コマンド例

```bash
# ローカル開発時（全チェック）
./scripts/run_comprehensive_review.sh --parallel --report

# CI/CD パイプライン
./scripts/run_comprehensive_review.sh --parallel --report

# 特定のチェックのみ
python scripts/check_doc_test_triplet.py
python scripts/docker_dependency_validator.py
```

---

## 5. 実装パターン

### 5.1 Python コード のベストプラクティス

```python
"""モジュール概要説明。

このモジュールは... 詳細説明
"""

from __future__ import annotations

from typing import Optional
import jax
import jax.numpy as jnp

Vector = jnp.ndarray  # 型注釈エイリアス

@jax.jit
def solve_linear_system(
    A: Vector, 
    b: Vector, 
    tol: float = 1e-6
) -> tuple[Vector, int]:
    """線形方程式 Ax=b を求解。
    
    Args:
        A: 係数行列 (n, n)
        b: 右辺ベクトル (n,)
        tol: 収束判定の許容誤差
        
    Returns:
        (x, iterations): 解ベクトル x と反復回数
        
    Raises:
        ValueError: A が正方行列でない場合
        RuntimeError: 収束失敗時
    """
    # 実装...
    return x, iterations

# テスト
def test_solve_linear_system_with_identity_matrix():
    """恒等行列 I@x = x で正確に求解することを検証。"""
    n = 5
    A = jnp.eye(n)
    b = jnp.ones(n)
    x, _ = solve_linear_system(A, b)
    assert jnp.allclose(x, b)
```

### 5.2 C++ コード のベストプラクティス

```cpp
#pragma once

#include <vector>
#include <memory>
#include <stdexcept>

namespace jax_util {

/**
 * 線形ソルバー（GMRES）.
 */
class LinearSolver {
public:
    /**
     * コンストラクタ.
     *
     * @param max_iterations 最大反復回数
     * @param tolerance 収束判定の許容誤差
     */
    LinearSolver(int max_iterations, double tolerance)
        : max_iterations_(max_iterations), tolerance_(tolerance) {}
    
    /**
     * 線形方程式 Ax = b を求解.
     *
     * @param A 係数行列 (n&times;n)
     * @param b 右辺ベクトル (n,)
     * @return 解ベクトル x
     *
     * @throws std::invalid_argument A が正方行列でない場合
     * @throws std::runtime_error 収束失敗時
     */
    std::vector<double> solve(
        const std::vector<std::vector<double>>& A,
        const std::vector<double>& b
    ) const {
        if (A.size() != A[0].size()) {
            throw std::invalid_argument("A must be square");
        }
        // 実装...
        return x;
    }
    
private:
    int max_iterations_;
    double tolerance_;
};

}  // namespace jax_util
```

---

## 6. よくあるレビューコメント

### Python

| パターン | セクション | 対応 |
|---------|-----------|------|
| 型注釈がない | 1.1 | `-> None:` 等を追加 |
| Docstring に Returns がない | 1.1 | 戻り値型を明記 |
| テストがない | 1.2 | pytest テストを追加 |
| 例外処理が generic | 1.3 | 具体的な例外型を throw |
| グローバル変数 | 1.3 | クラス属性またはパラメータ化 |

### C++

| パターン | セクション | 対応 |
|---------|-----------|------|
| ヘッダガードなし | 2.1 | `#pragma once` を追加 |
| メモリリーク | 2.1 | `unique_ptr` を使用 |
| 生ポインタ | 2.1 | スマートポインタに変更 |
| bounds チェックなし | 2.1 | 例外を throw |
| インデント混在 | 2.2 | 統一（2 or 4 space） |

---

## 7. レビュー効率化のツール

### 7.1 IDE 統合

```jsonc
// .vscode/settings.json
{
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.python",
    "editor.formatOnSave": true
  },
  "pylance.enable": true,
  "pylance.typeCheckingMode": "strict"
}
```

### 7.2 Git Hooks

```bash
# .git/hooks/pre-commit
#!/bin/bash

# Python tests
python -m pytest python/tests/ -q || exit 1

# Code style
python -m ruff check python/jax_util || exit 1

# Type checking
python -m pyright python/jax_util --outputjson > /dev/null || exit 1

exit 0
```

---

## 8. エスカレーション・判定フロー

```
┌──────────────────────────┐
│  PR レビュー開始         │
└─────────────┬────────────┘
              │
┌─────────────┴────────────────────┐
│  A層（基礎検証）                 │
│ - 型注釈、Docstring、テスト      │
└─────────────┬────────────────────┘
              │
         ┌────┴────┐
         │   合格? │
         │ ↙  ↘  │
         N    Y   │
         │         │
    ┌────┘     ┌───┴──────────────────────┐
    │          │  B層（深度検証）         │
    │          │ - 設計、テスト coverage │
    │          └───┬──────────────────────┘
    │              │
    │         ┌────┴────┐
    │         │   合格? │
    │         │ ↙  ↘  │
    │         N    Y   │
    │         │         │
    │    ┌────┘     ┌───┴──────────────────────┐
    │    │          │  C層（統合検証）         │
    │    │          │ - 規約整合性、三点セット │
    │    │          └───┬──────────────────────┘
    │    │              │
    │    │         ┌────┴────┐
    │    │         │   合格? │
    │    │         │ ↙  ↘  │
    │    │         N    Y   │
    │    │         │         │
    │    │    ┌────┘      ┌──┴────────────┐
    │    │    │           │  承認 ✅      │
    │    │    │           │  → Merge    │
    │    │    │           └──────────────┘
    │    │    │
    │    └────┤ コメント返却
    │         │ 修正要求
    └─────────┤ 再レビュー
             │
          ┌──┴───────────────┐
          │  修正完了？       │
          │ ↙           ↘   │
          Y             N   │
          │               再提出
          └───→ A層から再検証
```

---

## 9. 参考資料

- [code-review-SKILL.md](../.code-review-SKILL.md) - コードレビュースキル詳細
- [infrastructure-review-SKILL.md](../.infrastructure-review-SKILL.md) - インフラレビュースキル
- [documents/coding-conventions-project.md](../documents/coding-conventions-project.md) - プロジェクト規約
- [scripts/run_comprehensive_review.sh](../scripts/run_comprehensive_review.sh) - 統合実行スクリプト

---

**【情報】** 本ガイドは `.code-review-SKILL.md` v3.3 および `.infrastructure-review-SKILL.md` v1.0 に対応しています。
