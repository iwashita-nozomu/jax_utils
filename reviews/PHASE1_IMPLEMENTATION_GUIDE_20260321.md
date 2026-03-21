# エージェント体制・コード品質改善 実装ガイド

**対象**: GitHub Copilot + プロジェクト全体  
**期限**: 2026-03-28 （1 週間）  
**フェーズ**: Phase 1 （基盤強化）

---

## 完了済みタスク ✅

| タスク | 実施日 | 状態 |
|---|---|---|
| 禁止ファイル `python/test.py` 削除 | 2026-03-21 | ✅ `scripts/exploratory/test_jax_example.py` へ移動 |
| 禁止ファイル `_test_jax.py` 整理 | 2026-03-21 | ✅ `python/tests/solvers/test_jax_debug.py` へ移動 |
| Reviewer チェックリスト追加 | 2026-03-21 | ✅ `.github/AGENTS.md` に 18 項目追加 |
| 詳細レビューレポート生成 | 2026-03-21 | ✅ `reviews/DETAILED_THOROUGH_CODE_REVIEW_*` |

---

## Phase 1 実装タスク（残り）

### ✅ タスク 1: 禁止ファイル完全削除

**状態**: 完了 ✅

```bash
# 実行済み
mv python/test.py scripts/exploratory/test_jax_example.py
mv python/jax_util/solvers/_test_jax.py python/tests/solvers/test_jax_debug.py
```

**確認コマンド**:
```bash
cd /workspace && find . -name "_test_*.py" -o -name "test.py" | grep -v "python/tests"
# 結果: 出力なし（OK）
```

---

### 📋 タスク 2: pre_review.sh スクリプト作成

**目的**: PR 前の QA チェックを自動化

**所要時間**: 30 分

**ファイル**: `/workspace/scripts/ci/pre_review.sh` (新規)

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "PRE-REVIEW QA CHECKS"
echo "=========================================="

# 1. Type checking
echo ""
echo "1️⃣  Type Checking (Pyright strict mode)..."
python3 -m pyright ./python/jax_util || {
  echo "❌ Type errors found. Review code."
  exit 1
}

# 2. Test execution
echo ""
echo "2️⃣  Running pytest..."
python3 -m pytest python/tests/ -q --tb=short || {
  echo "❌ Test failures. Fix tests."
  exit 1
}

# 3. Docstring validation
echo ""
echo "3️⃣  Docstring validation (pydocstyle)..."
python3 -m pydocstyle python/jax_util || {
  echo "⚠️  Docstring issues. Review output above."
}

# 4. Code quality
echo ""
echo "4️⃣  Code quality checks (Ruff)..."
python3 -m ruff check python/jax_util --select E,F,I,D,UP || {
  echo "⚠️  Style issues found."
}

echo ""
echo "=========================================="
echo "✅ PRE-REVIEW CHECKS COMPLETE"
echo "=========================================="
echo ""
echo "Next: Commit changes and open PR"
```

**実装手順**:

1. ファイル作成
2. `chmod +x scripts/ci/pre_review.sh` で実行可能化
3. Git に登録: `git add scripts/ci/pre_review.sh`

**テスト**:
```bash
cd /workspace
scripts/ci/pre_review.sh
# 期待: "✅ PRE-REVIEW CHECKS COMPLETE"
```

---

### 📋 タスク 3: CI ワークフロー改善 (.github/workflows/ci.yml)

**状態**: 既に改善済み（前回のセッションで カバレッジ追加）

**確認**:
```bash
cat .github/workflows/ci.yml | grep -A 5 "coverage\|cov-report"
# 期待: cov-report=html, cov-report=term が見える
```

**不足があれば追加**:
```yaml
- name: Run pytest with coverage
  run: python -m pytest python/tests/ \
    --cov=python/jax_util \
    --cov-report=term \
    --cov-report=html \
    -v

- name: Upload coverage to artifacts
  if: always()
  uses: actions/upload-artifact@v3
  with:
    name: coverage-report
    path: htmlcov/
```

---

### 📋 タスク 4: JSON ログ統一（base テスト）

**現状**: base/test_*.py で JSON ログが未実装

**対象ファイル**:
- `python/tests/base/test_linearoperator.py`
- `python/tests/base/test_nonlinearoperator.py`
- `python/tests/base/test_env_value.py`

**所要時間**: 1h (3 ファイル)

**改善例** (linearoperator.py):

```python
# Before
def test_linearoperator_matmul_vector() -> None:
    """LinOp @ ベクトル適用テスト."""
    A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    op = LinOp(lambda v: A @ v)
    x = jnp.array([1.0, 2.0])
    y = op @ x
    assert jnp.allclose(y, A @ x)

# After
import json
from pathlib import Path

SOURCE_FILE = Path(__file__).name

def test_linearoperator_matmul_vector() -> None:
    """LinOp @ ベクトル適用テスト."""
    A = jnp.array([[4.0, 1.0], [1.0, 3.0]])
    op = LinOp(lambda v: A @ v)
    x = jnp.array([1.0, 2.0])
    y = op @ x
    
    # JSON ログ出力（追加）
    print(json.dumps({
        "case": "linop_matmul_vector",
        "source_file": SOURCE_FILE,
        "expected": (A @ x).tolist(),
        "actual": y.tolist(),
        "rel_err": float(jnp.max(jnp.abs(y - A @ x) / (jnp.abs(A @ x) + 1e-10)))
    }))
    
    assert jnp.allclose(y, A @ x)
```

**実装チェックリスト**:
- [ ] print(json.dumps({...})) を各テストに追加
- [ ] キー名を統一: `case`, `expected`, `actual`, `rel_err` など
- [ ] `pytest python/tests/base/test_*.py -s` で JSON 出力確認
- [ ] pydocstyle 通過確認

---

### 📋 タスク 5: Docstring 統一化

**現状**: public 関数の docstring が 95% 完備；private 関数が 85%

**対象**: private 関数の docstring 追加

**例** (solvers/pcg.py の `_init_residual` など):

```python
# Before
def _init_residual(
    b: Vector, A_mv: Callable[[Vector], Vector], x0: Vector | None
) -> tuple[Vector, Vector]:
    """残差ベクトルを初期化."""
    ...

# After
def _init_residual(
    b: Vector, A_mv: Callable[[Vector], Vector], x0: Vector | None
) -> tuple[Vector, Vector]:
    """残差ベクトルを初期化.
    
    初期解 x0 が与えられた場合は r = b - A @ x0 を計算する。
    
    Args:
        b: 右辺ベクトル
        A_mv: 行列ベクトル積関数 A @ v
        x0: 初期解（None の場合ゼロベクトル）
        
    Returns:
        残差ベクトル r と初期解 x0 のペア
    """
    ...
```

**実装チェックリスト**:
- [ ] `python -m pydocstyle python/jax_util` で 100% 通過
- [ ] 改造ファイル 5-6 個程度

---

## Phase 1 完了チェックリスト

```markdown
### ✅ 実施確認（すべてチェック必須）

- [x] python/test.py 削除 / 移動
- [x] _test_jax.py 削除 / 移動
- [x] .github/AGENTS.md に Reviewer チェックリスト追加
- [ ] scripts/ci/pre_review.sh 作成 + 実行確認
- [ ] CI yaml に カバレッジ設定確認（既済）
- [ ] base テスト JSON ログ追加（3 ファイル）
- [ ] private 関数 docstring 追加（5-6 ファイル）

### ✅ 最終確認コマンド

cd /workspace

# 禁止ファイル確認
find . -name "_test_*.py" -o -name "test.py" | grep -v "python/tests" | wc -l
# 期待: 0

# 型チェック
python3 -m pyright ./python/jax_util
# 期待: 0 errors, 0 warnings

# テスト実行
python3 -m pytest python/tests/ -q
# 期待: XX passed

# Docstring 検証
python3 -m pydocstyle python/jax_util
# 期待: 0 warnings

# Ruff
python3 -m ruff check python/jax_util --select E,F,I,D,UP
# 期待: No errors

# pre_review.sh 実行
scripts/ci/pre_review.sh
# 期待: ✅ PRE-REVIEW CHECKS COMPLETE
```

---

## Phase 2 予定（3-4 週間後）

| タスク | 内容 | 時間 |
|---|---|---|
| 監査レポート Template | reviews/AUDIT_REPORT_TEMPLATE.md 作成 | 30 分 |
| Integrator CI 強化 | merge conflict 検出・通知機能 | 2h |
| Coordinator TAG 導入 | GitHub Labels/Projects 設定 | 1h |

---

## 実装進捗記録

**このファイルを活用して、Phase 1 の進捗を追跡してください。**

実装者は以下の方式で更新：

```markdown
### [2026-03-21] pre_review.sh 作成
- 作成者: [Name]
- 実行確認: ✅
- 備考: CI yaml と連携確認予定

### [2026-03-22] JSON ログ統一
- ファイル 3 個完了
- 備考: base/test_linearoperator.py, test_nonlinearoperator.py, test_env_value.py
```

---

## 参考資料

- **規約**: [documents/coding-conventions-python.md](../documents/coding-conventions-python.md)
- **テスト規約**: [documents/coding-conventions-testing.md](../documents/coding-conventions-testing.md)
- **詳細レビュー**: [reviews/DETAILED_THOROUGH_CODE_REVIEW_20260321.md](DETAILED_THOROUGH_CODE_REVIEW_20260321.md)
- **AGENTS 運用**: [.github/AGENTS.md](../.github/AGENTS.md)

---

**作成日**: 2026-03-21  
**次回フォローアップ**: 2026-03-28  
**完了予想**: Phase 1 完了後、Phase 2 へ移行
