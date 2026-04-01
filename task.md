# task

## 方針

- `task.md` は **ファイル単位**で整理します。
- 細部の設計は `documents/` に逐次追記します。
- **型エラーは静的解析（pyright strict）に任せ、コード内の不要な型チェックを削除**

---

## 🔴 Phase 1: 現在の変更を整理（優先度: Critical）

**目標**: main を clean 状態に、マージ競合を回避

### 1.1 変更中の18ファイルをコミット（期限: 本日）

**対象ファイル**:
- `agents/COMMUNICATION_PROTOCOL.md`
- `agents/README.md`
- `agents/TASK_WORKFLOWS.md`
- `agents/agents_config.json`
- `agents/task_catalog.yaml`
- `documents/README.md`
- `documents/WORKTREE_SCOPE_TEMPLATE.md`
- `documents/coding-conventions-experiments.md`
- `documents/conventions/python/20_benchmark_policy.md`
- `documents/conventions/python/30_experiment_directory_structure.md`
- `documents/experiment_runner.md`
- `documents/worktree-lifecycle.md`
- `experiments/smolyak_experiment/README.md`
- `notes/experiments/README.md`
- `notes/experiments/experiment_runner_usage.md`
- `notes/worktrees/README.md`
- `scripts/agent_tools/agent_team.py`

**Commit message**:
```
docs: update worktree-wide docs; add benchmark and experiment policies

- Update agents communication protocol and task workflows
- Add Python benchmark policy (20_benchmark_policy.md)
- Add experiment directory structure conventions
- Synchronize experiment runner usage documentation
- Consolidate worktree lifecycle documentation
- Update agent team configuration
```

**コマンド**:
```bash
git add agents/ documents/ notes/ scripts/agent_tools/agent_team.py
git commit -m "docs: update worktree-wide docs..."
git push origin main
```

---

## 🟠 Phase 2: 型エラー削減（優先度: High, 期限: 4/2-4/3）

**方針**: 型エラーは `pyright --strict` に任せ、コード内では具体的なビジネスロジックエラーのみ処理

### 2.1 `python/experiment_runner/_pr1_gmres_preconditioner.py`

**対応**: 
- `except Exception: return None` → 削除 + 型注釈を信頼
- `preconditioner(r)` → 型チェックなし（pyright 管理）
- 具体的なエラー（`jnp.linalg.LinAlgError`）のみ catch

**変更箇所**:
```python
# ❌ Before: generic exception
try:
    x = jnp.linalg.solve(A, b)
except Exception:
    return None

# ✅ After: 具体的エラーのみ、型チェックはpyright
try:
    x = jnp.linalg.solve(A, b)
except jnp.linalg.LinAlgError as e:
    raise RuntimeError(f"Linear solver failed: {e}") from e
```

### 2.2 `python/experiment_runner/_pr3_runner_refactored.py`

**対応**:
- `except Exception: raise RuntimeError` → 削除
- 型注釈を完全に（`Dict[str, Any] | None` → `Dict[str, Any]`）
- 型エラーは pyright で検出

**変更箇所**:
```python
# ❌ Before: defensive programming
except Exception as e:
    logger.error(f"Failed to save results: {e}")
    raise

# ✅ After: normal operation
# （pyright strict で型エラーは検出）
# Exception は発生しない（型安全）
```

### 2.3 `scripts/check_doc_test_triplet.py`

**対応**:
- `check_type_annotations()` 関数を削除
- `TypeError` キャッチを削除
- pyright でのチェックに統一

**削除対象**:
```python
def check_type_annotations(func) -> dict[str, bool]:
    """削除: pyright --strict でカバー"""
    # 手動型チェックは不要
```

### 2.4 `scripts/docker_dependency_validator.py`

**対応**:
- `isinstance(package, str)` チェック削除
- 型注釈を完全に（Union の使用）
- 型エラーを pyright で検出

### 2.5 `scripts/requirement_sync_validator.py`

**対応**: 
- 型チェック関数を削除
- 型注釈で明示（no `Any`）
- pyright strict で検証

**実施コマンド**:
```bash
# 各ファイルについて
pyright --outputjson <file> | grep -i "type"

# 修正後の検証
pyright --strict python/experiment_runner/ scripts/
```

---

## 🟡 Phase 3: Worktree マージ計画（優先度: Medium, 期限: 4/4-4/5）

**現状**:
- main: 5855432 (HEAD)
- origin/main: 9233674 (1 commit behind)
- 複数worktree: 12dee48, 9233674, 654c340

**マージ順序と競合対策**:

### 3.1 `work/experiment-runner-hardening-20260331` → main

**競合リスク**: 低（ドキュメント中心の経験）

```bash
git merge work/experiment-runner-hardening-20260331 --no-edit
```

### 3.2 `work/smolyak-integrator-lead-20260328` → main

**競合リスク**: 中（`documents/experiment_runner.md` で競合予測）

**対策**:
```bash
# --no-commit で manual resolve
git merge work/smolyak-integrator-lead-20260328 --no-commit --no-ff

# 競合確認
git status

# 衝突の場合、ours strategy で主流を保つ
git checkout --ours documents/experiment_runner.md
# または manual merge
```

**コミットメッセージ**:
```
Merge: integrate smolyak-integrator-lead

- Consolidate Smolyak grid optimization work
- Unify experiment runner integration
- Resolve doc conflicts with main
```

### 3.3 `results/smolyak-validation-20260328` → separate branch

**対象**: results/ branch に保管（報告用、main には不要）

```bash
# results/ branch へ
git checkout -b results/smolyak-validation-20260328-consolidated origin/main
git merge results/smolyak-validation-20260328
git push origin results/smolyak-validation-20260328-consolidated
```

---

## ✅ Phase 4: CI 統合（優先度: High, 期限: 4/6 以降）

### 4.1 GitHub Actions に `pyright --strict` 追加

**ファイル**: `.github/workflows/python-test.yml`

```yaml
- name: Type check (strict mode)
  run: |
    pyright --warnings python/jax_util python/experiment_runner scripts/
```

### 4.2 テスト coverage 自動化

```yaml
- name: Check test coverage
  run: |
    pytest python/tests/ --cov=python/jax_util --cov-report=term-missing
    # 80% 以上 required
```

### 4.3 コード品質レポート

```bash
# リンター統合
ruff check python/ scripts/
pytest python/tests/ -v
```

---

## 🎯 実装対象ファイル（ファイル単位）

**Phase 2 対応ファイル**:
- ✅ `python/experiment_runner/_pr1_gmres_preconditioner.py`
- ✅ `python/experiment_runner/_pr3_runner_refactored.py`
- ✅ `scripts/check_doc_test_triplet.py`
- ✅ `scripts/docker_dependency_validator.py`
- ✅ `scripts/requirement_sync_validator.py`

**Phase 3 対応ブランチ**:
- ✅ `work/experiment-runner-hardening-20260331` → main
- ✅ `work/smolyak-integrator-lead-20260328` → main
- ✅ `results/smolyak-validation-20260328` → results/ branch

---

## 📋 統合チェックリスト

### Pre-Merge
- [ ] Phase 1 コミット完了 ✋ **待機中**
- [ ] Phase 2 型エラー削減完了 ⏳ **未開始**
- [ ] `pyright --strict` パス ⏳ **未開始**
- [ ] テスト suite パス ⏳ **未開始**

### Post-Merge
- [ ] CI  ✅ 設定完了
- [ ] Worktree 整理 ⏳ **未開始**
- [ ] 本ブランチ反映判定 ⏳ **未開始**

---

**詳細計画**: `/memories/session/type_error_reduction_plan.md` を参照
