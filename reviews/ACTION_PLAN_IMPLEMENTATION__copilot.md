# 実行計画書（Action Plan）

**作成日**: 2026-03-18  
**対象 Worktree**: `nn-develop-20260318`  
**分析ベース**: PROJECT_SCOPE_ANALYSIS, DETAILED_DESIGN_NEURALNETWORK  
**ステータス**: 実装前の詳細計画（変更未実施）  

---

## 1. 実装タイムラインと優先度

### 全体工期：約 1 週間（5-7 営業日）

```
├─ Phase 1: 型整備          (0.5-1 日)    [緊急]
├─ Phase 2: sequential_train.py 修正 (1-1.5 日)  [緊急]
├─ Phase 3: Docstring 充実  (1-1.5 日)    [重要]
├─ Phase 4: テスト拡充      (2-2.5 日)    [重要]
├─ Phase 5: ドキュメント更新 (0.5-1 日)    [中]
└─ Phase 6: 統合検証        (1-2 日)      [中]
```

---

## 2. Day-by-Day 実装スケジュール

### Day 1: 型定義・基盤整備

**目標**: import エラーをすべて解決  
**担当**:  
**成果物**: 修正済み protocols.py, sequential_train.py がインポート可能

#### 午前（タスク 1-1）

**Task 1-1.1: protocols.py に LossFn, TrainingAux, LayerBackpropState を追加**

```python
# 追加する内容
from dataclasses import dataclass

# TypeAlias
LossFn: TypeAlias = Callable[[Params, Matrix], Scalar]

# dataclass
@dataclass(frozen=True)
class TrainingAux:
    loss: Scalar
    grad_norm: Scalar | None = None
    num_updates: int = 0

@dataclass(frozen=True)
class LayerBackpropState:
    grad_carry: Carry | None = None
    upper_grad: Matrix | None = None

# 既存の Aux / BackpropState を修正
Aux: TypeAlias = TrainingAux | None
BackpropState: TypeAlias = LayerBackpropState | None
```

**確認**:
- [ ] pyright で import エラーなし
- [ ] test_neuralnetwork_*.py が import 可能

#### 午後（タスク 1-2）

**Task 1-2.1: PyTreeOptim の定義位置を確定し、実装**

**確認事項**:
- functional.py に OptimizationProblem あるか？
- sequential_train.py が何を期待しているか？

**2 つの選択肢**:

**オプション A**: protocols.py に dataclass で追加
```python
@dataclass(frozen=True)
class PyTreeOptim:
    objective: Callable[[Params], Scalar]
```

**オプション B**: functional.py からインポート（既存の場合）
```python
# functional.py に既存の実装があるなら
from ..functional import PyTreeOptim
```

**推奨**: **オプション B** → functional.py を確認し、無ければ **オプション A** で protocols.py に追加

**確認**:
- [ ] `from jax_util.neuralnetwork import PyTreeOptim` が成功
- [ ] sequential_train.py で `PyTreeOptim(objective=...)` が実行時エラーなし

#### 夕方（タスク 1-3）

**Task 1-3.1: protocols.py を上記で修正した後、pyright で全体検査**

```bash
# 実行コマンド
pyright python/jax_util/neuralnetwork/

# 期待結果
# 0 errors, 0 warnings (または无关の警告のみ)
```

**修正が必要な場合**:
- [ ] import パス確認
- [ ] 型注釈の不一致確認
- [ ] circular import 確認

---

### Day 2: sequential_train.py の修正と Protocol 準拠

**目標**: sequential_train.py が nn_trainer_protocols.py と型安全に一致  
**成果物**: 修正済み sequential_train.py、テスト実行可能

#### 午前（タスク 2-1）

**Task 2-1.1: nn_trainer_protocols.py を正本として確定**

**確認**:
- [ ] SingleLayerBackprop の Protocol を読む
- [ ] 現在の sequential_train.py の GradientBackprop と比較
- [ ] シグネチャの相違点リストアップ

**相違点の例**:
```python
# Protocol では
def __call__(
    self,
    layer: NeuralNetworkLayer,
    cache: Tuple[Carry, Ctx],
    optim: OptimizationProblem[NeuralNetwork],  # ← 型 A
    backprop_state: BackpropState,              # ← backprop_state パラメータ
)

# sequential_train では
def __call__(
    self,
    layer: NeuralNetworkLayer,
    cache: Tuple[Carry, Ctx],
    obj: PyTreeOptimizationProblem,  # ← 型が違う
    # ← backprop_state がない
)
```

**修正方針決定**:
- [ ] Protocol に準拠するか
- [ ] または、Protocol を実装に合わせるか（非推奨）
- [ ] コメントで乖離を明記するか（暫定的）

**推奨**: Protocol に準拠

#### 午後（タスク 2-2）

**Task 2-2.1: GradientBackprop.__call__ のシグネチャを修正**

```python
# Before
def __call__(
    self,
    layer: NeuralNetworkLayer,
    cache: Tuple[Carry, Ctx],
    obj: PyTreeOptimizationProblem,
) -> Tuple["SingleLayerBackprop", NeuralNetworkLayer, PyTreeOptimizationProblem, Aux]:

# After
def __call__(
    self,
    layer: NeuralNetworkLayer,
    cache: Tuple[Carry, Ctx],
    optim: PyTreeOptimizationProblem,  # 名前変更 obj → optim
    backprop_state: LayerBackpropState | None = None,  # 追加
) -> Tuple["GradientBackprop", NeuralNetworkLayer, PyTreeOptimizationProblem, Aux]:
```

**修正のポイント**:
- [ ] パラメータ名・型を Protocol と一致
- [ ] backprop_state パラメータ追加
- [ ] docstring も修正

#### 夕方（タスク 2-3）

**Task 2-3.1: 実装詳細の修正**

- [ ] `buildoptim()` メソッド内で PyTreeOptim インスタンス化
- [ ] `update()` メソッドで aux を正しく返却
- [ ] GradientBackprop インスタンス返却時に optstate を保藍

**確認**:
```bash
# import テスト
python -c "from jax_util.neuralnetwork.sequential_train import GradientBackprop; print('OK')"

# 型チェック
pyright python/jax_util/neuralnetwork/sequential_train.py
```

---

### Day 3-4: Docstring 充実・テスト準備

**目標**: 全公開 API に詳細 docstring、テスト実行可能状態  
**成果物**: Docstring 追加済みコード、テスト suite 拡張

#### Day 3: Docstring 追加（分割実行）

**午前（タスク 3-1）**: neuralnetwork.py, layer_utils.py の docstring

```python
# neuralnetwork.py
- build_neuralnetwork()
- forward_with_cache()
- state_initializer()

# layer_utils.py
- StandardNNLayer (クラス docstring)
- IcnnLayer (クラス docstring)
- StandardNNLayer._factory()
- IcnnLayer._factory()
- module_to_vector()
- vector_to_module()
```

**各 docstring 形式**:
```python
def func(args) -> ReturnType:
    """1行のサマリー。
    
    詳細説明（複数行可）。
    
    Args:
        arg1: 説明と型
        arg2: 説明と型
    
    Returns:
        返却値の説明
    
    Raises:
        ExceptionType: 例外条件
    
    Example:
        >>> result = func(...)
        >>> print(result)
    """
```

**午後（タスク 3-2）**: train.py, sequential_train.py, nn_trainer_protocols.py の docstring

- [ ] train_step()
- [ ] train_loop()
- [ ] SingleLayerBackprop (Protocol docstring)
- [ ] IncrementalTrainer (Protocol docstring)
- [ ] GradientBackprop
- [ ] GradientTrainer

#### Day 4: テスト拡張

**午前（タスク 4-1）**: sequential_train.py テスト新規作成

```python
# test_sequential_train.py
- test_gradient_backprop_creation()
- test_gradient_backprop_single_update()
- test_gradient_trainer_initialization()
```

**午後（タスク 4-2）**: 既存テストの拡張

```python
# test_layer_utils.py (新規)
- test_module_to_vector()
- test_vector_to_module()
- test_module_conversion_reversibility()

# test_neuralnetwork_forward.py (拡張)
- test_forward_with_cache()
- test_forward_with_cache_vs_normal()

# test_neuralnetwork_train.py (拡張)
- test_train_loop()
- test_train_with_metrics()
```

**確認**:
```bash
pytest python/tests/neuralnetwork/ -v
# すべて pass
```

---

### Day 5: ドキュメント更新

**目標**: WORKTREE_SCOPE.md, 18_neuralnetwork.md の完成  
**成果物**: 更新済みドキュメント

#### 午前（タスク 5-1）

**WORKTREE_SCOPE.md の完成化**:

```markdown
# WORKTREE_SCOPE

## Worktree Summary
- Branch: nn-develop-20260318
- Worktree path: ./.worktrees/nn-develop-20260318
- Purpose: ニューラルネットワーク実装モジュールの設計・実装・テスト
- Owner: (担当者名)

## Editable Directories
- python/jax_util/neuralnetwork/
- python/tests/neuralnetwork/
- documents/conventions/python/18_neuralnetwork.md
- reviews/

## Read-Only Directories
- python/jax_util/base/
- python/jax_util/solvers/
- python/jax_util/optimizers/
- python/jax_util/hlo/
- python/jax_util/functional/

## Required References
- documents/conventions/python/18_neuralnetwork.md
- documents/design/base_components.md
- pyproject.toml

## Required Checks Before Commit
- pyright python/jax_util/neuralnetwork/
- pytest python/tests/neuralnetwork/ -v
- markdownlint (変更した Markdown)
```

#### 午後（タスク 5-2）

**18_neuralnetwork.md の拡張セクション追加**:

```markdown
### 追加: Layer-wise Learning

現在、sequential_train.py で層別逐次学習の実験を進行中。
- SingleLayerBackprop / IncrementalTrainer Protocol は nn_trainer_protocols.py で定義
- GradientBackprop / GradientTrainer 実装は実験段階
- 正式 API ではなく、設計検証目的

### 追加: forward_with_cache の用途

順伝播で中間 Carry をキャッシュ。
- 層別学習での逆伝播で、各層の出力を再利用
- compute_with_cache + grad(loss) により、勾配計算を効率化

### 追加: Ctx/Carry の設計

#### Standard NN
- Ctx: 不使用（StandardCtx は空）
- Carry: 層出力 z のみ

#### ICNN
- Ctx: 入力 x をキャッシュ
- Carry: 層出力 z のみ
```

---

### Day 6: 統合検証・最終確認

**目標**: 全モジュールが統合可能、テスト通過、ドキュメント完成  
**成果物**: 最終検証レポート

#### 午前（タスク 6-1）

**統合テスト実行**:

```bash
# 1. 全テスト実行
pytest python/tests/neuralnetwork/ -v

# 2. 型チェック
pyright python/jax_util/neuralnetwork/

# 3. import テスト
python -c "
from jax_util.neuralnetwork import (
    NeuralNetwork,
    build_neuralnetwork,
    train_step,
    train_loop,
)
print('All imports OK')
"

# 4. sequential_train import テスト
python -c "
from jax_util.neuralnetwork.sequential_train import (
    GradientBackprop,
    GradientTrainer,
)
print('Sequential imports OK')
"
```

#### 午後（タスク 6-2）

**最終ドキュメント確認**:

- [ ] PROJECT_SCOPE_ANALYSIS.md の内容と実装が一致
- [ ] DETAILED_DESIGN_NEURALNETWORK.md のチェックリストすべてクリア
- [ ] WORKTREE_SCOPE.md が完成
- [ ] 18_neuralnetwork.md が最新
- [ ] 全公開関数に docstring

**最終確認コマンド**:

```bash
# markdownlint
markdownlint documents/conventions/python/18_neuralnetwork.md

# すべての実装が規約に従っているか
grep -r "TODO\|FIXME\|XXX" python/jax_util/neuralnetwork/ || echo "No TODOs"
```

---

## 3. タスク依存関係図

```
Day 1
├─ Task 1-1: protocols.py 修正 (LossFn, Aux, etc.)
├─ Task 1-2: PyTreeOptim 定義
└─ Task 1-3: pyright 検査

      ↓ (Day 1 完了後)

Day 2
├─ Task 2-1: nn_trainer_protocols.py 確認
├─ Task 2-2: sequential_train.py シグネチャ修正
└─ Task 2-3: sequential_train.py 実装修正

      ↓ (Day 2 完了後)

Day 3-4
├─ Task 3-1~3-2: Docstring 追加
└─ Task 4-1~4-2: テスト拡張

      ↓ (Day 4 完了後)

Day 5
├─ Task 5-1: WORKTREE_SCOPE.md
└─ Task 5-2: 18_neuralnetwork.md 拡張

      ↓ (Day 5 完了後)

Day 6
├─ Task 6-1: 統合テスト
└─ Task 6-2: 最終ドキュメント確認
```

---

## 4. 成功基準（Definition of Done）

### コード面

- [ ] `pyright python/jax_util/neuralnetwork/` が 0 errors
- [ ] `pytest python/tests/neuralnetwork/ -v` が全 pass
- [ ] Sequential train・layer utils のテスト coverage が 70% 以上
- [ ] import エラーなし、実行時エラーなし

### ドキュメント面

- [ ] WORKTREE_SCOPE.md が完成（テンプレート→具体化）
- [ ] 全公開関数に詳細 docstring
- [ ] 18_neuralnetwork.md が layer-wise learning 情報を含む
- [ ] markdownlint で規約チェック通過

### 設計面

- [ ] sequential_train.py が nn_trainer_protocols.py と型安全に一致
- [ ] Protocol/実装の乖離が明記されている（必要な場合）
- [ ] 依存関係が明確：functional.py → sequential_train の方向確定

### 検収

- [ ] PROJECT_SCOPE_ANALYSIS で指摘された問題がすべて解決
- [ ] DETAILED_DESIGN_NEURALNETWORK のチェックリスト完了
- [ ] reviews/ に最終レポート（IMPLEMENTATION_COMPLETION）作成

---

## 5. リスク管理と対応

| リスク | 確率 | 影響 | 対応 |
| --- | --- | --- | --- |
| functional.py の PyTreeOptim が見つからない | 中 | 高 | protocols.py に自分で実装 |
| Protocol シグネチャが複雑すぎる | 低 | 中 | 簡化版 Backprop を別に用意 |
| テスト実行時 JAX エラー | 中 | 中 | キー分割、形状確認 |
| Docstring の品質が低い | 低 | 低 | レビュー後に修正 |
| ドキュメント作成に時間超過 | 中 | 低 | テンプレートを活用 |

---

## 6. コミット単位の推奨

**Day 1 完了時**:
```bash
git commit -m "WIP: protocols.py に型定義を充実（LossFn, Aux, BackpropState）"
```

**Day 2 完了時**:
```bash
git commit -m "REFACTOR: sequential_train.py を nn_trainer_protocols に準拠"
```

**Day 3-4 完了時**:
```bash
git commit -m "DOCS: Docstring 充実、テスト拡張"
```

**Day 5 完了時**:
```bash
git commit -m "DOCS: WORKTREE_SCOPE, 18_neuralnetwork 更新"
```

**Day 6 完了時**:
```bash
git commit -m "RELEASE: neuralnetwork モジュール整備完了"
```

---

## 7. レビューポイント

実装完了時にレビューする項目：

- [ ] **型安全性**: `pyright --strict` で通るか（最低限）
- [ ] **テスト可読性**: テスト関数名が目的を明示しているか
- [ ] **docstring 完全性**: Args, Returns, Raises, Example すべての主要関数に
- [ ] **設計一貫性**: name convention, error handling, logging が規約と一致
- [ ] **依存関係**: 循環参照がないか、import 順序は正しいか

---

**このアクションプランに従うことで、5-7 営業日で neuralnetwork モジュール整備が完了し、  
implementation ready な状態に持ち込めます。**

