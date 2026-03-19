# 修正ロードマップ（補足）

**作成日**： 2026-03-19
**ステータス**： worktree 内実装完成に向けた補足
**前提**： このworktreeはmain置き換え前提のため、main との整合性は不要

---

## I. 現在の実装状況の正確な評価

### 1. 既に完成している部分

#### GradientBackprop（sequential_train.py）
- `__call__` メソッド：層パラメータ更新ロジック ✓
- `buildoptim` メソッド：局所最適化問題構築 ✓
- `update` メソッド：勾配適用 ✓
- 実装ロジック自体は正しい

#### GradientTrainer（sequential_train.py）
- `__call__` メソッド：`lax.scan` で複数層を逐次更新 ✓
- `train_step` メソッド：単一ステップのラッパー ✓
- 構造は適切

#### nn_trainer_protocols.py
- `SingleLayerBackprop` Protocol：署名定義済み ✓
- `IncrementalTrainer` Protocol：署名定義済み ✓

### 2. 未完成/バグ部分

#### 問題 1： PyTreeOptim クラス の欠落（🔴 緊急）

**症状**:
```python
# sequential_train.py line 60, 76
new_obj = PyTreeOptim(objective=new_loss)
layer_optim = PyTreeOptim(objective=layer_objective)
```

**原因**： 旧版 main との diff で削除されたが、sequential_train.py では参照されたまま

**対応**:
```python
# protocols.py に追加（dataclass として）

from dataclasses import dataclass

@dataclass(frozen=True)
class PyTreeOptim:
    """PyTree パラメータ向けの最適化問題実装"""
    objective: Callable[[Params], Scalar]
```

**配置**： `protocols.py` に追加するか、`sequential_train.py` 内で定義するか、いずれでも OK

#### 問題 2： model.layer vs model.layers（🔴 緊急）

**症状** (sequential_train.py, GradientTrainer.__call__, line 112):
```python
lambda carry, layer_trainer: layer_trainer(layer=model.layer, cache=carry, obj=optim)
```

**問題**： neuralnetwork.py を確認すると、NeuralNetwork のフィールドは `layers` （複数形）

**対応のオプション**:

**オプション A**： model.layers の最初の1層だけ処理する
```python
lambda carry, layer_trainer: layer_trainer(layer=model.layers[0], ...)
```

**オプション B**： layers をすべて処理する（lax.scan nested）
```python
def update_layer(layer, layer_trainer):
    return layer_trainer(layer=layer, cache=carry, obj=optim)

new_layers = [update_layer(layer, trainer) for layer, trainer in zip(model.layers, self.layer_trainers)]
```

**推奨**： どの意図か確認してから決定

#### 問題 3: lax.scan の lambda シグネチャ（🟡 重要）

**現在のコード**:
```python
new_model, aux = lax.scan(
    lambda carry, layer_trainer: layer_trainer(...),
    cache,
    self.layer_trainers,
)
```

**問題**： `layer_trainer` を呼び出すだけだとうまくいかない可能性
- `layer_trainer` は `SingleLayerBackprop` Protocol
- `__call__` が `layer` を期待するが、loop では `layer_trainer` だけ渡される
- layer が unpack されていない

**対応案**:
```python
def step_fn(cache, layer_trainer):
    carry, ctx = cache
    # layer_trainer が持つ layer を何らか処理
    # または layer_trainer.__call__ を適切に呼び出す
    return layer_trainer(carry, ctx), None  # または適切な戻り値

new_model, _ = lax.scan(step_fn, cache, self.layer_trainers)
```

#### 問題 4： nn_trainer_protocols.py の未使用インポート（🟢 低）

**症状**： pyright で以下の warning
```
Import "TypeAlias" is not accessed
Import "PyTree" is not accessed
Import "Array" is not accessed
... (その他)
```

**対応**： Protocol ファイルなので、実装では使わない import もある程度は許容。ただし整理するなら：

```python
# 必要なもののみ残す
from typing import Any, Protocol, Tuple

from ..base import OptimizationProblem, OptimizationState

from .protocols import (
    Params,
    Static,
    Ctx,
    Carry,
    NeuralNetworkLayer,
    Aux,
    BackpropState,
    PyTreeOptimizationProblem,
    PyTreeOptimizationState,
)

from .neuralnetwork import NeuralNetwork
```

---

## II. 優先度別修正計画

### Phase 0： 即座対応（構文エラー除去）

**0-1. PyTreeOptim をどこに定義するか決定**

- **案 A**： protocols.py に dataclass で定義
  - 利点：Protocol と実装が同じファイル
  - 欠点：protocols.py が protocol だけのファイル ではなくなる

- **案 B**： sequential_train.py の先頭で定義
  - 利点：sequential_train.py が self-contained
  - 欠点：他のモジュールから import しづらい

**推奨**： 案 A（protocols.py）

**0-2. model.layer vs model.layers の確認**

```python
# neuralnetwork.py を確認
class NeuralNetwork(eqx.Module):
    layers: Tuple[NeuralNetworkLayer, ...]  # ← 複数形
```

**対応**： 設計意図を確認して修正

**0-3. lax.scan lambda の修正**

設計意図を確認した後、適切なシグネチャに修正

---

### Phase 1： PyTreeOptim 実装（0.5日）

```python
# protocols.py に追加

from dataclasses import dataclass

@dataclass(frozen=True)
class PyTreeOptim:
    """PyTree パラメータに対する最適化問題の実装。

    Attributes:
        objective: Callable[[Params], Scalar]
            パラメータから損失値を計算する関数
    """
    objective: Callable[[Params], Scalar]
```

**チェック項目**:
- [ ] import 可能 (`from protocols import PyTreeOptim`)
- [ ] sequential_train.py で使用可能
- [ ] pyright エラーなし

---

### Phase 2： model.layers 修正（0.5日）

**確認すべき事項**:
1. NeuralNetwork の実装（layer vs layers）
2. GradientTrainer の意図（全層処理か、最初の層だけか）
3. forward_with_cache との関係

**修正例**:
```python
# 現在
lambda carry, layer_trainer: layer_trainer(layer=model.layer, cache=carry, obj=optim)

# 修正案 1：全層処理
def update_with_trainer(layer, trainer):
    result, _, _, _ = trainer(layer=layer, cache=carry, obj=optim, backprop_state=None)
    return result, result

# または修正案 2：最初の層だけ
lambda carry, layer_trainer: layer_trainer(layer=model.layers[0], ...)
```

---

### Phase 3: lax.scan lambda の修正（1日）

lax.scan の carry と動作ロジックを確認した後：

```python
def step_fn(carry, layer_trainer):
    """層別学習の1ステップ"""
    # carry: (Carry, Ctx) または更新後の層
    # layer_trainer: SingleLayerBackprop

    # 適切な処理をここに記述
    # return: new_carry, output
    pass

new_model, _ = lax.scan(step_fn, initial_carry, self.layer_trainers)
```

---

### Phase 4： テスト・整合性確認（1-2日）

```python
# 基本的なテスト

def test_gradient_backprop_instantiation():
    """GradientBackprop の生成が可能か"""
    backprop = GradientBackprop(optstate=None)
    assert backprop is not None

def test_pyree_optim_creation():
    """PyTreeOptim の生成が可能か"""
    def dummy_loss(params):
        return jnp.zeros(())

    optim = PyTreeOptim(objective=dummy_loss)
    assert optim.objective is not None

def test_sequential_train_import():
    """sequential_train モジュール全体が import 可能か"""
    from jax_util.neuralnetwork.sequential_train import GradientBackprop, GradientTrainer
    assert GradientBackprop is not None
    assert GradientTrainer is not None
```

---

## III. 実装の意図に関する質問

以下を明確にすると修正が確実になります：

1. **layer vs layers**
   - GradientTrainer は全層を処理するのか、それとも最初の層だけ？

2. **lax.scan の carry 管理**
   - carry は層の出力ですか、それとも層そのものですか？
   - 層は更新されるべきですか、それとも各ステップで独立ですか？

3. **SingleLayerBackprop の使用方法**
   - Protocol の `backprop_state` パラメータは必須か、None で OK か？

4. **PyTreeOptim の実装位置**
   - protocols.py に混ぜるのか、sequential_train.py で定義するのか

---

## IV. 完了基準（最小限）

修正完了時に以下を確認：

```bash
# インポート可能か
python3 -c "from jax_util.neuralnetwork.sequential_train import GradientBackprop, GradientTrainer; print('OK')"

# 型チェック通過か
pyright python/jax_util/neuralnetwork/sequential_train.py
# → 0 errors, 0 critical warnings

# Protocol 不整合がないか（警告でも OK、エラーのみ対応）
pyright python/jax_util/neuralnetwork/nn_trainer_protocols.py
```

---

## V. 本分析との整合性

元の分析資料（PROJECT_SCOPE_ANALYSIS など）はほぼそのまま有効。修正が必要な部分は：

1. **PyTreeOptim の実装方法**：設計で未決定だった部分を実装
2. **lax.scan の詳細**：実装に合わせた docstring 追加
3. **model.layer の修正**：単純な typo 修正

実装の方向性・考え方は変わってません。

