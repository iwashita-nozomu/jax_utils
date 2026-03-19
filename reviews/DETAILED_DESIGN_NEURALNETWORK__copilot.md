# 詳細設計書：ニューラルネットワークモジュール整備

**作成日**: 2026-03-18  
**ステータス**: 実装予定前の詳細設計  
**対象**: `python/jax_util/neuralnetwork/` 整備計画  

---

## 1. 設計原則

### 1.1 基本姿勢

| 原則 | 説明 | 実装への影響 |
| --- | --- | --- |
| **段階性** | 実験段階→安定段階への段階性を意識。今は安定の要件緩和 | API は暫定、ただし互換性破壊は通知 |
| **型安全性** | pyright strict mode で検査可能な状態を保つ | Protocol と実装の不一致検出 |
| **JAX主義** | `jax.lax` 制御フロー優先、Python ループ最小化 | `lax.scan` で逐次学習実装 |
| **Equinox 統合** | PyTree Module として設計、自動微分と親和性重視 | eqx.partition/combine で層操作 |

### 1.2 命名規約の適用

```
层型:         StandardNNLayer, IcnnLayer
状態型:       StandardCarry, StandardCtx
工厂関数:     standardNN_layer_factory, icnn_layer_factory
トレーナー:   GradientBackprop, GradientTrainer
最適化問題:   PyTreeOptimizationProblem (Protocol)
```

**ルール**:
- `Layer` で終わる → 具体層実装
- `Ctx` / `Carry` → 層固有の状態
- `*_factory` → ランダム鍵を消費する生成関数
- Protocol 名は末尾に `Protocol` ⇒ 不要（18_neuralnetwork.md の設定に従う）

---

## 2. 型システムの詳細設計

### 2.1 修正が必要な型定義

#### 2.1.1 LossFn の定義

**現在**: 定義されていない  
**提案**:

```python
# protocols.py に追加
from typing import Callable
from jaxtyping import Array

# 損失関数の型
LossFn: TypeAlias = Callable[[Params, Matrix], Scalar]
    # loss_fn(params, batch) → 損失スカラー値
```

**根拠**:
- train.py で既に使用されている
- 関数シグネチャの統一

#### 2.1.2 Aux の具体化

**現在**: 空の Protocol  
**提案**:

```python
# protocols.py

from dataclasses import dataclass

@dataclass(frozen=True)
class TrainingAux:
    """訓練ステップの補助情報"""
    loss: Scalar            # 損失値（必須）
    grad_norm: Scalar | None = None  # 勾配ノルム（層別学習用）
    num_updates: int = 0    # 更新回数

# Aux として使用を推奨
Aux: TypeAlias = TrainingAux | None
```

**理由**:
- sequential_train では現在 None 返却
- 将来、メトリクス追加時の拡張性確保
- frozen=True で JAX 互換性維持

#### 2.1.3 BackpropState の詳細化

**現在**: 空の Protocol  
**提案**:

```python
# protocols.py

@dataclass(frozen=True)
class LayerBackpropState:
    """層逆伝播時の状態情報"""
    grad_carry: Carry | None = None  # 勾配Carry
    upper_grad: Matrix | None = None  # 上位層からの勾配
    
class BackpropState(Protocol):
    """上位レイヤーからの伝搬情報"""
    grad_carry: Carry | None
    upper_grad: Matrix | None
```

**用途**:
- sequential_train.py での層別逆伝播の情報伝搬
- forward_with_cache とペアで動作

---

### 2.2 新規追加が必要な型

#### 2.2.1 PyTreeOptim 実装クラス

**現在の問題**:
```python
# sequential_train.py line 48
new_obj = PyTreeOptim(objective=new_loss)  # ← undefined
```

**提案 A: protocols.py に追加（シンプル版）**

```python
# protocols.py

@dataclass(frozen=True)
class SimpleOptimizationProblem:
    """パラメータ空間上の最適化問題の簡易実装"""
    objective: Callable[[Params], Scalar]
    # objective(params) → loss_scalar
    
    def objective(self, params: Params) -> Scalar:
        """目的関数を評価"""
        return self.objective(params)

# Protocol の実装としても使用
PyTreeOptim: TypeAlias = SimpleOptimizationProblem
```

**提案 B: functional.py に定義（拡張性重視）**

```python
# functional.py あるいは neuralnetwork.py

from dataclasses import dataclass

@dataclass(frozen=True)  
class PyTreeOptim(PyTreeOptimizationProblem):
    """PyTree パラメータ向け最適化問題"""
    objective: Callable[[Params], Scalar]
    # functional.OptimizationProblem の Protocol を満たす
```

**推奨**: **提案 B**（functional.py への配置）
- 理由: 他の最適化関連 Protocol と統一性
- import: `from ..functional import PyTreeOptim`

---

## 3. モジュール別の詳細設計

### 3.1 protocols.py（型エイリアス・Protocol 集約）

#### 現状

```python
Params: TypeAlias = PyTree[Array]
Static: TypeAlias = PyTree[Any]

Ctx (Protocol): 固定状態
Carry (Protocol): z: Matrix
NeuralNetworkLayer (abstract): __call__(carry, ctx) → Carry

Aux (Protocol): 空
BackpropState (Protocol): 空

PyTreeOptimizationProblem (Protocol)
ConstrainedPyTreeOptimizationProblem (Protocol)
PyTreeOptimizationState (Protocol)
ConstrainedPyTreeOptimizationState (Protocol)
```

#### 修正提案

```python
# 追加 1: LossFn
LossFn: TypeAlias = Callable[[Params, Matrix], Scalar]

# 追加 2: TrainingAux dataclass
@dataclass(frozen=True)
class TrainingAux:
    loss: Scalar
    grad_norm: Scalar | None = None
    num_updates: int = 0

# 追加 3: LayerBackpropState
@dataclass(frozen=True)
class LayerBackpropState:
    grad_carry: Carry | None = None
    upper_grad: Matrix | None = None

# 追加 4: Aux / BackpropState を具体型で再定義
Aux: TypeAlias = TrainingAux | None
BackpropState: TypeAlias = LayerBackpropState | None

# 最後に __all__ を更新
__all__ = [
    "Params", "Static",
    "Ctx", "Carry", "NeuralNetworkLayer",
    "LossFn",  # 新規
    "Aux", "TrainingAux",  # 新規
    "BackpropState", "LayerBackpropState",  # 新規
    "PyTreeOptimizationProblem",
    "ConstrainedPyTreeOptimizationProblem",
    "PyTreeOptimizationState",
    "ConstrainedPyTreeOptimizationState",
]
```

#### Docstring の追加

```python
"""ニューラルネットワークモジュールの型定義とプロトコル。

型エイリアス:
    Params: パラメータ PyTree
    Static: 静的情報 PyTree
    
Protocol:
    Ctx: 層の固定入力（network_type 依存）
    Carry: 層の状態（z: Matrix を必須フィールド）
    NeuralNetworkLayer: 層の抽象基底
    
最適化問題:
    PyTreeOptimizationProblem: パラメータ空間上の最適化問題
    
補助情報:
    Aux: 訓練ステップの補助情報（メトリクス等）
    BackpropState: 逆伝播時の層間情報伝搬
"""
```

---

### 3.2 neuralnetwork.py（順伝播・ファクトリ）

#### 現状

```python
class NeuralNetwork(eqx.Module):
    layers: Tuple[NeuralNetworkLayer, ...]
    network_type: str  # "standard" or "icnn"
    layer_sizes: Tuple[int, ...]

def __call__(x: Matrix) -> Matrix
def state_initializer(network_type, x, layer_sizes) → (Ctx, Carry)
def build_neuralnetwork(...) → NeuralNetwork
def forward_with_cache(x, network) → (Matrix, Tuple[Carry], Ctx)
```

#### Docstring 充実案

```python
def build_neuralnetwork(
    network_type: str,
    layer_sizes: Tuple[int, ...],
    activation: str,
    random_key: Scalar,
) -> NeuralNetwork:
    """ニューラルネットワークの構築。
    
    既定の層サイズと活性化関数から、NeuralNetwork インスタンスを生成します。
    
    Args:
        network_type: "standard" または "icnn"
            - "standard": 全結合層（w @ z + b）
            - "icnn": 入力正則化ネットワーク（非負重み利用）
        
        layer_sizes: 各層のユニット数
            例: (2, 3, 1) ⇒ 入力2 → 隠れ層3 → 出力1
        
        activation: 活性化関数
            許可値: "relu", "tanh", "sigmoid", "softplus", "identity"
        
        random_key: jax.random.PRNGKey
            He初期化に使用
    
    Returns:
        NeuralNetwork インスタンス
    
    Raises:
        ValueError: network_type が未知、activation が未知、
                   または layer_sizes が不正な場合
    
    Note:
        - は実際のバッチ軸別に (feature, batch) 形式を想定
        - モデルは eqx.Module なので、自動微分可能
        - forward 実行時に層順序で逐次適用される
    
    Example:
        >>> key = jax.random.PRNGKey(0)
        >>> model = build_neuralnetwork(
        ...     network_type="standard",
        ...     layer_sizes=(2, 8, 1),
        ...     activation="relu",
        ...     random_key=key,
        ... )
        >>> x = jnp.ones((2, 5))  # (2 features, 5 batch)
        >>> y = model(x)  # (1, 5)
    """
```

```python
def forward_with_cache(
    x: Matrix,
    network: NeuralNetwork,
) -> Tuple[Matrix, Tuple[Carry, ...], Ctx]:
    """順伝播+キャッシュ。
    
    各層の中間出力（Carry）を保存しながら順伝播。
    層別学習（sequential_train.py）で逆伝播時の参照に使用。
    
    Args:
        x: 入力 (n_features, batch)
        network: NeuralNetwork インスタンス
    
    Returns:
        output: 最終出力 (n_out, batch)
        carries: 各層の Carry タプル
                 [layer0_carry, layer1_carry, ..., final_carry]
        context: Ctx インスタンス（層固有）
    
    Note:
        - メモリ効率性のため、スキャン操作を利用
        - 逆伝播が必要ない場合は、通常の __call__ でよい
    
    Example:
        >>> output, carries, ctx = forward_with_cache(x, model)
        >>> for carry in carries:
        ...     print(carry.z.shape)  # (hidden_size, batch)
    """
```

---

### 3.3 layer_utils.py（層実装・変換ユーティリティ）

#### Docstring 充実

```python
def standardNN_layer_factory(
    input_dim: int,
    output_dim: int,
    activation: Callable[[Matrix], Matrix],
    random_key: Scalar,
) -> Tuple[StandardNNLayer, Scalar]:
    """Standard ニューラルネットワーク層の生成。
    
    Args:
        input_dim: 入力次元
        output_dim: 出力次元
        activation: 活性化関数
        random_key: 乱数種（He初期化用）
    
    Returns:
        (StandardNNLayer, new_key)
    
    初期化:
        W: Normal(0, 1) * sqrt(2 / input_dim)  [He初期化]
        b: zeros
    
    操作:
        z_new = activation(W @ z + b)
    """
```

```python
def module_to_vector(module: M) -> Tuple[Vector, RebuildState[M]]:
    """PyTree Module → flat Vector + 復元情報。
    
    層パラメータをフラットなベクトルに変換。
    層別学習で勾配計算効率化に使用。
    
    Args:
        module: 任意の eqx.Module
    
    Returns:
        (flat_vector, rebuild_state)
        
    rebuild_state から復元可能:
        mod_reconstructed = vector_to_module(new_vector, rebuild_state)
    
    Note:
        - 可微分配列のみを抽出・扁平化
        - 静的部分（関数など）は rebuild_state に保存
    """
```

#### クラス docstring

```python
class StandardNNLayer(eqx.Module):
    """標準全結合層。
    
    操作: z' = activation(W @ z + b)
    
    Attributes:
        W: 重み行列 (output_dim, input_dim)
        b: バイアスベクトル (output_dim,)
        activation: 活性化関数 callable
    """
    
class IcnnLayer(eqx.Module):
    """入力正則化ネットワーク層。
    
    操作: z' = activation(W @ z + W_x @ x + b)
        （W, W_x は非負重み）
    
    Attributes:
        W: 隠れ層重み (output_dim, input_dim)、非負
        W_x: 入力直結重み (output_dim, x_dim)
        b: バイアス (output_dim,)
        activation: 活性化関数
    
    特徴:
        - 入力を保持し、各層で直結
        - 非負重み制約で単調性保証
        - 凸最適化問題での利用を想定
    """
```

---

### 3.4 train.py（標準学習ループ）

#### 現状足りない部分

```python
def train_step(
    params: Params,
    batch: Matrix,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_fn: LossFn,
) -> tuple[Params, optax.OptState, Dict[str, Scalar]]:
```

#### 改善案

```python
def train_step(
    params: Params,
    batch: Matrix,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    loss_fn: LossFn,
) -> Tuple[Params, optax.OptState, Dict[str, Scalar]]:
    """1ステップの訓練。
    
    Args:
        params: 学習対象パラメータ (PyTree)
        batch: バッチデータ (訓練向け形式に合わせる)
        optimizer: optax GradientTransformation
        opt_state: オプティマイザ状態
        loss_fn: 損失関数 Callable[[Params, Matrix], Scalar]
    
    Returns:
        (new_params, new_opt_state, metrics)
        
    metrics: {"loss": scalar}
    
    処理フロー:
        1. loss_fn で損失と勾配を同時計算
        2. optax で勾配更新と パラメータ更新
        3. メトリクスを dict で返却
    
    Note:
        DEBUG ガード内で jax.debug.print でログ出力
    """
```

#### その他改善

現在の実装では typo が見える：

```python
# 現状 (list ではなく dict)
metrics = {"loss": loss_val}

# 将来拡張の余地
metrics = {
    "loss": loss_val,
    "param_update_norm": jnp.linalg.norm(updates),  # 将来追加可能
    "step": current_step,  # 将来追加可能
}
```

---

### 3.5 sequential_train.py（層別学習・実験）

#### 設計上の決定

**3.5.1 nn_trainer_protocols.py との整合**

```python
# nn_trainer_protocols.py の SingleLayerBackprop (正本)
class SingleLayerBackprop(Protocol):
    def __call__(
        self,
        layer: NeuralNetworkLayer,
        cache: Tuple[Carry, Ctx],
        optim: OptimizationProblem[NeuralNetwork],  # ← ここ
        backprop_state: BackpropState,
    ) -> Tuple["SingleLayerBackprop", NeuralNetworkLayer, PyTreeOptimizationProblem, Aux]:
        ...
```

**現在の sequential_train.py**:
```python
class GradientBackprop:
    def __call__(
        self,
        layer: NeuralNetworkLayer,
        cache,
        obj: PyTreeOptimizationProblem,  # ← 型が違う + backprop_state がない
    ) -> ...
```

**改善案**:

```python
# sequential_train.py を修正して Protocol に準拠させる

class GradientBackprop(eqx.Module):
    """層別逆伝播（Protocol 準拠版）"""
    optstate: PyTreeOptimizationState
    
    def __call__(
        self,
        layer: NeuralNetworkLayer,
        cache: Tuple[Carry, Ctx],
        optim: PyTreeOptimizationProblem,  # ← 統一（OptimizationProblem[Params] として）
        backprop_state: LayerBackpropState = LayerBackpropState(),  # ← 追加
    ) -> Tuple["GradientBackprop", NeuralNetworkLayer, PyTreeOptimizationProblem, Aux]:
        """層の勾配更新を実行。
        
        Args:
            layer: 対象層
            cache: (Carry, Ctx) タプル（キャッシュされた状態）
            optim: 各層の局所最適化問題
            backprop_state: 逆伝播情報（上位層からの勾配など）
        
        Returns:
            (GradientBackprop, updated_layer, new_optim, aux)
        """
        # ... 実装
```

#### 詳細実装方針

```python
# GradientBackprop.__call__ の流れ

def __call__(
    self, layer, cache, optim, backprop_state=None
) -> Tuple["GradientBackprop", NeuralNetworkLayer, PyTreeOptimizationProblem, Aux]:
    
    # 1. 層パラメータを分離
    param, static = eqx.partition(layer, eqx.is_inexact_array)
    
    # 2. 局所最適化問題を構築
    #    cache (Carry, Ctx) と optim から、各層の目的関数を作成
    layer_optim = self.buildoptim(layer, optim, cache)
    
    # 3. 勾配計算と更新
    new_param, new_optstate, aux = self.update(param, layer_optim)
    
    # 4. 層を再構築
    new_layer = eqx.combine(new_param, static)
    
    # 5. 次層への新しい目的関数を設定（forward の新値を反映）
    new_optim = update_optim_with_new_layer(new_layer, optim)
    
    return GradientBackprop(optstate=new_optstate), new_layer, new_optim, aux
```

---

### 3.6 nn_trainer_protocols.py（インターフェース確定）

#### Docstring 追加

```python
class SingleLayerBackprop(Protocol):
    """1層分の逆伝播・更新を行うインターフェース。
    
    層別逐次学習（layer-wise greedy learning）で、
    各層を順に最適化するための Protocol。
    
    Attributes:
        optstate: オプティマイザ状態（PyTreeOptimizationState）
    
    Methods:
        __call__: 層を更新し、新しい Backprop インスタンスを返す
        buildoptim: 層パラメータに対する局所最適化問題を構築
        update: パラメータを更新
    
    使用例:
        layer_backprops = [layer_backprop_0, layer_backprop_1, ...]
        for backprop in layer_backprops:
            backprop, layer, optim, aux = backprop(
                layer=current_layer,
                cache=current_cache,
                optim=current_optim,
                backprop_state=None,  # or LayerBackpropState(...)
            )
    """

class IncrementalTrainer(Protocol):
    """複数層を逐次学習するインターフェース。
    
    layer_wise_backprop インスタンスのタプルを持ち、
    モデル全体を層別に訓練する。
    
    Attributes:
        layer_trainers: Tuple[SingleLayerBackprop, ...]
    
    Methods:
        __call__: モデルを層別に訓練
    """
```

---

## 4. テスト設計

### 4.1 sequential_train.py のテスト

#### test_sequential_train.py（新規作成）

```python
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

from jax_util.neuralnetwork import build_neuralnetwork
from jax_util.neuralnetwork.sequential_train import GradientBackprop, GradientTrainer
from jax_util.neuralnetwork.protocols import PyTreeOptimizationProblem, LayerBackpropState

def test_gradient_backprop_creation():
    """GradientBackprop インスタンスの生成が可能か"""
    backprop = GradientBackprop(optstate=None)  # 簡略版
    assert backprop is not None

def test_gradient_backprop_single_step():
    """1層の GradientBackprop 更新が実行できるか"""
    # setup: 層、キャッシュ、最適化問題を構築
    # 実行: backprop(layer, cache, optim)
    # assert: 新しい層の loss が低下している
    pass

def test_gradient_trainer_integration():
    """GradientTrainer がモデル全体を訓練できるか"""
    pass
```

### 4.2 layer_utils テストの拡張

```python
def test_module_to_vector_standard_layer():
    """StandardNNLayer の module_to_vector / vector_to_module"""
    # layer → vector → layer' の復元が可能か
    pass

def test_module_to_vector_icnn_layer():
    """IcnnLayer の module_to_vector / vector_to_module"""
    pass

def test_module_conversion_with_modification():
    """変換後の vector を修正して layer に復元"""
    pass
```

### 4.3 forward_with_cache テスト

```python
def test_forward_with_cache_output_shape():
    """forward_with_cache が正しい形状で出力"""
    x = jnp.ones((2, 5))
    output, carries, ctx = forward_with_cache(x, model)
    
    assert output.shape == (1, 5)  # 最終出力
    assert len(carries) == len(model.layers)  # キャッシュ数が層数と一致
    
def test_forward_with_cache_vs_normal_forward():
    """forward_with_cache と通常 forward の一致性"""
    x = jnp.ones((2, 5))
    
    normal_output = model(x)
    cached_output, _, _ = forward_with_cache(x, model)
    
    assert jnp.allclose(normal_output, cached_output)
```

---

## 5. 実装チェックリスト

### Phase 1: 型定義完成（0.5-1日）

- [ ] **protocols.py 修正**
  - [ ] `LossFn: TypeAlias` 追加
  - [ ] `TrainingAux` dataclass 追加
  - [ ] `LayerBackpropState` dataclass 追加
  - [ ] `Aux`, `BackpropState` を具体型に変更
  - [ ] `__all__` に新規型を追加

- [ ] **PyTreeOptim 実装**
  - [ ] functional.py に `PyTreeOptim` class 追加（または protocols.py に dataclass 追加）
  - [ ] sequential_train.py のインポート確認

### Phase 2: sequential_train.py 修正（1-1.5日）

- [ ] **方針決定**
  - [ ] nn_trainer_protocols.py が正本と确认
  - [ ] GradientBackprop のシグネチャを Protocol に合わせるか決定

- [ ] **シグネチャ修正**
  - [ ] GradientBackprop.__call__ に backprop_state パラメータ追加
  - [ ] optim パラメータの型を統一
  
- [ ] **実装補完**
  - [ ] PyTreeOptim インスタンス化がエラーしないよう修正
  - [ ] update() メソッドの戻り値が aux を返すよう修正
  - [ ] 逆伝播処理の詳細実装（現在のコメントを展開）

### Phase 3: Docstring 充実（1-1.5日）

- [ ] **neuralnetwork.py**
  - [ ] `build_neuralnetwork()` に詳細 docstring
  - [ ] `forward_with_cache()` に詳細 docstring
  - [ ] `state_initializer()` に詳細 docstring

- [ ] **layer_utils.py**
  - [ ] StandardNNLayer / IcnnLayer にクラス docstring
  - [ ] `standardNN_layer_factory()` / `icnn_layer_factory()` に詳細 docstring
  - [ ] `module_to_vector()` / `vector_to_module()` に詳細 docstring

- [ ] **train.py**
  - [ ] `train_step()` に詳細 docstring
  - [ ] `train_loop()` に詳細 docstring

- [ ] **sequential_train.py / nn_trainer_protocols.py**
  - [ ] GradientBackprop / GradientTrainer に詳細 docstring
  - [ ] Protocol に使用例追加

### Phase 4: テスト拡充（2-2.5日）

- [ ] **test_sequential_train.py 新規作成**
  - [ ] GradientBackprop インスタンス生成テスト
  - [ ] 層別更新テスト
  - [ ] GradientTrainer 統合テスト

- [ ] **test_layer_utils.py 拡張**
  - [ ] module_to_vector / vector_to_module テスト
  - [ ] ICNN 層テスト
  - [ ] 変換の可逆性テスト

- [ ] **test_neuralnetwork.py 拡張**
  - [ ] forward_with_cache テスト
  - [ ] forward_with_cache vs 通常 forward の一致性

- [ ] **エラーハンドリングテスト**
  - [ ] 不正な layer_sizes
  - [ ] 不正な activation 名
  - [ ] バッチ軸が異なる入力

### Phase 5: ドキュメント更新（0.5-1日）

- [ ] **WORKTREE_SCOPE.md 完成化**
  - [ ] Branch名、目的、所有者を埋める
  - [ ] Editable/Read-only ディレクトリ明記
  - [ ] 必須チェックコマンド追加

- [ ] **18_neuralnetwork.md 拡張**
  - [ ] Layer-wise learning 実験状況
  - [ ] forward_with_cache 用途説明
  - [ ] Ctx/Carry 設計背景

### Phase 6: 統合テスト・検証（1-2日）

- [ ] **pytest 実行**
  - [ ] `pytest python/tests/neuralnetwork/ -v`
  - [ ] 全テスト通過確認

- [ ] **pyright 検査**
  - [ ] `pyright python/jax_util/neuralnetwork/`
  - [ ] エラー・警告がないか確認

- [ ] **コード review**
  - [ ] Protocol 準拠性確認
  - [ ] 依存関係の一貫性確認
  - [ ] 命名規約準拠性確認

---

## 6. 実装上の注意点

### 6.1 JAX・Equinox との相互作用

```python
# ❌ 避けるべき
layer.W = new_W  # In-place 修正は JAX と非互換

# ✓ 推奨
new_layer = eqx.combine(new_param, static)  # Equinox の rebuild
```

### 6.2 pyright strict mode への対応

```python
# ❌ Any を使いすぎない
def some_fn(x: Any) -> Any:
    ...

# ✓ 型を明記する
def some_fn(x: Params) -> Scalar:
    ...
```

### 6.3 ログ出力

```python
# ❌ 常に出力
print(f"Loss: {loss}")

# ✓ DEBUG ガード
if DEBUG:
    jax.debug.print("Loss: {loss}", loss=loss)
```

### 6.4 PRNGKey の扱い

```python
# ❌ 同じキーを使い回す
key = jax.random.PRNGKey(0)
W1 = jax.random.normal(key, shape=(3, 3))
W2 = jax.random.normal(key, shape=(4, 4))  # 同じキー NG

# ✓ キーを分割
key = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(key)
W1 = jax.random.normal(k1, shape=(3, 3))
W2 = jax.random.normal(k2, shape=(4, 4))
```

---

## 7. 完了基準

### 実装完了時の確認項目

- [ ] **型チェック通過**
  ```bash
  pyright python/jax_util/neuralnetwork
  # 0 errors
  ```

- [ ] **テスト通過**
  ```bash
  pytest python/tests/neuralnetwork -v
  # all passed
  ```

- [ ] **Protocol 準拠**
  - [ ] sequential_train.py が nn_trainer_protocols.py に準拠
  - [ ] または、乖離がドキュメントで説明されている

- [ ] **ドキュメント整備**
  - [ ] WORKTREE_SCOPE.md 完成
  - [ ] 18_neuralnetwork.md 更新
  - [ ] 全公開関数に docstring

- [ ] **import エラーなし**
  ```bash
  python -c "from jax_util.neuralnetwork import *"
  # no error
  ```

---

## 8. 施工上のリスク・対応

| リスク | 対応 |
| --- | --- |
| PyTreeOptim import 失敗 | functional.py 定義確認 + import path 検証 |
| Protocol/実装 mismatch | nn_trainer_protocols を正本に統一 |
| テスト失敗 | 各層の入出力形状・dtype を丁寧に確認 |
| pyright エラー | 型注釈を strictify、型安全性を高める |

---

**まとめ**: このドキュメントに従い、段階的に実装・テスト・ドキュメント整備を進めることで、
neuralnetwork モジュールの実装安定性と保守性を高めることができます。

