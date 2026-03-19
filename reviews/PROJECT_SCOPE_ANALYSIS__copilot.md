# プロジェクトスコープ徹底解析・詳細設計

**作成日**: 2026-03-18  
**対象 Worktree**: `nn-develop-20260318`  
**対象モジュール**: `python/jax_util/neuralnetwork/`  
**対象テスト**: `python/tests/neuralnetwork/`  

---

## I. 全体スコープ

### 1. Worktree の目的

```
このworktreeは、ニューラルネットワーク実装モジュールの設計・実装・テストを進める専用環境です。
- 実装対象：/python/jax_util/neuralnetwork/
- テスト対象：/python/tests/neuralnetwork/
- 変更対象外：base, solvers, optimizers, hlo 等の安定モジュール
- 文書対象：documents/conventions/python/18_neuralnetwork.md
```

### 2. プロジェクトの位置づけ

#### 2.1 全体構成

| モジュール | 状態 | 責務 | 文書 |
| --- | --- | --- | --- |
| `base` | 安定 | 型エイリアス・定数・Protocol の基盤 | [`base_components.md`](../documents/design/base_components.md) |
| `solvers` | 安定 | 数値ソルバ・最適化アルゴリズム | [`apis/solvers.md`](../documents/design/apis/solvers.md) |
| `optimizers` | 安定 | 勾配降下・確率最適化 | [`apis/optimizers.md`](../documents/design/apis/optimizers.md) |
| `hlo` | 安定 | HLO ダンプ・解析 | [`apis/hlo.md`](../documents/design/apis/hlo.md) |
| **`neuralnetwork`** | **実験段階** | **順伝播・層設計・学習** | [`18_neuralnetwork.md`](../documents/conventions/python/18_neuralnetwork.md) |
| `functional` | 補助 | 関数最適化の Protocol | `(base に依存)` |

#### 2.2 ニューラルネットワークモジュールの現在地

- **状態**: 実験段階
- **安定性**: 低 → 設計・API が変わることを前提にしている
- **依存**: `base` に依存、`solvers`/`optimizers` に未依存（実験段階ゆえ分離）
- **公開規約**: `documents/conventions/python/18_neuralnetwork.md` による最低限ルール

---

## II. コード構成・現在の実装状況

### 1. ファイル構成

```
python/jax_util/neuralnetwork/
├── __init__.py              # 公開API: NeuralNetwork, build_neuralnetwork, train_step, train_loop
├── protocols.py            # 型エイリアス・Protocol定義（Params, Static, Ctx, Carry, Aux...）
├── neuralnetwork.py        # 順伝播（forward）とファクトリ、state_initializer
├── layer_utils.py          # 具体的な層実装（StandardNNLayer, IcnnLayer）+ 変換ユーティリティ
├── train.py                # 標準的な学習ループ（train_step, train_loop）
├── sequential_train.py     # 層別学習（実験中）GradientBackprop, GradientTrainer
└── nn_trainer_protocols.py # 層別学習の Protocol（SingleLayerBackprop, IncrementalTrainer）

python/tests/neuralnetwork/
├── test_neuralnetwork_forward.py      # 順伝播テスト（standard/icnn）
├── test_neuralnetwork_train.py        # 標準学習ループテスト
└── test_layer_utils_and_training.py  # (名前のみ、内容未確認)
```

### 2. 主要な型・Protocol

#### 2.2.1 基本型エイリアス

```python
# protocols.py の定義から

Params = PyTree[Array]          # パラメータ（可微分配列 PyTree）
Static = PyTree[Any]           # 静的情報（微分不可能な部分）

Ctx = Protocol(...)            # 固定状態（入力など）
Carry = Protocol:
    z: Matrix                  # 層出力（(n, batch) 形式）

NeuralNetworkLayer = eqx.Module:  # 層の抽象基底
    def __call__(carry, ctx) -> Carry

Aux = Protocol(...)            # 補助情報（未定義に近い）
BackpropState = Protocol(...)  # 逆伝播状態（未詳細化）
```

#### 2.2.2 最適化 Protocol

```python
# protocols.py の定義から

PyTreeOptimizationProblem = OptimizationProblem[Params] & Protocol
    # 目的関数が Params → Scalar の形
    objective: Callable[[Params], Scalar]

ConstrainedPyTreeOptimizationProblem = 
    ConstrainedOptimizationProblem[Params, Vector, Vector] & PyTreeOptimizationProblem
    # 制約条件付き最適化
```

### 3. 実装の詳細

#### 3.1 neuralnetwork.py

**現在の状態**：
- `NeuralNetwork(eqx.Module)`: layers, network_type, layer_sizes をフィールドに持つ
- `__call__(x: Matrix) -> Matrix`: 層を順伝播し出力
- `build_neuralnetwork()`: 指定された層サイズ・活性化でネットワークを構築
- `forward_with_cache()`: 中間キャッシュを保存しながら順伝播

**形式**: 
- 入力 `x: Matrix` は `(n_features, batch)` 形式を想定
- 出力 `y: Matrix` も同じ形式

**実装上の特徴**:
- `state_initializer()`: network_type(standard/icnn)に応じて ctx, carry を初期化
- Standard NN と ICNN（入力正則化ネットワーク）両対応

#### 3.2 layer_utils.py

**層の定義**：

1. **Standard 層**:
   ```python
   StandardNNLayer(eqx.Module):
       W: Matrix              # 重み (output_dim, input_dim)
       b: Vector              # バイアス
       activation: Callable
   ```

2. **ICNN 層**:
   ```python
   IcnnLayer(eqx.Module):
       W: Matrix              # 隠れ層重み（非負）
       W_x: Matrix            # 入力重み
       b: Vector              # バイアス
       activation: Callable
   ```

**ユーティリティ**:
- `standardNN_layer_factory()`: He初期化で Standard 層生成
- `icnn_layer_factory()`: 非負重みで ICNN 層生成
- `module_to_vector()`: eqx.Module → flat Vector + RebuildState
- `vector_to_module()`: flat Vector + RebuildState → eqx.Module

#### 3.3 train.py

**実装**：
```python
train_step(params, batch, optimizer: optax, opt_state, loss_fn)
    → (new_params, new_opt_state, metrics)
    # 1ステップ: grad計算 → optax更新

train_loop(params, batch, optimizer, opt_state, loss_fn, num_steps)
    → (final_params, final_opt_state)
    # jax.lax.scan を使ったループ
```

**性質**:
- シンプル、汎用的
- optax との結合で標準的な SGD/Adam 可能
- メトリクス出力は `{'loss': scalar}`

#### 3.4 sequential_train.py（実験中）

**目的**：層別逐次学習（incremental/greedy な学習）

**現在の状態**：
- `GradientBackprop(eqx.Module)`: 1層分の逆伝播を実行
- `GradientTrainer(eqx.Module)`: 複数層を順に学習

**問題点** ⚠️ **（後述）**：
1. `PyTreeOptim` クラスが定義されていない
2. `buildoptim()` が `PyTreeOptim` を返そうとしているが、その定義なし
3. `nn_trainer_protocols.py` との二重定義・不一致

---

## III. 現在の問題点・ギャップ

### 1. コード上の問題

#### 1.1 sequential_train.py 未完成

**症状**:
```python
# sequential_train.py, line 48 付近
new_obj = PyTreeOptim(objective=new_loss)  # ← PyTreeOptim未定義
```

**原因**:
- `PyTreeOptim` は concrete 実装クラスか？
  - protocols.py には Protocol としての `PyTreeOptimizationProblem` しかない
  - 具体的な実装クラスが `functional` にあるはずだが、import が曖昧

**影響**:
- sequential_train.py 全体がインポート時に失敗する可能性
- テストが実行できない

#### 1.2 LossFn Protocol の欠落

**症状**:
```python
# train.py, line 25-26
loss_fn: LossFn,  # ← LossFn は使われているが...
```

**原因**:
- `LossFn` が protocols.py に定義されていない
- train.py でだけ使われている
- 現在の pyright は見過ごしているが、将来の strict 検査で問題になる

**型の推測値**:
```python
LossFn = Callable[[Params, Matrix], Scalar]  # 推測
```

#### 1.3 Aux の未定義

**症状**:
```python
# protocols.py, line 38-39
class Aux(Protocol):
    ...  # 空
```

**問題**:
- Aux は実装上では `None` で返されることが多い
- しかし Protocol として空なので、返却型が曖昧

**用途**:
- メトリクス、統計情報などの補助出力用と思われるが、実装上は捨てられている

#### 1.4 BackpropState の未詳細化

**症状**:
```python
# protocols.py, line 40-41
class BackpropState(Protocol):
    ...
```

**問題**:
- 目的は「上位レイヤーからの伝搬情報」だが、フィールドが未定義
- sequential_train.py で使われる形跡がない

**推定される責務**:
- 逆伝播時の勾配・状態の伝搬情報
- layer-wise learning では層の出力勾配など

---

### 2. 設計上のギャップ

#### 2.1 Layer-wise Learning の設計が unfold 状態

**背景**:
- `sequential_train.py` は層別逐次学習を実装しようとしている
- しかし、`nn_trainer_protocols.py` との役割分担が不明

**問題点**:

| 側面 | nn_trainer_protocols.py | sequential_train.py | 関係 |
| --- | --- | --- | --- |
| SingleLayerBackprop の定義 | Protocol として定義 | 依存しているが... | 不一致の可能性 |
| 呼び出し形式 | `__call__(layer, cache, optim, backprop_state)` | `__call__(layer, cache, obj)` | **シグネチャ不一致** |
| 返却型 | `Tuple[SingleLayerBackprop, ...]` | `Tuple[GradientBackprop, NeuralNetworkLayer, ...]` | 実装と Protocol の不一致 |

**推定される原因**:
- nn_trainer_protocols.py を定義した後、sequential_train.py で仕様変更が起きたが、Protocol を更新していない
- または、逆に実装が先に進まず Protocol だけが残っている

#### 2.2 forward_with_cache() の使命が不明

**症状**:
```python
# neuralnetwork.py, line 90-103
def forward_with_cache(x: Matrix, network: NeuralNetwork) 
    → Tuple[Matrix, Tuple[Carry], Ctx]
```

**問題**:
- 中間状態をキャッシュする機能だが、呼び出し側がない
- システム中で、forward_with_cache 依存のコードが見あたらない

**推定される用途**:
- 層別学習で逆伝播時の中間出力を再利用するための補助
- しかし、sequential_train.py では直接使われていない

#### 2.3 Ctx / Carry の使い分けが曖昧

**背景**:
- Standard NN では `StandardCalry(z=x)` で初期化、Ctx は空
- ICNN では `IcnnCarry(z=x)`, `IcnnCtx(x=x)` で両方使用

**問題**:
- 「固定」と「更新」の境界が層タイプに依存
- Protocol としては統一形式なのに、実装では分岐

**設計上の選択肢**:
1. Ctx / Carry を層タイプごとに特殊化する（現在）→ 型安全性UP、冗長性UP
2. 汎用 Carry で全て統一する → シンプル、ただし表現力が制限
3. 層の内部で自由に定義、Protocol で型を隠蔽 → 最も柔軟だが検証困難

---

### 3. 文書・規約面のギャップ

#### 3.1 WORKTREE_SCOPE.md がテンプレート状態

**今の状態**:
```
Branch: (空)
Worktree path: (空)
Purpose: (空)
Owner or agent: (空)
```

**必要な情報**:
- `nn-develop-20260318` の実際の Branch/目的/Owner を埋める
- Editable/Read-only ディレクトリを明記
- 必須チェック項目を追加

#### 3.2 18_neuralnetwork.md の詳細不足

**概要**:
- 実験段階モジュールの位置づけだけ
- Protocol 名の統一（✓ 既に遵守）が主

**必要な追加**:
- Layer 設計方針
- Ctx/Carry の使い分け
- Layer-wise learning の設計原則
- `sequential_train.py` の実験段階ステータス

#### 3.3 API ドキュメント / Docstring の不足

**現状**:
```python
# docstring がほぼ皆無
def build_neuralnetwork(
    network_type: str,
    layer_sizes: Tuple[int,...],
    activation: str,
    random_key: Scalar,
)->NeuralNetwork:
    """標準的な NN または ICNN を構築するファクトリ関数。"""
    # ← これだけ
```

**必要な情報**:
- パラメータの詳細（network_type の許可値、activation の許可値）
- 返却値の形式
- 部分の行動（バッチ軸の定義、初期化戦略）

---

## IV. テスト状況

### 1. 現在のテストカバレッジ

| テストファイル | テスト数 | 対象 | 状態 |
| --- | --- | --- | --- |
| test_neuralnetwork_forward.py | 2 | forward (standard/icnn) | ✓ 実装済 |
| test_neuralnetwork_train.py | 1 | train_step | ✓ 実装済 |
| test_layer_utils_and_training.py | ? | (未確認) | ? |

### 2. 未カバー領域

- ❌ sequential_train.py（GradientBackprop, GradientTrainer）
- ❌ layer_utils.py の module_to_vector / vector_to_module
- ❌ forward_with_cache の動作
- ❌ エラーハンドリング（異常カタマ—の層サイズなど）
- ❌ バッチ軸の異なる入力への対応

### 3. テスト実行状況

```
pytest --collect-only    # 83 tests (全体)
pytest tests/neuralnetwork/  # 推定3テスト

現在のステータス：
- test_neuralnetwork_forward.py: ✓ PASS
- test_neuralnetwork_train.py: ✓ PASS (ただし metrics 出力のみ検証)
```

---

## V. 依存・インポート構造

### 1. 内部依存図

```
neuralnetwork/
├─ protocols.py (base に依存)
│  └─ base: Matrix, Scalar, Vector, OptimizationProblem...
│
├─ layer_utils.py (protocols に依存)
│
├─ neuralnetwork.py (layer_utils に依存)
│
├─ train.py (protocols に依存, 外部: optax, jax)
│
├─ sequential_train.py (protocols, layer_utils, neuralnetwork に依存)
│  ├─ functional: OptimizationProblem (未確認の詳細)
│  └─ nn_trainer_protocols に依存（ただし Protocol と実装のmismatch）
│
└─ nn_trainer_protocols.py (protocols に依存)
```

### 2. 外部依存

| 依存 | 用途 | 必須 |
| --- | --- | --- |
| `equinox` | PyTree Module 定義 | ✓ |
| `jax` | JAX 制御フロー、自動微分 | ✓ |
| `jax.numpy` | 配列操作 | ✓ |
| `optax` | 最適化器（train.py） | ✓ |
| `jaxtyping` | 型注釈 | ✓ |

---

## VI. 詳細設計の提案

### 1. 即座に対応すべき項目

#### 1.1 PyTreeOptim の定義位置を明確化

**案**:
```python
# functional.py または neuralnetwork.py に追加
@dataclass(frozen=True)
class SimpleOptimizationProblem:
    """簡易最適化問題の実装"""
    objective: Callable[[Params], Scalar]
    
    @property
    def objective(self) -> Callable:
        return self.objective
```

**判断基準**:
- 既存の functional モジュール設計に従うか
- 新規実装するか、functional からのみインポートするか

#### 1.2 LossFn を protocols.py に追加

**提案**:
```python
# protocols.py に追加

LossFn: TypeAlias = Callable[[Params, Matrix], Scalar]
    # loss_fn(params, batch) → scalar
```

#### 1.3 Aux と BackpropState を具体化

**Aux の案**:
```python
@dataclass(frozen=True)
class TrainingAux:
    """学習ステップの補助情報"""
    loss: Scalar
    grad_norm: Scalar | None = None
    timestamp: int | None = None
```

**BackpropState の案**:
```python
@dataclass(frozen=True)
class LayerBackpropState:
    """層の逆伝播状態"""
    grad_carry: Carry      # 勾配情報
    auxiliary: Auxiliary | None = None
```

---

### 2. 設計改善案

#### 2.1 sequential_train.py / nn_trainer_protocols.py の整合

**現状の問題**:
```python
# nn_trainer_protocols.py の Protocol
class SingleLayerBackprop(Protocol):
    def __call__(
        self,
        layer,
        cache,
        optim: OptimizationProblem[NeuralNetwork],  # ← ここ
        backprop_state,
    ) -> ...

# sequential_train.py の実装
class GradientBackprop(eqx.Module):
    def __call__(
        self,
        layer,
        cache,
        obj: PyTreeOptimizationProblem,  # ← 型が違う
    ) -> ...
```

**改善案**:
1. `nn_trainer_protocols.py` を正本に定める
2. `sequential_train.py` を Protocol に合わせる
3. または、Protocol を実装に合わせて修正する

#### 2.2 forward_with_cache の明確化

**提案**:
```python
def forward_with_cache(
    x: Matrix,
    network: NeuralNetwork,
) -> Tuple[Matrix, Tuple[Carry, ...], Ctx]:
    """
    順伝播を実行し、中間状態をキャッシュ。
    
    層別学習（sequential_train.py）で逆伝播時に
    キャッシュされた Carry を参照することで、
    各層の勾配を効率的に計算する。
    
    Args:
        x: 入力 (n_features, batch)
        network: NeuralNetwork インスタンス
    
    Returns:
        (output, carries_cache, context)
    """
```

#### 2.3 Ctx / Carry の統一設計

**案 A: 層タイプごとに特殊化（現在）**
- 利点: 型安全性が高い
- 欠点: 冗長、Protocol が複雑

**案 B: 汎用 Carry へ統一**
```python
@dataclass(frozen=True)
class UnifiedCarry(eqx.Module):
    z: Matrix              # 層出力
    auxiliary: dict[str, Any] | None = None  # 層固有データ
```
- 利点: シンプル、拡張容易
- 欠点: 型安全性が低下

**推奨**: 案 A の継続（現在の設計は実験段階として妥当）

---

### 3. ドキュメント整備案

#### 3.1 WORKTREE_SCOPE.md を完成化

```markdown
# WORKTREE_SCOPE

## Worktree Summary
- Branch: nn-develop-20260318
- Worktree path: ./.worktrees/nn-develop-20260318
- Purpose: ニューラルネットワーク実装の設計・実装・テスト
- Owner: (指定)

## Editable Directories
- python/jax_util/neuralnetwork/
- python/tests/neuralnetwork/
- documents/conventions/python/18_neuralnetwork.md
- reviews/

## Read-Only Or Avoid Directories
- python/jax_util/base/
- python/jax_util/solvers/
- python/jax_util/optimizers/
- python/jax_util/hlo/
- python/jax_util/functional/  ← (依存関係は読むが修正しない)

## Required References Before Editing
- documents/conventions/python/18_neuralnetwork.md
- documents/conventions/python/03_operators.md
- documents/conventions/python/04_type_annotations.md
- documents/design/base_components.md

## Required Checks Before Commit
- pyright python/jax_util/neuralnetwork
- pytest python/tests/neuralnetwork/
```

#### 3.2 18_neuralnetwork.md 拡張案

```markdown
## 追加セクション

### Layer-wise Learning の実験段階

- sequential_train.py は現在、層別逐次学習の設計実験中
- 正式API ではなく、Protocol（nn_trainer_protocols.py）との整合を確認中
- メインの公開API（train_step, train_loop）に変更なし

### Ctx と Carry の使い分け

- Standard NN: Ctx は不使用（空の StandardCtx）
- ICNN: Ctx は入力キャッシュ、Carry は各層出力
- プロトコルとして統一形式だが、実装は層タイプに依存

### Module 変換ユーティリティ

- module_to_vector / vector_to_module: 層の flat 化と復元
- Sequential learning で勾配計算効率化に使用
```

---

## VII. リスク・対応表

| リスク | 現状 | 影響 | 対応 | 優先度 |
| --- | --- | --- | --- | --- |
| PyTreeOptim 未定義 | sequential_train.py が失敗 | import 不可、テスト実行不可 | 定義位置を明確化、実装 | 🔴 高 |
| Protocol/実装 mismatch | nn_trainer_protocols vs sequential_train.py | 将来の修正時混乱 | どちらかを正本に定め、統一 | 🟡 中 |
| Aux/BackpropState 空 | 将来の追加情報に対応困難 | 拡張時の修正コスト増 | 型定義を充実させる | 🟡 中 |
| WORKTREE_SCOPE テンプレート | 他者への引き継ぎが困難 | worktree 運用の混乱 | 完成化・詳細化 | 🟡 中 |
| forward_with_cache 使途不明 | コード理解困難、死コード可能性 | メンテナンス負荷 | 用途を明記、テスト追加 | 🟢 低 |
| Docstring 不足 | API 理解困難 | 新規利用者の立ち上り遅延 | 各関数に詳細 docstring 追加 | 🟢 低 |

---

## VIII. 今後の重点項目（推奨順）

### Phase 1: 即座対応（1-2日）

1. **PyTreeOptim の定義**
   - functional.py または sequential_train.py に実装クラスを追加
   - import 失敗を即座に解決

2. **nn_trainer_protocols vs sequential_train.py の整合**
   - nn_trainer_protocols.py を正本と定める
   - sequential_train.py のシグネチャをプロトコルに合わせる
   - またはコメントで仕様の乖離を明記

3. **LossFn, Aux, BackpropState の型定義**
   - protocols.py に TypeAlias / dataclass で具体化
   - 将来の拡張を想定した余地を残す

### Phase 2: 文書整備（1-2日）

4. **WORKTREE_SCOPE.md の完成化**
   - Branch名、目的、所有者を埋める
   - Editable/Read-only ディレクトリを確定
   - 必須チェックコマンドを追加

5. **18_neuralnetwork.md 拡張**
   - Layer-wise learning 実験状況を記述
   - forward_with_cache の用途を明記
   - Ctx/Carry 設計の背景を補足

6. **Docstring 充実**
   - build_neuralnetwork(), forward_with_cache() 等に詳細 docstring 追加
   - パラメータの許可値、形状要件を明記

### Phase 3: テスト拡充（2-3日）

7. **sequential_train.py のテスト追加**
   - GradientBackprop, GradientTrainer の動作確認
   - Protocol との準拠性を検証

8. **エッジケース対応**
   - 不正な layer_sizes、activation の入力
   - バッ・・軸異なる入力への処理

9. **forward_with_cache テスト追加**
   - キャッシュ内容の整合性確認
   - 逆伝播での活用検証

### Phase 4: 安定化（随時）

10. **機能検証ループ**
    - functional.py との連携確認
    - solvers/optimizers との依存チェック

---

## IX. 設計決定チェックリスト

実装前に以下を確認してください：

- [ ] **Protocol と実装の対応**
  - `nn_trainer_protocols.py` と `sequential_train.py` のシグネチャが一致しているか
  - または乖離がドキュメントで説明されているか

- [ ] **型定義の完全性**
  - `Aux`, `BackpropState` に dataclass で具体型を定めるか、Protocol のままか
  - `LossFn` を TypeAlias として protocols.py に追加したか

- [ ] **依存関係の明確化**
  - `PyTreeOptim` がどこで定義されるか決定したか
  - import エラーは解決したか

- [ ] **ドキュメントの整合性**
  - WORKTREE_SCOPE.md, 18_neuralnetwork.md, docstring すべてが最新か
  - 実装と文書に乖離がないか

- [ ] **テストカバレッジ**
  - forward: ✓ (2 テスト)
  - train_step: ✓ (1 テスト)
  - sequential_train: ? (未テスト)
  - layer_utils: ? (活動未確認)

---

## X. まとめ

このプロジェクトは、**ニューラルネットワーク実装の実験段階モジュール**である。

**強み**:
- 基本的な forward / train_step の実装が安定している
- テストも標準的な層・最適化器で動作確認済み
- Protocol による設計が清潔

**課題**:
1. **sequential_train.py が未完成**状態（PyTreeOptim 未定義）
2. **Protocol と実装の乖離**（nn_trainer_protocols vs sequential_train）
3. **設計ドキュメントの詳細不足**（Ctx/Carry の責務分離が不明確）
4. **型定義の未確定**（Aux, BackpropState が空）

**方針**:
- 現在の設計を尊重しつつ、未完成部分を整える
- 実験段階の柔軟性を維持しながら、情報を充実させる
- 次の safety/complexity の段階へ安全に進むための基盤作り

この分析を踏まえて、詳細実装設計を進めてください。

