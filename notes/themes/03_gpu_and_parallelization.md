# 3. GPU and Parallel Computing：層別訓練と並列化の実装戦略

**作成日**： 2026-03-19  
**対象**： 分散・並列環境での層別訓練の実装  
**関連文献**： Ben-Nun & Hoefler (2018)、Vezhnevets et al. (2016)

---

## 概要

層別訓練は、理論的には **層間の独立性** により並列化に適している。しかし、実装上はデータフロー、メモリバンド幅、通信コストなど複数の考慮が必要である。本章では、GPU 環境と分散システムにおける実装戦略を分析する。

---

## 1. 層別訓練の並列化の利点

### 1.1 計算グラフの独立性

**バックプロパゲーション**：
```
y = W_n σ(W_{n-1} σ(... σ(W_1 x)))

勾配流：
dy/dW_1 → dy/dW_2 → ... → dy/dW_n  (順序的)
```

**層別訓練**：
```
層 1 の最適化: x → h_1 (独立)
層 2 の最適化: h_1 → h_2 (層1固定後は独立)
...
層 n の最適化: h_{n-1} → h_n (層n-1固定後は独立)
```

### 1.2 パイプライン並列化の可能性

**パイプライン戦略**：
```
Time t:     GPU1 学習 層1       GPU2 待機      GPU3 待機
Time t+1:   GPU1 推論 層1       GPU2 学習 層2   GPU3 待機
Time t+2:   GPU1 準備          GPU2 推論 層2   GPU3 学習 層3
```

各 GPU が異なる層を同時処理することで、スループット向上。

---

## 2. メモリ・計算・通信のトレードオフ

### 2.1 メモリ使用量の比較

**バックプロパゲーション**：
各層の活性化を保存（逆伝播のため）：
$$M_{BP} = \sum_{i=1}^n m_i \times \text{batch\_size}$$

全層の活性化が同時にメモリに保存。

**層別訓練**：
各層を順序的に処理：
$$M_{LW} = \max_i (m_i \times \text{batch\_size}) + \text{Layer params}$$

一般に $M_{LW} < M_{BP}$ for 深いネットワーク。

### 2.2 計算量の比較

両者の forward/backward pass の計算量はほぼ同等．
- **バックプロパゲーション**： 1 パス＝（1 forward + 1 backward）
- **層別訓練**： 1 層＝（1 forward + 1 backward）、層数分反復

実質的な計算量は同程度だが、キャッシュ局所性が異なる。

### 2.3 通信コスト（分散システム）

**パラメータサーバーモデル**：

計算機 $k$ が層 $i$ を処理：
$$\text{Comm Cost} = \text{Read}(W_i) + \text{Send}(\Delta W_i)$$

層別訓練では、各층の重みが定期的に更新・共有されるため、通信頻度が多い可能性。

---

## 3. 並列化アーキテクチャ

### 3.1 データ並列 (Data Parallelism)

複数 GPU で同じ層を処理：

$$\text{for batch in batches}:$$
$$\quad \text{GPU}_1 \text{: process batch}_1 \text{ of layer i}$$
$$\quad \text{GPU}_2 \text{: process batch}_2 \text{ of layer i}$$
$$\quad \text{Synchronize and average gradients}$$

**実装例**（JAX/Optax）：
```python
def parallel_layer_update(layer, batch_chunks, optimizer_state):
    results = []
    for gpu_id, batch in enumerate(batch_chunks):
        result = jax.device_put(
            layer(batch), device=jax.devices()[gpu_id]
        )
        results.append(result)
    return jax.tree_util.tree_map(lambda *x: jnp.concatenate(x), *results)
```

### 3.2 モデル並列 (Model Parallelism)

ネットワークを複数 GPU に分割：

```
GPU 1: 層 1, 層 2     |
GPU 2: 層 3, 層 4     |  (パイプラインで処理)
GPU 3: 層 5, 層 6     |
```

**通信ボトルネック**：
層間の活性化をネットワーク越しに送信。

**層別訓練での緩和**：
各 GPU が割り当てられた層を固定後に最適化すれば、層間通信を最小化可能。

### 3.3 パイプライン並列 (Pipeline Parallelism)

時間軸で層を重ねる（GPipe アーキテクチャ）：

```
Time  | GPU1  | GPU2  | GPU3
------|-------|-------|------
t     | L1    |       |
t+1   | L2    | L1    |
t+2   | L3    | L2    | L1
t+3   |       | L3    | L2
```

**層別訓練との親和性**: パイプライン内で各層を独立最適化可能。

---

## 4. 分散層別訓練の実装戦略

### 4.1 同期更新 (Synchronous Update)

```python
# Pseudo-code: Synchronous layer-wise training

def distributed_layer_wise_train(network, data, num_layers):
    for layer_idx in range(num_layers):
        # 全ワーカーが同じ層を処理
        layer = network.layers[layer_idx]
        
        for batch in data:
            # 各ワーカーで局所勾配計算
            gradients = []
            for worker_id in range(num_workers):
                batch_subset = data_distribute(batch, worker_id)
                grad = compute_local_gradient(layer, batch_subset)
                gradients.append(grad)
            
            # 全ワーカーが勾配を同期
            avg_gradient = allreduce(gradients, op="mean")
            
            # 重み更新
            layer.weights -= learning_rate * avg_gradient
        
        # 層を固定
        network.layers[layer_idx] = freeze(layer)
```

**メリット**：
- 実装が簡潔
- 決定的（再現性が高い）

**デメリット**：
- ワーカー間の同期待機（ストラッグラー問題）

### 4.2 非同期更新 (Asynchronous Update)

```python
# Pseudo-code: Asynchronous layer-wise training

def async_layer_wise_train(network, data, num_workers):
    layer_locks = [Lock() for _ in range(len(network.layers))]
    
    def worker_task(worker_id):
        for layer_idx in range(len(network.layers)):
            with layer_locks[layer_idx]:  # 排他的ロック
                layer = network.layers[layer_idx]
                
                # ローカル処理
                for batch in get_batches_for_worker(worker_id):
                    grad = compute_gradient(layer, batch)
                    layer.weights -= learning_rate * grad
    
    # 並列実行
    threads = [Thread(target=worker_task, args=(i,)) 
               for i in range(num_workers)]
    for t in threads:
        t.start()
```

**メリット**：
- 同期待機なし
- 高スループット

**デメリット**：
- 層間の依存性が曖昧（特にワーカー数が多い場合）
- 収束性の理論的保証が弱い

---

## 5. Ben-Nun & Hoefler (2018) の並列化分類

### 5.1 DNN における並列化次元

Ben-Nun & Hoefler は以下の分類を提案：

| 並列化方式 | データ | モデル | パイプ | 時系列 |
|----------|--------|--------|--------|--------|
| **Data Parallel** | ○ | × | × | × |
| **Model Parallel** | × | ○ | × | × |
| **Pipeline Parallel** | × | △ | ○ | × |
| **Layer-wise** | △ | × | △ | ○ |

層別訓練は **時系列** 次元（時間をかけて層を順序処理）の並列化。

### 5.2 通信vs計算のバランス

**通信量**: $C = \text{(層数)} \times \text{(パラメータ数)}$

**計算量**: $F = \text{(層平均計算)} \times \text{(イテレーション数)}$

最適並列化を求めるには：
$$\text{Speedup} \propto \frac{F}{C + T_{latency}}$$

層別訓練では、$C$ が比較的小だが、層間の依存性が強いため $T_{latency}$ が支配的。

---

## 6. JAX/Equinox での実装例

### 6.1 層別訓練的な構造

```python
import jax
import equinox as eqx
import optax

class LayerWiseTrainerJAX:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.optimizer = optax.adam(learning_rate)
        self.opt_state = self.optimizer.init(model)
    
    def train_layer(self, layer_idx, data_batch):
        """単一層を最適化"""
        def loss_fn(layer_params):
            # layer_idx より前の層は固定
            modified_model = self._set_layer(self.model, layer_idx, layer_params)
            # Forward pass
            predictions = jax.vmap(modified_model)(data_batch)
            # ローカル損失（例：再構成誤差）
            return jnp.mean((predictions - data_batch)**2)
        
        layer_params = self._get_layer(self.model, layer_idx)
        grads = jax.grad(loss_fn)(layer_params)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        new_params = optax.apply_updates(layer_params, updates)
        
        self.model = self._set_layer(self.model, layer_idx, new_params)
        return new_params
    
    def _get_layer(self, model, idx):
        """層を抽出"""
        return model.layers[idx]
    
    def _set_layer(self, model, idx, new_layer):
        """層を更新して新しいモデルを返す"""
        return eqx.tree_at(
            lambda m: m.layers[idx],
            model,
            new_layer
        )
```

### 6.2 pmap を使用した GPU 並列化

```python
import jax
import jax.numpy as jnp
import jax.lax as lax

def parallel_layer_update(layer_params, batches_per_device, loss_fn, optimizer):
    """複数 device で並列層更新"""
    
    def single_device_update(params, batch):
        grads = jax.grad(loss_fn)(params, batch)
        return jax.grad(loss_fn)(params, batch)
    
    # 各デバイスで loss_fn を実行
    all_grads = jax.vmap(single_device_update)(
        jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (num_devices, *x.shape[1:])),
            layer_params
        ),
        batches_per_device
    )
    
    # 勾配平均
    avg_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), all_grads)
    
    return avg_grads
```

---

## 7. 通信隠蔽 (Communication Hiding)

### 7.1 重複計算・通信

```
従来:
GPU1: [計算] → [通信] → [計算] → [通信] → ...

隠蔽:
GPU1: [計算 層1] || [通信 層1] → [計算 層2] || [通信 層2] → ...
                   ^^^^^^^^^^^^^^^^
                     同時実行
```

層別訓練では、層 $i$ の通信中に層 $i+1$ の計算が可能（パイプライン効果）。

### 7.2 勾配累積での実装

```python
def gradient_accumulation_with_communication(layers, batches, num_accumulation=4):
    """
    勾配を累積しながら通信を隠蔽
    """
    accumulated_grads = None
    
    for batch_idx, batch in enumerate(batches):
        for layer_idx, layer in enumerate(layers):
            # forward/backward で勾配取得
            grads = compute_gradients(layer, batch)
            
            # 勾配累積
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = jax.tree_util.tree_map(
                    lambda x, y: x + y, accumulated_grads, grads
                )
            
            # 一定数ステップごとに通信（非ブロッキング）
            if batch_idx % num_accumulation == 0:
                async_communicate(accumulated_grads)
                accumulated_grads = None
```

---

## 8. 実装上の課題と対策

### 8.1 負荷不均衡 (Load Imbalance)

**問題**：各層の計算量が異なる

$$C_i = \text{Complexity}(layer_i) = O(m_i \times n_i)$$

**対策**：層の計算量に応じて計算機を割り当て、パイプライン設計を最適化

### 8.2 非決定性 (Non-determinism)

非同期更新では，同じデータを繰り返し処理しても結果が異なる可能性。

**対策**：
- 同期ポイント（barrier）を定期的に挿入
- シード固定（確率的操作の場合）

### 8.3 スケーラビリティ

層数が限定されるため、スケーラビリティに限界：
$$\text{最大 GPU 数} \leq \text{層数}$$

**対策**：
- 層内での細分化（sub-layer parallelism）
- データ並列との組み合わせ

---

## 9. 結論と推奨

### 9.1 並列化の有効性

**層別訓練が有効な場面**：
- 超深いネットワーク（100+ 層）
- メモリ制約が厳しい
- 層ごとの計算リソースが大きく異なる場合

**バックプロパゲーション優位な場面**：
- 中程度の深さ（50層以下）
- GPU/TPU リソースが豊富
- 学習速度が最優先

### 9.2 実装推奨

```
【推奨ハイブリッド戦略】
1. Pre-training フェーズ：データ並列バックプロパゲーション
2. 深層最適化フェーズ：層別訓練 + パイプライン並列
3. Fine-tuning フェーズ：小学習率でバックプロパゲーション
```

---

## 参考文献

- **Ben-Nun, T., & Hoefler, T.** (2018). Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis. ACM Computing Surveys, 52(4).
- **Vezhnevets, A., Mnih, V., Agapiou, J., Osindero, S., Graves, A., Vinyals, O., & Kavukcuoglu, K.** (2016). Strategic Attentive Writer for Learning Macro-Actions. arXiv:1606.04695.
- **Huang, Y., Cheng, Y., Bapna, A., et al.** (2019). GPipe: Efficient Training of Giant Models on Multiple GPUs. ICLR.
- **Lepikhin, D., Lee, H., Xu, Y., et al.** (2020). GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. ICLR.
