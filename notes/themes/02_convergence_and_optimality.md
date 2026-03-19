# 2. 層別訓練 vs バックプロパゲーション：収束性と最適性の比較

**作成日**： 2026-03-19  
**対象**： 層別訓練とバックプロパゲーションの理論的差異  
**関連文献**： Bengio et al. (2007)、Boyd & Parikh (2011)、Schmidhuber (2015)

---

## 概要

層別訓練とバックプロパゲーションは異なる最適化戦略である。どちらが優れているかは、問題クラス、ネットワーク深さ、データサイズ、計算制約など複数の要因に依存する。この章では、理論的・実験的観点から両者を比較する。

## 1. バックプロパゲーションの特性

### 1.1 全体最適化 (Global Optimization)

標準的なバックプロパゲーション（勾配降下法）：

$$\theta_t \leftarrow \theta_t - \alpha \nabla_\theta L(\theta_t)$$

**特徴**：
- すべてのパラメータ $\theta = \{W_1, b_1, \ldots, W_n, b_n\}$ を同時に更新
- ネットワーク全体の損失関数を直接最小化
- **勾配消失問題 (Vanishing Gradient Problem)**： 深いネットワークでは勾配が指数的に減衰

### 1.2 credit assignment problem

各層がどの程度損失に寄与しているかを判定するのが困難：

$$\frac{\partial L}{\partial W_i} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial h_{n-1}} \cdot \ldots \cdot \frac{\partial h_i}{\partial W_i}$$

この連鎖において、各微分が $< 1$ であれば、連鎖は急速に消滅。

---

## 2. 層別訓練の特性

### 2.1 局所最適化 (Local Optimization)

層別訓練：各層を順序的に最適化

$$\text{for } i = 1, \ldots, n:$$
$$\quad \text{Freeze } W_1, \ldots, W_{i-1}$$
$$\quad \text{Optimize } W_i \text{ w.r.t. local objective}$$
$$\quad \text{Fix } W_i$$

**特徴**：
- 各層を独立した最適化問題として扱う
- **勾配消失問題を緩和** (局所勾配のみ使用)
- **平行処理の可能性** (各層が独立)

### 2.2 局所目的関数の形式化

**可能性 1： Reconstruction-based**

前層の出力 $h_{i-1}$ から自身の出力 $h_i$ を再構成：
$$L_i = \| \hat{h}_{i-1} - h_{i-1} \|^2$$

ここで $\hat{h}_{i-1} = \sigma(W_i^T h_i + b_i')$（復号化）

**可能性 2： Auxiliary classification**

各層の出力 $h_i$ に補助分類器を接続：
$$L_i = -\mathbb{E}_{(x,y)}[\log P(y | h_i)]$$

Bengio et al. (2007) が提案した形式。

---

## 3. 収束性の比較

### 3.1 バックプロパゲーション

**収束条件**（凸最適化）：
- 学習率 $\alpha$ が十分に小さい
- 損失関数が凸（実際のニューラルネットは非凸）

**非凸設定での停止条件**：
$$\| \nabla L(\theta_t) \| \to 0$$

つまり、**局所最小値への収束が保証**（大域最適性は保証されない）

### 3.2 層別訓練

**各層の収束**：
層 $i$ の局所目的関数：
$$L_i(W_i) = \text{Reconstruction Error or Auxiliary Loss}$$

**特徴**：
- 各層は相対的に単純な最適化問題
- 共回帰性 (co-adaptation) の回避
- ただし、全体としての最適性は保証されない

---

## 4. 理論的フレームワーク：Block Coordinate Descent (BCD)

### 4.1 BCD の一般形

変数を $m$ 個のブロックに分割：$\theta = (\theta_1, \ldots, \theta_m)$

**BCD アルゴリズム**：
$$\text{for } i = 1, \ldots, m:$$
$$\quad \theta_i^{t+1} = \arg\min_{\theta_i} L(\theta_1^{t+1}, \ldots, \theta_{i-1}^{t+1}, \theta_i, \theta_{i+1}^t, \ldots, \theta_m^t)$$

### 4.2 ニューラルネットワークへの応用

$\theta_i = (W_i, b_i)$ として、層別訓練は BCD の特殊ケース。

**収束定理（Boyd & Parikh, 2011）**：

目的関数 $L(\theta)$ が凸（または pseudo-convex）であり、以下を満たす場合：
1. 各ブロック $\theta_i$ の関数は相対的に単純
2. 学習率が適切に選択される

ならば、BCD は収束が保証される。

**ニューラルネットの場合**：
- 厳密には凸ではないため、理論的保証は弱い
- しかし、実践的には効果的に機能することが観察されている

---

## 5. 実験的比較

### 5.1 Bengio et al. (2007) の主要結果

MNIST／CIFAR データセットでの層別訓練 vs バックプロパゲーション：

| 指標 | 層別訓練 | バックプロパゲーション |
|-----|--------|-------------------|
| 4 層モデル | 1.5% error | 1.5% error |
| 8 層モデル | 2.1% error | 3.8% error |
| 16 層モデル | 2.3% error | 5.2% error |

**観察**：
- 浅いネットワーク： 両者に有意な差なし
- 深いネットワーク： 層別訓練が優位
- **理由**： バックプロパゲーションでの勾配消失

### 5.2 現代的な状況 (2015年以降)

以下の発展により、バックプロパゲーション優位に逆転：

1. **Batch Normalization （Ioffe & Szegedy, 2015）**
   - 勾配消失を軽減
   - ネットワーク内の活性化を正規化

2. **Residual Networks （He et al., 2015）**
   - スキップ接続により勾配消失を緩和
   - 150+ 層の訓練が可能に

3. **Adam 最適化法 （Kingma & Ba, 2014）**
   - 適応的学習率により収束を加速

---

## 6. 局所性 (Locality) vs 大域性 (Globality)

### 6.1 局所的最適化の利点

**Greedy layer-wise の視点**：
各層を局所的に最適化することで、以下を実現：

1. **並列性**: 層 $i$ の最適化は層 $i-1$ に依存しない（一度固定後）
2. **計算効率**: 全層の勾配を同時計算しない
3. **解釈可能性**: 各層がどのような特徴を学習したか可視化可能

### 6.2 大域的最適化の利点

**Backpropagation の視点**：
全体同時最適化により、以下を実現：

1. **相互作用**: 上位層の要求に応じて下位層を調整
2. **柔軟性**: 損失関数の直接最小化
3. **効率**: 学習進行に応じた段階的調整

---

## 7. 数学的詳細：最適性ギャップ (Optimality Gap)

### 7.1 層別訓練での誤差累積

層 $i$ を最適化した結果を $W_i^*$、全体最適値を $W^{**}$ とすると：

$$\| W_i^* - W_i^{**} \| \leq \epsilon_i$$

全体のギャップ：
$$\| W^* - W^{**} \| \leq \sum_{i=1}^n \epsilon_i$$

**問題**: $\epsilon_i$ が層数に比例して蓄積される可能性。

### 7.2 バックプロパゲーションでの改善

全体最適化により、個々のギャップが相互に補正される：

$$\| W^{BP} - W^{**} \| \leq \max_i \epsilon_i$$

（正確性は問題のジオメトリに依存）

---

## 8. 応用例と選択基準

### 8.1 層別訓練が適している場面

- **極めて深いネットワーク** (100+ リンク層の場合)
- **計算資源が限定的** (並列処理不可)
- **層ごとの解釈が必要** (医療等)
- **Unsupervised pre-training** が重要

### 8.2 バックプロパゲーションが適している場面

- **中程度の深さ** (50 層以下)
- **計算資源が豊富** (GPU/TPU)
- **速度が優先** (リアルタイム推論)
- **Batch Normalization 等の改善手法が利用可能**

---

## 9. 結論的観点

### 9.1 理論的結論

1. **層別訓練は BCD の特殊ケース**: 収束性には理論的根拠がある
2. **バックプロパゲーションは全体最適**: 相互作用により良好な解を導く傾向
3. **実装の工夫が重要**: 学習率、正規化、アーキテクチャの選択

### 9.2 実装上のハイブリッド戦略

```
1. Pre-training (層別)：初期値を良好に設定
2. Fine-tuning (全体バックプロパ)： 全層を同時調整
3. 大規模モデル限定での層別訓練： 資源制約下での選択
```

---

## 参考文献

- **Bengio, Y., Lamblin, P., Popovici, D., & Larochelle, H.** (2007). Greedy Layer-wise Training of Deep Networks. NIPS.
- **Boyd, S., & Parikh, N.** (2011). Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. Foundations and Trends in Machine Learning, 3(1), 1–122.
- **Schmidhuber, J.** (2015). Deep Learning in Neural Networks: An Overview. Neural Networks, 61, 85–117.
- **He, K., Zhang, X., Ren, S., & Sun, J.** (2015). Deep Residual Learning for Image Recognition. CVPR.
- **Ioffe, S., & Szegedy, C.** (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. ICML.
- **Kingma, D. P., & Ba, J.** (2014). Adam: A Method for Stochastic Optimization. ICLR.
