# 1. RBM と Contrastive Divergence の理論・実装詳細

**作成日**： 2026-03-19
**対象**： Restricted Boltzmann Machine (RBM) と学習アルゴリズムの詳細
**関連文献**： Smolensky (1986)、Hinton (2002)、Fischer & Igel (2012)

---

## 概要

Restricted Boltzmann Machine (RBM) は、一見すると層別訓練とは無関係に見えるが、Deep Belief Networks (DBN) を通じて密接に関連している。RBM の学習アルゴリズムである Contrastive Divergence (CD) は、層別訓練の理論的基礎として重要な役割を果たす。

## 1. RBM の構造と確率モデル

### 1.1 基本的な設定

RBM は以下の特徴を持つ生成確率モデル：
- **可視層 (visible layer)**： $v \in \{0, 1\}^m$
- **隠れ層 (hidden layer)**： $h \in \{0, 1\}^n$
- **二部グラフ構造 (bipartite graph)**： 層内の接続なし、層間のみ接続
- **重み行列**： $W \in \mathbb{R}^{m \times n}$、バイアス $a \in \mathbb{R}^m$、$b \in \mathbb{R}^n$

### 1.2 エネルギー関数と確率分布

**エネルギー関数**：
$$E(v, h) = -a^T v - b^T h - v^T W h$$

この形式は、Hopfield ネットワークの統計物理的モデルから由来する。

**同時確率分布**：
$$P(v, h) = \frac{1}{Z} e^{-E(v, h)}$$

ただし、分配関数 (partition function) は：
$$Z = \sum_{v, h} e^{-E(v, h)}$$

### 1.3 条件付き独立性

RBM の最大の特徴は、**層内での条件付き独立性**である：
$$P(v | h) = \prod_{i=1}^m P(v_i | h)$$
$$P(h | v) = \prod_{j=1}^n P(h_j | v)$$

個別活性化確率：
$$P(h_j = 1 | v) = \sigma\left( b_j + \sum_{i=1}^m w_{i,j} v_i \right)$$
$$P(v_i = 1 | h) = \sigma\left( a_i + \sum_{j=1}^n w_{i,j} h_j \right)$$

この特性により、Gibbs サンプリングが効率的に実行可能。

---

## 2. Contrastive Divergence (CD) アルゴリズム

### 2.1 従来の最尤推定の問題

尤度関数：
$$\mathcal{L}(W) = \mathbb{E}[P(v)]$$

その勾配：
$$\nabla_W \mathcal{L} = \mathbb{E}_{data}[v h^T] - \mathbb{E}_{model}[v' h'^T]$$

**問題**： 第二項（モデル期待値）の計算は、分配関数 $Z$ の計算が困難なため、全層の組み合わせ探索が必要。

### 2.2 Contrastive Divergence の着想

Hinton (2002) により提案された CD アルゴリズムの核心：

> KL 発散を直接最小化せず、**Contrastive 項の差分**を最小化する。

**アイデア**：
- データ分布 $P_{data}$
- モデル分布 $P_{model}$

を明示的に計算せず、離散的な反復により **エネルギー差分** を最小化。

### 2.3 CD-1 (1-step) アルゴリズム

単一サンプル $v$ に対する基本手順：

**ステップ 1**： $v$ を入力、確率 $P(h|v)$ から $h$ をサンプル
$$h \sim P(h | v)$$

**ステップ 2**： 外積を計算（**正の勾配**）
$$\Delta W^+ = v h^T$$

**ステップ 3**： $h$ から $v$ を再構成、確率 $P(v|h)$ から $v'$ をサンプル
$$v' \sim P(v | h), \quad h' \sim P(h | v')$$

**ステップ 4**： 外積を計算（**負の勾配**）
$$\Delta W^- = v' h'^T$$

**ステップ 5**： 重み更新
$$\Delta W = \epsilon (\Delta W^+ - \Delta W^-) = \epsilon (v h^T - v' h'^T)$$

**ステップ 6**： バイアス更新
$$\Delta a = \epsilon (v - v'), \quad \Delta b = \epsilon (h - h')$$

### 2.4 CD の収束性と理論的要件

Sutskever & Tieleman (2010) は CD-1 の収束性を分析：

- **弱い条件下での収束**： 学習率が十分に小さく、バッチサイズが大きければ収束する傾向
- **限界**： 理論的には、$\infty$-step (完全な MCMC) の方が厳密だが、実装上 1-step で十分実用的

**数学的原因**：
- CD は **Divergence の最小化**ではなく、**Contrastive 項の減少**を目指す
- エネルギー関数が凸でない場合、局所的な最小値に収束する可能性

---

## 3. RBM と層別訓練の関連

### 3.1 Deep Belief Networks (DBN) での位置づけ

DBN は複数の RBM を積み重ステッピングしたもの：

1. **第1層 RBM** を学習 (unsupervised)
2. **第1層を固定**し、その出力を第2層入力として使用
3. **第2層 RBM** を学習
4. 必要に応じて **バックプロパゲーション** で全層を微調整 (fine-tuning)

### 3.2 層別訓練への視点

**RBM の CD 学習 = 局所的な最適化**

$$v_\ell \to h_\ell \to v_\ell' \to h_\ell' \text{ (within layer } \ell \text{)}$$

この反復は、1 層内でのパラメータ更新であり、**上位層の情報を直接使用しない**点が層別訓練の特徴。

---

## 4. 実装上の考慮事項

### 4.1 学習率の設定

CD での学習率 $\epsilon$ は、バッチごとに調整することが一般的：

$$\epsilon_t = \epsilon_0 \cdot \text{decay}^t$$

推奨値：$\epsilon_0 \in [0.01, 0.1]$、decay $= 0.95 \sim 0.99$

### 4.2 バッチサイズと収束

- **小バッチ** ($< 16$)： ノイズが多く、収束が不安定
- **中バッチ** ($16 \sim 128$)： 推奨される実装上の選択
- **大バッチ** ($> 256$)： 計算効率は上がるが、学習が遅い可能性

### 4.3 可視層と隠れ層の型

**二値可視層 (Binary Visible)**：
$$P(v_i = 1 | h) = \sigma(\cdots)$$

**多値可視層 (Multinomial Visible)**：
$$P(v_i^k = 1 | h) = \frac{\exp(a_i^k + \Sigma_j W_{ij}^k h_j)}{\Sigma_{k'} \exp(a_i^{k'} + \Sigma_j W_{ij}^{k'} h_j)}$$

多値版は topic modeling や recommender systems で活用される。

---

## 5. CD と EM アルゴリズムとの関係

### 5.1 EM の観点

完全な EM では：
$$Q(v, h | W_t) = \log P(v | W_t)$$

E-step： $Q$ を評価
M-step： $W$ を更新

### 5.2 CD の近似

CD は EM の以下を近似：

1. **E-step 近似**： 1 ステップの Gibbs サンプリング
2. **M-step 近似**： 単純な勾配更新（解析的解ではなく）

この近似により、計算量を大幅に削減すると同時に、実用的な収束性を保証。

---

## 6. 関連するコード実装の示唆

```python
# Pseudo-code: CD-1 learning step

def cd_update(v, W, a, b, lr):
    # Forward pass (positive phase)
    h = sigmoid(b + v @ W)  # Expectations
    h_sample = sample_bernoulli(h)

    # Negative phase (reconstruction)
    v_recon = sigmoid(a + h_sample @ W.T)
    v_recon_sample = sample_bernoulli(v_recon)

    h_recon = sigmoid(b + v_recon_sample @ W)
    h_recon_sample = sample_bernoulli(h_recon)

    # Weight update
    dW = lr * (v[:, None] @ h_sample[None, :] -
               v_recon_sample[:, None] @ h_recon_sample[None, :])
    da = lr * (v - v_recon_sample)
    db = lr * (h_sample - h_recon_sample)

    return W + dW, a + da, b + db
```

---

## 7. 現代的な位置づけ

### 7.1 RBM 時代の終焉？

2010 年代以降、RBM の人気は低下：
- **バッチ正規化** の登場により、RBM の pre-training の必要性が減少
- **残差ネットワーク (ResNet)** により、深いモデルの直接訓練が可能に

### 7.2 現在の応用

- **Quantum Physics**： 量子多体系の波動関数表現
- **Transfer Learning**： 特定分野での feature learning
- **理論研究**： 確率モデルと最適化理論の基礎研究

---

## 参考文献

- **Smolensky, P.** (1986). Information Processing in Dynamical Systems. In Rumelhart, D. E., & McClelland, J. L. (eds.), Parallel Distributed Processing.
- **Hinton, G. E.** (2002). Training Products of Experts by Minimizing Contrastive Divergence. Neural Computation, 14(8), 1771–1800.
- **Sutskever, I., & Tieleman, T.** (2010). On the Convergence Properties of Contrastive Divergence. AISTATS.
- **Fischer, A., & Igel, C.** (2012). An Introduction to Restricted Boltzmann Machines. LNCS 7441, pp. 14–36.
- **Hinton, G.** (2010). A Practical Guide to Training Restricted Boltzmann Machines. UTML TR 2010–003.
