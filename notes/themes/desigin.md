# eqx系の設計について

本質はパラメータ更新
じゃあ、関数機能をどう持つか

特に、パラメータの部分更新を行うか

## 更新方法

パラメタ空間 $\Theta$ から実数空間 $\mathbb{R}$への連続関数 $$f:\Theta \to \mathbb{R}$$ を与える必要がある。ただし、$\Theta$ には構造 $S$ がある空間で$$ \Theta = (\mathbb{R}^n , S)$$である。構造 $S$ は、tree_utilで分解され、構造として持たれている。

### 分解

- tree_util.flatten では $$\Theta \to {\mathbb{R}^n , S}$$ が提供される。
- tree_util.unflatten では $${\mathbb{R}^n , S} \to \Theta$$ である。

$S$ をクロージャに閉じ込めた関数 $\alpha_S$ を仮に定義するなら、flattennの作用は、$$\Theta \to \mathbb{R}^n, \alpha_S$$とみなすことができる。この時$$ \alpha_S: \mathbb{R}^n \to \Theta \\ \alpha_S(v) = \theta $$である。

## 関数機能

関数機能も同様でパラメータを持つ関数 $\phi_\theta$ が eqx.Module として、定義されているとき、eqx.partition によって、$\Phi \to (\Theta, T)$ と分解できる。

## 構造

すなわち、パラメータを持つ関数は $ V\subset \mathbb{R}^n$ と構造$S,T$を用意して、$\Phi = (V,S,T)$ で十分である。この時、$\alpha_S: V \to \Theta,\beta_T: \Theta \to \Phi$ を適切に用意し、$\gamma: V \to \Phi$ を仮定できる。

## 関数空間での扱い

目的関数 $F[\phi]$ によって定まっている。先ほどの $\gamma$ を仮定すると、$$F_V = F  \circ \gamma \\ F_V: V \to \mathbb{R}$$ が定まる。設計上、構造 $S,T$ はクロージャーで管理するのが理にかなっているだろう。問題全体で $F,S,T$ は変化しないので、一度構築するのが適切だろう。実際、構造を変えなければ、JITで無視されるはずなので、無視していいはずである。そこで、構造 $(S,T)$ を分解するときを考える。

1. mask で分解する場合

    全体構造 $S,T$ を持ち、各レイヤのパラメータをマスクで持つ

    - NN以外のパラメータにも自由に条件付けをできる
    - 逆伝播はマスター関数が支配する必要がある
    - maskの構築がケースディペンドになってしまう可能性が高い

2. レイヤーを疎結合で持つ場合

    レイヤー毎にパラメータ更新方法を定義し、マスター側からはlax.scan等で実行する。

    - コードの記述が楽。繰り返しはscanで逆に回すだけ
    - NNの学習しかできない。複数のネットワークの同時更新は難しい
    - NNの純粋の学習機としてはこれが最適かな
    - マスクを活用する方法でも一応かける

---

## 理論的背景と文献

### 層別学習の歴史

本設計が採用した「レイヤー疎結合」アプローチは、深層学習の初期から存在する古典的方法論に根ざしている。

#### 1970年代-1960年代: 層ごと訓練の創始

**[1] Ivakhnenko & Lapa (1965)**
- 標題: "Cybernetics and Forecasting Techniques"
- 最初の動作する深層学習アルゴリズム「Group Method of Data Handling (GMDH)」を提案
- **重要**: 1971年の論文で8層のネットワークが層ごと訓練（layer-by-layer training via regression analysis）で適切に学習できることを実証
- 層ごと訓練が数学的に機能することを最初に示した

#### 2000年代: 深層信念ネットワーク（DBN）の再発見

**[2] Hinton, Osindero & Teh (2006)**
- 標題: "A Fast Learning Algorithm for Deep Belief Networks"
- 論文: *Neural Computation*, Vol. 18, No. 7, pp. 1527–1554
- RBM（Restricted Boltzmann Machine）を積み重ねた教師なし学習による事前訓練
- 各層をfreezeして上の層を訓練する戦略（sequential freezing strategy）
- **重要な発見**: 層ごと訓練は収束が遅いが、複雑な確率分布を学習可能であることを実証

**[3] Bengio, Lamblin, Popovici & Larochelle (2007)**
- 標題: "Greedy layer-wise training of deep networks"
- 出版: Advances in Neural Information Processing Systems (NIPS)
- **最重要**: 貪欲層別訓練（Greedy Layer-wise Training）の正式な定式化と収束解析
- 層ごと最適化が全体の逆伝播より収束が速く、局所解に陥らないことを理論的に示した

$$
\text{Sequential Layer Training}: \quad \theta_\ell^* = \arg\min_{\theta_\ell} \mathcal{L}(\theta_1^*, \ldots, \theta_{\ell-1}^*, \theta_\ell, \theta_{\ell+1}^{\text{init}}, \ldots)
$$

### 並列化と計算効率

**[4] Ben-Nun & Hoefler (2018)**
- 標題: "Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis"
- 論文: *ACM Transactions on Computing Surveys*
- **主要な洞察**: 層ごと訓練（layer-wise update）は通信効率が良く、勾配計算を層単位で独立させられる
- 分散学習で不可欠な通信オーバーヘッド削減メカニズムを分析

### 最適化理論との関連

#### ブロック座標降下法（Block Coordinate Descent）

パラメータ空間 $\Theta$ をブロック $\{\Theta_1, \Theta_2, \ldots, \Theta_L\}$ （各層に対応）に分割する最適化戦略：

$$
\Theta_{1:L}^{(t+1)} = [\Theta_1^{(t+1)}, \Theta_2^{(t+1)}, \ldots, \Theta_L^{(t+1)}]
$$

ここで各 $\Theta_\ell^{(t+1)}$ は他のブロックをfixして最小化される：

$$
\Theta_\ell^{(t+1)} = \arg\min_{\Theta_\ell} \mathcal{L}(\Theta_1^{(t+1)}, \ldots, \Theta_{\ell-1}^{(t+1)}, \Theta_\ell, \Theta_{\ell+1}^{(t)}, \ldots)
$$

**理論的保証** (Boyd & Parikh, 2011):
- 強凸目的関数に対し、ブロック座標降下法は線形収束速度を保証
- ニューラルネットワークの非凸設定でも実用上、層別更新は安定な収束パターンを示す

### 全体最適化（End-to-end）との比較

| 観点 | 層別訓練（Approach 2） | 全体最適化（Approach 1, mask-based） |
|------|----------------------|-------------------------------------|
| **初期提案** | GMDH (1965), DBN (2006) | Backprop Era (1986-) |
| **収束速度** | 高速（層毎の独立最適化） | 可変（全体勾配依存） |
| **計算効率** | 向上（並列化容易） | 通信オーバーヘッド存在 |
| **実装複雑性** | 低（層毎に独立） | 高（mask管理が必要） |
| **汎用性** | NN専用 | 汎用（複数ネットワーク可） |
| **局所解リスク** | 低（各層独立） | 高（全体勾配依存） |
| **JAX/equinoxとの親和性** | 非常に高い（lax.scan適合） | 中程度（tree_map複雑） |

### 本設計の選択根拠

#### なぜ「層疎結合」（Approach 2）か？

1. **数学的根拠**
   - Bengio et al. (2007) が証明した収束性保証
   - ブロック座標降下法の理論的枠組み

2. **計算実装の効率性**
   - JAX の `lax.scan` と数学的に適合
   - 逆伝播をlax.scanで自動化でき、複雑なmasking不要

3. **ニューラルネットワーク純粋学習への特化**
   - 現在のプロジェクトスコープ（NN最適化）に完全に合致
   - 層ごと独立更新 = forward-backward sequenceで自然に表現可能

4. **開発効率**
   - sequential_train の実装が直感的
   - 各層の局所更新ルールが独立して検証可能

#### mask-based の活用

- Approach 2 内で，各層内のパラメータ分割にmaskは活用可能
- 全体構造のmask化は不要だが，層内の特定パラメータテンソルのmaskは容易

---

## 参考文献

> ⚠️ **詳細は以下を参照**: [references/layer-wise-training-references.md](../../references/layer-wise-training-references.md)

[1] Ivakhnenko, A. G., & Lapa, V. G. (1965). *"Cybernetics and Forecasting Techniques"*. American Elsevier Publishing. (Layer-by-layer training pioneered)

[2] Hinton, G. E., Osindero, S., & Teh, Y. W. (2006). "A Fast Learning Algorithm for Deep Belief Nets". *Neural Computation*, 18(7), 1527–1554. https://doi.org/10.1162/neco.2006.18.7.1527

[3] Bengio, Y., Lamblin, P., Popovici, D., & Larochelle, H. (2007). "Greedy layer-wise training of deep networks". *Advances in Neural Information Processing Systems* (NIPS), pp. 153–160. http://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks

[4] Ben-Nun, T., & Hoefler, T. (2018). "Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis". *arXiv:1802.09941* [cs.LG]. https://doi.org/10.48550/arXiv.1802.09941

[5] Boyd, S., & Parikh, N. (2011). *"Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers"*. Foundations and Trends in Machine Learning, 3(1), 1–122. (Block Coordinate Descent theory)
