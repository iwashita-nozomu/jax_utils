# 動的計画法

## 定式化

$N$ 層のニューラルネットワークが表現する関数を$\phi_N \in \Phi_N$ とする．$k$ 層の出力を$\phi_k \in \Phi_k^{m_k}$ とする．レイヤーで行われている計算を

$$
f_k:\Phi_{k-1}^{m_{k-1}} \to \Phi_{k}^{m_{k}} \\
\phi_{k,i} = f_{k,i}(\phi_{k-1})
$$

とする．

さらに，$\Phi_k \subset \mathcal{H}^1$ である．

ニューラルネットワークの学習における損失関数は  $J:\Phi_N\to\mathbb{R}$ とし，各段における目的関数$V_k : \Phi_k^{m_k} \to \mathbb {R}$ があるとする．

コストは最終段における目的関数 $J[\phi_N]$ のみが有効なので再帰は

$$
\left\{
    \begin{aligned}
    V_N(\phi_N) &= J[\phi_N] = J[f_N \circ \phi_{N-1}] 　\\[4pt]
    V_k(\phi_k) &= \min_{f_{k+1}} V_{k+1}(T(\phi_k, f_{k+1})) \quad 1\leq k<N
    \end{aligned}
\right.
$$

最適化問題はminの部分のことなので、ここの構築がキモ

## ラグランジアンによる導出

### 変数定義

$ \phi_k = f_k (\phi_{k-1}), \phi_{k+1} = f_{k+1} (\phi_k)$

### 単段階

最適解$ f^*_{k+1} (f_k)$

$F (f_k, f_{k+1}) = V_{k+1} (\phi _{k+1}) $

$\tilde{F} (f_k) = F(f_k,f^*_{k+1}(f_k))$

### ラグランジアン

$$

\mathcal{L}_k
= V_{k+1} (\phi_{k+1}) +
\sum_{i=1}^{m_k}{\left<p_{k,i},\phi_{k,i} - f_{k,i} (\phi_{k-1}) \right>} +
\sum_{i=1}^{m_{k+1}}{\left<p_{k+1,i},\phi_{k+1,i} - f_{k+1,i} (\phi_{k}) \right>}

$$
の全微分を考える

$$
    \delta \mathcal{L}_k
     = \sum _{i=1}^{m_{k+1}}D_{\phi_{k+1.i}} V_{k+1}(\phi_{k+1}) [\delta\phi_{k+1,i}] \\
     +\sum_{i=1}^{m_{k+1}} \left< p_{k+1,i}, \delta \phi_{k+1,i} \right> \\
    + \sum_{i=1}^{m_k}\left< p_{k,i}, \delta \phi _{k,i} \right> \\
    - \sum_{i=1}^{m_{k+1}}\sum_{j=1}^{m_k}\left< p_{k+1,i}, D_{\phi_{k,j}}f_{k+1,i}(\phi_k) [\delta \phi_{k,j}]\right> \\
    - \sum_{i=1}^{m_k}\left< p_{k,i}, \delta f_{k,i}(\phi_{k-1})\right> \\
    - \sum _{i=1}^{m_{k+1}}\left< p_{k+1,i}, \delta f_{k+1,i} (\phi_k )\right> \\
    = \sum_{i=1}^{m_{k+1}}\left<D_{\phi_{k+1,i}} V_{k+1}(\phi_{k+1}) + p_{k+1,i}, \delta \phi_{k+1,i}\right> \\
    + \sum _{i=1}^{m_k}\left< p_{k,i} - \sum_{j=1}^{m_{k+1}}(D_{\phi_{k,i}}f_{k+1,j}(\phi_k))^*p_{k+1,j}, \delta \phi_{k,i} \right> \\
    - \sum_{i=0}^{m_k}\left< p_{k,i}, \delta f_{k,i}(\phi_{k-1})\right>
     - \sum_{i=0}^{m_{k+1}}\left< p_{k+1,i}, \delta f_{k+1,i} (\phi_k )\right>
$$

と変形できる．

ただし，

$$

\sum_{i=1}^{m_{k+1}}\sum_{j=1}^{m_k}\left< p_{k+1,i}, D_{\phi_{k,j}}f_{k+1,i}(\phi_k) [\delta \phi_{k,j}]\right> \\
=\sum_{i=1}^{m_{k+1}}\sum_{j=1}^{m_k}\left< D_{\phi_{k,j}}f_{k+1,i}(\phi_k) ^* p_{k+1,i}, \delta \phi_{k,j}\right> \\
=\sum_{i=1}^{m_k}\sum_{j=1}^{m_{k+1}}\left< D_{\phi_{k,i}}f_{k+1,j}(\phi_k) ^* p_{k+1,j}, \delta \phi_{k,i}\right> \\
=\sum_{i=1}^{m_k}\left< \sum_{j=1}^{m_{k+1}} D_{\phi_{k,i}}f_{k+1,j}(\phi_k) ^* p_{k+1,j}, \delta \phi_{k,i}\right>

$$

### 随伴変数による消去

$$

\begin{align*}
D_{\phi_{k+1,i}} V_{k+1}(\phi_{k+1}) + p_{k+1,i} =0 && i = 1,...,m_{k+1}\\
p_{k,i} - \sum_{j=1}^{m_{k+1}} D_{\phi_{k,i}}f_{k+1,j}(\phi_k) ^* p_{k+1,j} =0 && i =1 ,...,m_k
\end{align*}
$$

により，

$$
\delta \mathcal{L}_k = - \sum_{i=1}^{m_k}\left< p_{k,i}, \delta f_{k,i}(\phi_{k-1})\right> - \sum_{i=1}^{m_{k+1}}\left< p_{k+1,i}, \delta f_{k+1,i} (\phi_k )\right>
$$

### 後段の最適性条件

後段を最適化するとすると，

$$
\delta \mathcal{L} = - \sum_{i=1}^{m_k}\left< p_{k,i}, \delta f_{k,i}(\phi_{k-1})\right>
$$

となる．すると，

$$D_{f_{k,i}} \tilde{F} (f_k) [h_i] = - \left< p_{k,i}, h_i(\phi_{k-1})\right>$$

となる．よって，$\Delta f_{k,i} = p_{k,i} $ が最急方向である．

### 逆伝播状態

上段からは$p_{k+1}$ を受け，$p_k$を計算し，更新を行い，$p_k$を下層に伝播する．しかしながら，こいつは関数空間の双対であり，実装はむずいのでは？

## パラメータ（有限次元ベクトル空間）

$f_{k,i}(\phi_{k-1}) = f_{k,i}(\phi_{k-1};\theta_{k,i})$ とする．

これは，$\phi_{k-1}, f_k$ が生成する多様体 $\mathcal{M}_k$ 上の元として

$$
\mathcal{M}_k := \{f_k^{(i)}(\phi_{k-1};\theta_{k,i})|\theta_{k,i} \in \Theta_k\}
$$

$$
\phi_{k,1},...,\phi_{k,m_k} \in \mathcal{M}_k
$$

が表せる．

簡略に$f_{k,i} = f_k(\cdot,\theta_i)$ と書くことにする．

### 勾配のパラメータ化

$\mathcal{M}_k \subseteq \Phi_k$ なので， 当然$\mathcal{M}_k$にも内積が誘導されている．

このとき，$\phi_{k,i}$ 近傍での接空間が定まって

$$
\mathcal{T}_{\phi_{k,i}}\mathcal{M}_k = span\{\partial_{\theta_k^ \alpha}f_k (\phi_{k-1};\theta_k)\}
$$



## 式展開

標準的なニューラルネットワークでは $\Theta_k = \mathbb{R}^{m_{k-1}} \times \mathbb{R}$ なので

$$
f_k(\phi_{k-1};\theta_k) = \sigma(w^\top \phi_{k-1} + b)
$$

である

$$
\begin{align*}
    \partial_w f_k &= \sigma'(w^\top \phi_{k-1} + b) \phi_{k-1} \\
    \partial_b f_k &= \sigma'(w^\top \phi_{k-1} + b)

\end{align*}
$$

すると，計量

$$
\begin{align*}
    g_{k,i} &= \left<\partial_{\theta_k}f_k,\partial_{\theta_k}f_k\right>\\
\end{align*}
$$
