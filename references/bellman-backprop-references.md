# Bellman 型逆伝播と双対・計量の参考文献

## 概要

本ドキュメントは [`notes/themes/bellman_bp.md`](../notes/themes/bellman_bp.md) で使っている
Bellman 型の逆伝播を，以下の観点から支える参考文献をまとめたものです。

- 逆伝播を「双対元の pullback」としてどう捉えるか
- 関数空間の双対とパラメータ空間の双対をどう区別するか
- Riesz 写像と質量行列がどの段階で必要になるか
- 関数空間の計量をパラメータ空間に引き戻したときの Gram / mass matrix の意味

本ノートでの主張を先にまとめると，次の 3 点です。

1. 逆伝播の本体は dual space 上の pullback であり，この段階では質量行列は不要である。
2. 双対元を primal 側の勾配ベクトルとして表したいとき，Riesz 写像が必要になる。
3. 関数空間の計量をパラメータ空間に引き戻すと，座標表示では
   $M_{\Theta_k}^{\mathrm{nat}} = B_k^\top M_{\Phi_k} B_k$
   という Gram / mass matrix が現れる。

ここで

$$
B_k := D_{\theta_k} f_k(\phi_{k-1}; \theta_k) : \Theta_k \to \Phi_k
$$

である。

---

## 主要参考文献

### [1] Baydin et al. (2018): reverse-mode AD は cotangent の伝播

**文献情報**
- 著者: Atilim G. Baydin, Barak A. Pearlmutter, Alexey A. Radul, Jeffrey Mark Siskind
- 論文: *Automatic Differentiation in Machine Learning: a Survey*
- 掲載誌: *Journal of Machine Learning Research*, 18(153), 1-43
- 出版年: 2018
- URL: https://www.jmlr.org/papers/v18/17-468.html

**要点**
- reverse-mode AD は，順方向の計算グラフに対して双対量を後ろ向きに伝播する手続きとして整理される。
- survey では backpropagation を AD の special case と位置づけている。
- 実装上の基本演算は transposed Jacobian-vector product である。

**本ノートとの対応**
- 本ノートでの
  $$
  p_{k-1} = A_k^\star p_k, \qquad \tilde p_k = B_k^\star p_k
  $$
  という書き方は，reverse-mode AD を cotangent の pullback と見る立場そのものである。
- ここでの $p_k \in \Phi_k^*$ と $\tilde p_k \in \Theta_k^*$ は「勾配ベクトル」ではなく dual element だと読むのが自然である。

---

### [2] Frostig et al. (2021): reverse-mode は linearization + transposition

**文献情報**
- 著者: Roy Frostig, Matthew J. Johnson, Dougal Maclaurin, Adam Paszke, Alexey Radul
- 論文: *Decomposing reverse-mode automatic differentiation*
- 出版: arXiv:2105.09469
- 出版年: 2021
- URL: https://arxiv.org/abs/2105.09469

**要点**
- reverse-mode AD を「線形化」と「転置」の合成として定式化している。
- 非線形写像をまず線形化し，その後に線形写像の transposition rule を適用するという見方を明確化している。
- reverse-mode の本質を dual space 上の転置作用として切り出している。

**本ノートとの対応**
- 各層に対して
  $$
  A_k := D_{\phi_{k-1}} f_k, \qquad B_k := D_{\theta_k} f_k
  $$
  とおくと，逆伝播は $A_k^\star, B_k^\star$ の適用として読める。
- この文献から直接得られる教訓は，「転置」の段階ではまだ primal 側の内積を選ぶ必要がないということである。

---

### [3] Schwedes, Funke, Ham (2016): Riesz 表現を取らないと mesh-dependent になる

**文献情報**
- 著者: Tobias Schwedes, Simon W. Funke, David A. Ham
- 論文: *An iteration count estimate for a mesh-dependent steepest descent method based on finite elements and Riesz inner product representation*
- 出版: arXiv:1606.08069
- 出版年: 2016
- URL: https://arxiv.org/abs/1606.08069

**要点**
- 関数空間最適化では，導関数は dual space に属し，更新方向を得るには Riesz 表現が必要である。
- 離散化後に単純な Euclidean 勾配を使うと，連続問題の計量とずれて mesh-dependent な挙動を起こしうる。
- どの inner product で gradient を定義するかが，反復法の挙動に本質的に効く。

**本ノートとの対応**
- 本ノートで $p_k \in \Phi_k^*$ をそのまま backward に流すのはよいが，
  そこから更新方向を作るときに
  $$
  g_k = R_{\Phi_k}^{-1} p_k
  $$
  を通さずに Euclidean 的に読んでしまうのは危険である。
- 特に $\phi_k$ を「入力から中間表現への関数」とみなして関数空間として扱うなら，
  Riesz 写像またはその離散表現である質量行列を無視しない方が筋がよい。

---

### [4] dolfin-adjoint: gradient は dual space にあり，必要なら Riesz representation を取る

**文献情報**
- 著者: dolfin-adjoint / pyadjoint documentation
- 題目: *Differentiating functionals*
- URL: https://www.dolfin-adjoint.org/en/release/documentation/maths/3-gradients.html

**要点**
- PDE constrained optimization の文脈で，機能の微分は自然には dual space にあることを明示している。
- そのままでは可視化や更新に使いづらいため，指定した inner product に関する Riesz representation を返す，という実装方針を説明している。
- `riesz_representation="L2"` のような指定は，dual element を primal 側の gradient に変換する操作に対応する。

**本ノートとの対応**
- いま議論している「質量行列が必要か」という問いに対して，この資料はかなり直接的である。
- 逆伝播そのものは dual の伝播であり，更新のために primal の gradient が欲しい段階で Riesz representation が必要になる。

---

### [5] dolfin-adjoint Klein bottle example: `compute_gradient` は Riesz 表現を返せる

**文献情報**
- 著者: Sebastian Mitusch
- 題目: *Sensitivity analysis of the heat equation on a Gray’s Klein bottle*
- URL: https://www.dolfin-adjoint.org/en/release/documentation/klein/klein.html

**要点**
- 例題では `compute_gradient(J, m, options={"riesz_representation": "L2"})` を使い，
  勾配を dual ではなく primal 側の関数として返している。
- ドキュメント中でも，これを行わないと dual のままであり，そのままでは扱いにくいことが述べられている。

**本ノートとの対応**
- 関数空間側で質量行列が必要になるのは，まさにこの dual-to-primal 変換の段階である。
- Bellman 型逆伝播に引きつけて言えば，$p_k$ をそのまま次層へ渡すのと，
  $g_k = R_{\Phi_k}^{-1} p_k$ を作って更新に使うのは別の段階である。

---

### [6] van Erp et al. (2022): 引き戻し計量の Gram 行列

**文献情報**
- 著者: Thijs van Erp ほか
- 論文: *Invariance properties of the natural gradient in overparametrised systems*
- 掲載誌: *Information Geometry*
- 出版年: 2022
- URL: https://link.springer.com/article/10.1007/s41884-022-00067-9

**要点**
- モデル空間上の計量をパラメータ化を通じて引き戻すと，パラメータ空間には Gram 行列
  $$
  G_{ij} = g(\partial_i,\partial_j)
  $$
  が現れる。
- 過剰パラメータ化された場合には Gram 行列が退化しうるため，Moore-Penrose 逆を使う議論が必要になる。

**本ノートとの対応**
- 本ノートの記法では，この Gram 行列は
  $$
  M_{\Theta_k}^{\mathrm{nat}} = B_k^\top M_{\Phi_k} B_k
  $$
  に対応する，というのが自然な読みである。
- これは文献の記号を本ノートの $B_k = D_{\theta_k} f_k$ に写した解釈であり，
  その意味でここは文献内容を本プロジェクトの記法に引き直した整理である。

---

## 補助参考文献

### [7] Rumelhart, Hinton, Williams (1986)

**文献情報**
- 著者: David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams
- 論文: *Learning representations by back-propagating errors*
- 掲載誌: *Nature*, 323, 533-536
- 出版年: 1986
- URL: https://www.nature.com/articles/323533a0

**要点**
- 多層ネットワークにおける backpropagation の古典的定式化。
- 実用上の重み勾配の形を与える一次資料。

**本ノートとの対応**
- 本文で必要なのは主に dual と計量の議論だが，
  線形層 + 活性化の具体式を通常の backprop と照合するときの参照元として有用である。

---

### [8] Chen et al. (2018)

**文献情報**
- 著者: Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud
- 論文: *Neural Ordinary Differential Equations*
- 出版: arXiv:1806.07366
- 出版年: 2018
- URL: https://arxiv.org/abs/1806.07366

**要点**
- continuous-time の adjoint method により，状態随伴とパラメータ勾配を同時に記述する。
- 離散層の逆伝播を連続時間極限で見たときの比較対象になる。

**本ノートとの対応**
- Bellman / adjoint 的な見方を連続時間へ延長するとどうなるかを確認する補助資料。
- 今回の「質量行列をどこで入れるか」という論点への直接文献ではないが，
  状態随伴とパラメータ勾配を分けて扱う視点を補強する。

---

## いまの設計に対する文献ベースのまとめ

### 1. 双対としての逆伝播

文献 [1], [2] から，逆伝播の本体は dual space 上の pullback とみなすのが自然である。
したがって

$$
p_k \in \Phi_k^*, \qquad \tilde p_k \in \Theta_k^*
$$

をまず dual element として持つのがよい。

### 2. 質量行列が必要になる段階

文献 [3], [4], [5] から，質量行列や Riesz 写像が必要になるのは，
dual element を primal 側の勾配ベクトルに変換して更新に使う段階である。

### 3. パラメータ空間の自然な計量

文献 [6] の Gram 行列の議論を本ノートの記法に引き直すと，
関数空間の内積を $B_k = D_{\theta_k}f_k$ で引き戻した自然なパラメータ計量は

$$
\langle \xi, \eta \rangle_{\Theta_k}^{\mathrm{nat}}
:=
\langle B_k \xi, B_k \eta \rangle_{\Phi_k}
$$

であり，その座標表示は

$$
M_{\Theta_k}^{\mathrm{nat}} = B_k^\top M_{\Phi_k} B_k
$$

となる。

### 4. 実務上の読み替え

したがって，もし本当に「関数空間側の計量から自然に誘導されたパラメータ更新」をしたいなら，
更新方向 $\Delta \theta_k$ は単に $-\tilde p_k$ ではなく，

$$
R_{\Theta_k}^{\mathrm{nat}} \Delta \theta_k = - \tilde p_k
$$

あるいは座標表示で

$$
M_{\Theta_k}^{\mathrm{nat}} \Delta \theta_k = - \tilde g_k
$$

のように解く必要がある。

---

## 参考URL一覧

1. https://www.jmlr.org/papers/v18/17-468.html
2. https://arxiv.org/abs/2105.09469
3. https://arxiv.org/abs/1606.08069
4. https://www.dolfin-adjoint.org/en/release/documentation/maths/3-gradients.html
5. https://www.dolfin-adjoint.org/en/release/documentation/klein/klein.html
6. https://link.springer.com/article/10.1007/s41884-022-00067-9
7. https://www.nature.com/articles/323533a0
8. https://arxiv.org/abs/1806.07366
