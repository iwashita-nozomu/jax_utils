# Bellman BP Chat Summary

`bellman_bp.md` を詰める過程で，チャット上で整理した論点を残すための補助メモ．
元ファイルの本文ではなく，議論の履歴として置いておく．

## 問題設定

`N` 層のネットワークを考える．各段の状態空間 `\Phi_k` は Hilbert 空間であり，

$$
\phi_k = (\phi_{k,1},\dots,\phi_{k,m_k}) \in \Phi_k^{m_k}
$$

を `k` 層の出力とする．ここで `\phi_{k,i}` は `i` 番目ユニットが表す関数である．

各層のユニットは，同じ関数形

$$
f_k(\cdot;\theta)
$$

を共有し，`i` 番目ユニットの出力は

$$
\phi_{k,i} = f_k(\phi_{k-1};\theta_{k,i})
$$

で与えられる．したがって層全体のパラメータは

$$
\theta_k = (\theta_{k,1},\dots,\theta_{k,m_k}) \in \Theta_k^{layer},
\qquad
\Theta_k^{layer} = \prod_{i=1}^{m_k}\Theta_k^{unit}
$$

である．全結合層ではこの `\theta_k` の具体的な座標表示が `(W_k,b_k)` である．

最終出力に対して損失関数

$$
J : \Phi_N^{m_N} \to \mathbb R
$$

を与え，Bellman 的には各段の価値関数

$$
V_k : \Phi_k^{m_k} \to \mathbb R
$$

を考える．理論上は，関数空間側の随伴

$$
p_k = (p_{k,1},\dots,p_{k,m_k}) \in (\Phi_k^*)^{m_k}
$$

が逆伝播する．

本メモの主題は，この `p_k` をそのまま保持するのではなく，

1. 可到達集合とその接空間に局所化すること
2. 関数空間での最急法を先に記述し，それを `\theta` に引き戻すこと
3. 下層への伝播も含めた backward 設計へ落とすこと

の 3 点である．

## Bellman 的な局所問題

終端では

$$
V_N(\phi_N) = J(\phi_N)
$$

であり，各段では

$$
V_k(\phi_k) = \min_{f_{k+1}} V_{k+1}(f_{k+1}(\phi_k))
$$

を考える．

上位層が最適化済みであるとみなすと，`k` 層で実際に解くべき reduced problem は，固定した `\phi_{k-1}` に対して

$$
\widetilde F_k(\theta_k)
:=
V_{k+1}\bigl(\phi_{k+1}(\theta_k)\bigr)
$$

を `\theta_k` で最小化することである．ここで

$$
\phi_k(\theta_k)
=
\bigl(
f_k(\phi_{k-1};\theta_{k,1}),
\dots,
f_k(\phi_{k-1};\theta_{k,m_k})
\bigr)
$$

であり，`\phi_{k+1}(\theta_k)` はその後段の写像を通した結果である．

同時に，関数空間側では current layer の出力 `\phi_k` を変数とみなした reduced functional

$$
\widehat F_k(\phi_k)
:=
V_{k+1}\bigl(f_{k+1}^*(\phi_k)(\phi_k)\bigr)
$$

を考えてよい．ここで `f_{k+1}^*(\phi_k)` は後段の最適応答を表す記号である．
随伴方程式は，この reduced problem の微分を後段の最適化を陽に微分せずに書くための装置になっている．

## 可到達集合と接空間

固定した `\phi_{k-1}` に対し，1 ユニットが取りうる関数の族を

$$
\mathcal M_k(\phi_{k-1})
:=
\{ f_k(\phi_{k-1};\theta) \mid \theta \in \Theta_k^{unit} \}
\subset \Phi_k
$$

と置くと

$$
\phi_{k,i} \in \mathcal M_k(\phi_{k-1})
$$

と書ける．

ただし実際に必要なのは `\mathcal M_k` 全体より，その点での接空間である．
Taylor 展開の一次の項から

$$
f_k(\phi_{k-1};\theta_{k,i}+\delta\theta)
=
\phi_{k,i}
+
D_\theta f_k(\phi_{k-1};\theta_{k,i})[\delta\theta]
+
o(\|\delta\theta\|)
$$

なので，

$$
T_{\phi_{k,i}}\mathcal M_k
=
\left\{
D_\theta f_k(\phi_{k-1};\theta_{k,i})[\delta\theta]
\mid
\delta\theta \in \Theta_k^{unit}
\right\}
=
\operatorname{Im} D_\theta f_k(\phi_{k-1};\theta_{k,i})
$$

である．

## 随伴変数による導出

`k` 層と `k+1` 層に注目し，状態変数 `\phi_k,\phi_{k+1}` を独立変数として扱う．
制約付き問題

$$
\phi_{k,i} = f_k(\phi_{k-1};\theta_{k,i}),
\qquad
\phi_{k+1,j} = f_{k+1}(\phi_k;\theta_{k+1,j})
$$

に対して，ラグランジアンを

$$
\mathcal L_k
=
V_{k+1}(\phi_{k+1})
+
\sum_{i=1}^{m_k}
\left<
p_{k,i},
\phi_{k,i}-f_k(\phi_{k-1};\theta_{k,i})
\right>
+
\sum_{j=1}^{m_{k+1}}
\left<
p_{k+1,j},
\phi_{k+1,j}-f_{k+1}(\phi_k;\theta_{k+1,j})
\right>
$$

とおく．

この全微分は

$$
\begin{aligned}
\delta \mathcal L_k
=\;&
\sum_{j=1}^{m_{k+1}}
D_{\phi_{k+1,j}}V_{k+1}(\phi_{k+1})[\delta\phi_{k+1,j}]
+
\sum_{j=1}^{m_{k+1}}
\langle p_{k+1,j},\delta\phi_{k+1,j}\rangle
\\
&+
\sum_{i=1}^{m_k}
\langle p_{k,i},\delta\phi_{k,i}\rangle
-
\sum_{j=1}^{m_{k+1}}
\left<
p_{k+1,j},
D_{\phi_k}f_{k+1}(\phi_k;\theta_{k+1,j})[\delta\phi_k]
\right>
\\
&-
\sum_{i=1}^{m_k}
\left<
p_{k,i},
D_{\theta}f_k(\phi_{k-1};\theta_{k,i})[\delta\theta_{k,i}]
\right>
-
\sum_{j=1}^{m_{k+1}}
\left<
p_{k+1,j},
D_{\theta}f_{k+1}(\phi_k;\theta_{k+1,j})[\delta\theta_{k+1,j}]
\right>.
\end{aligned}
$$

第 3 項と第 4 項をまとめると，

$$
\sum_{i=1}^{m_k}
\left<
p_{k,i}
-
\sum_{j=1}^{m_{k+1}}
\bigl(D_{\phi_{k,i}}f_{k+1}(\phi_k;\theta_{k+1,j})\bigr)^*p_{k+1,j},
\delta\phi_{k,i}
\right>
$$

となるので，

$$
\begin{aligned}
\delta \mathcal L_k
=\;&
\sum_{j=1}^{m_{k+1}}
\left<
D_{\phi_{k+1,j}}V_{k+1}(\phi_{k+1}) + p_{k+1,j},
\delta\phi_{k+1,j}
\right>
\\
&+
\sum_{i=1}^{m_k}
\left<
p_{k,i}
-
\sum_{j=1}^{m_{k+1}}
\bigl(D_{\phi_{k,i}}f_{k+1}(\phi_k;\theta_{k+1,j})\bigr)^*p_{k+1,j},
\delta\phi_{k,i}
\right>
\\
&-
\sum_{i=1}^{m_k}
\left<
p_{k,i},
D_{\theta}f_k(\phi_{k-1};\theta_{k,i})[\delta\theta_{k,i}]
\right>
\\
&-
\sum_{j=1}^{m_{k+1}}
\left<
p_{k+1,j},
D_{\theta}f_{k+1}(\phi_k;\theta_{k+1,j})[\delta\theta_{k+1,j}]
\right>.
\end{aligned}
$$

ここで随伴方程式

$$
D_{\phi_{k+1,j}}V_{k+1}(\phi_{k+1}) + p_{k+1,j} = 0,
\qquad
p_{k,i}
=
\sum_{j=1}^{m_{k+1}}
\bigl(D_{\phi_{k,i}}f_{k+1}(\phi_k;\theta_{k+1,j})\bigr)^*p_{k+1,j}
$$

を課すと，

$$
\delta \mathcal L_k
=
-
\sum_{i=1}^{m_k}
\left<
p_{k,i},
D_{\theta}f_k(\phi_{k-1};\theta_{k,i})[\delta\theta_{k,i}]
\right>
-
\sum_{j=1}^{m_{k+1}}
\left<
p_{k+1,j},
D_{\theta}f_{k+1}(\phi_k;\theta_{k+1,j})[\delta\theta_{k+1,j}]
\right>
$$

となる．

上位層 `k+1` がすでに最適化されていて，その一次変分が消えるとみなせば，

$$
\delta \widetilde F_k
=
-
\sum_{i=1}^{m_k}
\left<
p_{k,i},
D_{\theta}f_k(\phi_{k-1};\theta_{k,i})[\delta\theta_{k,i}]
\right>
$$

であり，同時に関数空間側では admissible な変分 `\delta\phi_{k,i}\in T_{\phi_{k,i}}\mathcal M_k` に対して

$$
D\widehat F_k(\phi_k)[\delta\phi_k]
=
-
\sum_{i=1}^{m_k}
\langle p_{k,i},\delta\phi_{k,i}\rangle
$$

と読むことができる．

## 関数空間での最急法

`\Phi_k` は Hilbert 空間なので，Riesz 写像

$$
R_{\Phi_k} : \Phi_k \to \Phi_k^*
$$

を通じて

$$
g_{k,i} := R_{\Phi_k}^{-1}p_{k,i} \in \Phi_k
$$

と表せる．

ただし実際に到達可能な変分は `T_{\phi_{k,i}}\mathcal M_k` に限られるので，
関数空間での最急降下方向は

$$
\delta\phi_{k,i}^{sd}
:=
\arg\min_{\delta\phi \in T_{\phi_{k,i}}\mathcal M_k}
\left\{
D\widehat F_k(\phi_k)[\delta\phi]
+
\frac12 \|\delta\phi\|_{\Phi_k}^2
\right\}
$$

で定義される．

これは `-g_{k,i}` の接空間への射影とみなせる．

## パラメータ側への引き戻し

パラメータ写像

$$
\Psi_{k,i}(\theta) := f_k(\phi_{k-1};\theta)
$$

を使うと，パラメータ側の covector は

$$
\tilde p_{k,i}
:=
(D\Psi_{k,i}(\theta_{k,i}))^*p_{k,i}
\in
(\Theta_k^{unit})^*
$$

で与えられる．

座標 `\theta=(\theta^1,\dots,\theta^d)` を取り，

$$
v_a := \partial_{\theta^a} f_k(\phi_{k-1};\theta_{k,i}) \in T_{\phi_{k,i}}\mathcal M_k
$$

とおくと，

$$
(\tilde p_{k,i})_a
=
p_{k,i}[v_a]
=
\langle g_{k,i}, v_a \rangle_{\Phi_k}
$$

である．

また reduced functional の `\theta_{k,i}` 方向の微分は

$$
D\widetilde F_k(\theta_{k,i})[\delta\theta]
=
-\,\tilde p_{k,i}[\delta\theta]
$$

と書ける．

## 自然計量と `\theta` 空間での最急法

接空間上の内積をパラメータ空間に引き戻すと，自然計量

$$
G^{(i)}_{ab}
:=
\langle v_a, v_b \rangle_{\Phi_k}
=
\left\langle
\partial_{\theta^a} f_k(\phi_{k-1};\theta_{k,i}),
\partial_{\theta^b} f_k(\phi_{k-1};\theta_{k,i})
\right\rangle_{\Phi_k}
$$

が得られる．これは局所的な Gram 行列であり，有限次元表示では質量行列として振る舞う．

関数空間での最急法を `\delta\phi = D_\theta f_k(\phi_{k-1};\theta_{k,i})[\delta\theta]` を通じて `\theta` 側へ移すと，

$$
\delta\theta_{k,i}^{sd}
:=
\arg\min_{\delta\theta}
\left\{
D\widetilde F_k(\theta_{k,i})[\delta\theta]
+
\frac12
\|D_\theta f_k(\phi_{k-1};\theta_{k,i})[\delta\theta]\|_{\Phi_k}^2
\right\}
$$

となる．その Euler-Lagrange 方程式は

$$
G^{(i)} \delta\theta_{k,i}^{sd} = - \tilde p_{k,i}
$$

であり，ステップ幅 `\eta_k > 0` を用いた最急法は

$$
\theta_{k,i}^{new}
=
\theta_{k,i} - \eta_k (G^{(i)})^{-1}\tilde p_{k,i}
$$

である．退化している場合は

$$
\theta_{k,i}^{new}
=
\theta_{k,i} - \eta_k (G^{(i)})^{+}\tilde p_{k,i}
$$

とし，`+` は Moore-Penrose 擬似逆を表す．

層全体では，直積計量

$$
G_k
=
\operatorname{diag}(G^{(1)},\dots,G^{(m_k)})
$$

を用いて

$$
\theta_k^{new}
=
\theta_k - \eta_k G_k^{+}\tilde p_k
$$

とまとめて書ける．

## 何を逆伝播させるべきか

`\tilde p_k` だけでは 1 層下に情報を流すには足りない．
なぜなら，

$$
\tilde p_k = (D_{\theta_k} f_k)^* p_k
$$

はパラメータ更新用の情報しか持たず，下層への伝播には

$$
p_{k-1} = (D_{\phi_{k-1}} f_k)^* p_k
$$

が別に必要だからである．

したがって実装では，`p_k` を丸ごと explicit な関数として持つ代わりに，
その作用だけを持ち運ぶ見方が自然である．

## 逆伝播の設計案

### 作用素としての持ち運び

`p_k` を値として持つ代わりに，線形汎関数

$$
\Lambda_k : T_{\phi_k}\mathcal M_k \to \mathbb R,
\qquad
\Lambda_k(h) = p_k[h]
$$

として持ち運ぶことを考える．

このとき下層への伝播は

$$
\Lambda_{k-1}(u)
:=
\Lambda_k(D_{\phi_{k-1}} f_k[u])
$$

であり，パラメータ側は

$$
\widetilde{\Lambda}_k(\delta\theta_k)
:=
\Lambda_k(D_{\theta_k} f_k[\delta\theta_k])
$$

で与えられる．

これは，随伴方程式を「作用素の合成」として持ち運ぶ見方である．

### 1 層の backward API

`k` 層の backward は，forward で保存した

$$
(\phi_{k-1},\theta_k,\phi_k)
$$

をキャッシュとして持ち，上位から incoming adjoint action

$$
\Lambda_k : T_{\phi_k}\mathcal M_k \to \mathbb R
$$

を受け取る．

出力は

1. パラメータ側の covector `\tilde p_k`
2. pullback metric `G_k`
3. 更新方向 `\Delta\theta_k`
4. 下層へ渡す adjoint action `\Lambda_{k-1}`

の 4 つである．

各ユニットごとに

$$
\tilde p_{k,i}(\delta\theta)
:=
\Lambda_{k,i}\!\left(D_\theta f_k(\phi_{k-1};\theta_{k,i})[\delta\theta]\right)
$$

を計算し，さらに

$$
G^{(i)}(\delta\theta,\delta\eta)
:=
\left\langle
D_\theta f_k(\phi_{k-1};\theta_{k,i})[\delta\theta],
D_\theta f_k(\phi_{k-1};\theta_{k,i})[\delta\eta]
\right\rangle_{\Phi_k}
$$

を作る．そのあと

$$
G^{(i)} \Delta\theta_{k,i} = - \tilde p_{k,i}
$$

を解いて更新方向を得る．同時に，下層へ渡す随伴は

$$
\Lambda_{k-1}(u)
:=
\Lambda_k(D_{\phi_{k-1}} f_k[u])
$$

で計算する．

したがって 1 層の backward は，概念的には

$$
\Lambda_k
\longmapsto
(\tilde p_k,\; G_k,\; \Delta\theta_k,\; \Lambda_{k-1})
$$

という写像である．

### 座標化した実装

実際には `\Theta_k^{unit} \cong \mathbb R^{d_k}` と座標化し，標準基底 `e_a` を使えば十分である．

$$
v_a := D_\theta f_k(\phi_{k-1};\theta_{k,i})[e_a]
$$

とおくと，

$$
(\tilde p_{k,i})_a = \Lambda_{k,i}(v_a),
\qquad
G^{(i)}_{ab} = \langle v_a,v_b\rangle_{\Phi_k}
$$

となるので，局所的な linear solve

$$
G^{(i)} \Delta\theta_{k,i} = - \tilde p_{k,i}
$$

だけで更新方向が得られる．

### JAX 実装での closure

数学的には `\Lambda_k(h)=p_k[h]` という closure 表現は非常に自然である．
JAX 実装でも，**設計レベルでは closure を使ってよい**．

ただし，実際の `jax.jit` / `lax.scan` / `vmap` に載せる carry や戻り値としては，
Python の callable を直接運ぶより，配列や pytree による状態に落とした方が安全である．

したがって推奨方針は次の二層構造である．

- 概念設計:
  `\Lambda_k` を作用素あるいは closure として考える．
- 数値実装:
  各層で必要な基底ベクトル `v_a` 上の係数，Gram 行列 `G_k`，キャッシュ `(\phi_{k-1},\theta_k,\phi_k)` を pytree として保持する．

すなわち，

- 理論では `p_k` / `\Lambda_k` を主役にする
- 実装では `(\tilde p_k,G_k)` を各層で局所的に組み立てる
- 下層への伝播に必要な情報も配列の形で持つ

という設計にすると，関数空間の随伴と JAX 上の有限次元実装が矛盾なく両立する．

## 文献との対応

このメモに書いた内容のうち，どこまでが既知で，どこからが本ノート向けの整理かをまとめる．

### 文献で既知の部分

以下は既存文献やドキュメントで直接確認できる内容である．

1. reverse-mode AD / backprop を dual, cotangent, adjoint の pullback とみなす見方
   Baydin et al. (2018) および Frostig et al. (2021) に対応する．
   本メモでの
   $$
   p_{k-1} = (D_{\phi_{k-1}}f_k)^* p_k,
   \qquad
   \tilde p_k = (D_{\theta_k}f_k)^* p_k
   $$
   という読みはこの系統の整理である．

2. 導関数は自然には dual space に属し，更新に使うには Riesz 表現が必要になるという点
   dolfin-adjoint のドキュメントおよび Schwedes-Funke-Ham (2016) に対応する．
   本メモでの
   $$
   g_{k,i} = R_{\Phi_k}^{-1}p_{k,i}
   $$
   および pullback metric を通じて parameter gradient を得る流れは，この文脈の一般論に沿っている．

3. モデル空間の計量をパラメータ空間へ引き戻すと Gram 行列が現れるという点
   van Erp et al. (2022) の自然勾配の議論に対応する．
   本メモでの
   $$
   G^{(i)}_{ab}
   =
   \left\langle
   \partial_{\theta^a}f_k(\phi_{k-1};\theta_{k,i}),
   \partial_{\theta^b}f_k(\phi_{k-1};\theta_{k,i})
   \right\rangle_{\Phi_k}
   $$
   は，その本ノート向けの局所表示である．

4. 関数空間での adjoint からパラメータ勾配を得るという基本的な流れ
   Neural ODE や PDE constrained optimization の文脈で広く知られている．
   本メモでの
   $$
   D\widetilde F_k(\theta_{k,i})[\delta\theta]
   =
   -
   \left<
   p_{k,i},
   D_{\theta}f_k(\phi_{k-1};\theta_{k,i})[\delta\theta]
   \right>
   $$
   という形は，その離散層版として読める．

### 本メモでの整理・推論

以下は上の文献内容を，`bellman_bp.md` の記法と Bellman 的な局所問題に合わせて再構成したものである．

1. `\mathcal M_k(\phi_{k-1})` を各ユニットの可到達集合として置き，実際に必要なのは `T_{\phi_{k,i}}\mathcal M_k` だけだとする整理
   これは本ノートの問題設定に即した局所化であり，ここでの説明は本メモ側の整理である．

2. 関数空間での最急法を，まず `T_{\phi_{k,i}}\mathcal M_k` 上で定義し，それを `\theta` 空間へ引き戻して
   $$
   G^{(i)}\delta\theta_{k,i}^{sd} = -\tilde p_{k,i}
   $$
   と読む流れ
   これは既存の adjoint + Riesz + pullback metric の一般論からの自然な帰結だが，本メモでは Bellman 型の layer-wise reduced problem に合わせて記述している．

3. `p_k` を explicit な関数としてではなく，作用素
   $$
   \Lambda_k(h)=p_k[h]
   $$
   として持ち運ぶ逆伝播設計
   これは reverse-mode AD の見方と整合するが，本メモでの backward API
   $$
   \Lambda_k \mapsto (\tilde p_k, G_k, \Delta\theta_k, \Lambda_{k-1})
   $$
   という設計案そのものは，本プロジェクト向けの設計整理である．

4. JAX 実装では closure を概念設計として使い，実行時状態は pytree に落とすという二層構造
   これは理論をそのままコードへ移すための実装整理であり，本メモ側の提案である．

### 読み方

したがって，本メモは

- 随伴，Riesz 写像，pullback metric という既知の理論
- `bellman_bp.md` の記法に合わせた layer-wise / Bellman 的な reduced problem
- それを計算可能な backward 設計に落とすための実装整理

を接続する補助メモとして読むのがよい．

## 現時点での結論

- 関数空間側の随伴をまず導出する．
- その次に，関数空間での最急法を可到達接空間上で定義する．
- パラメータ更新は，その最急法を `D_\theta f_k` で引き戻し，pullback metric で `\theta` 側の最急法として解く．
- 下層への伝播も必要なので，`p_k` を単なる有限次元ベクトルに潰すより，作用素として持つ見方が有力である．
- JAX 実装では closure は概念設計として許容できるが，runtime state は配列 / pytree に落とす方が実装しやすい．

## 関連ファイル

- 主ノート: [`bellman_bp.md`](./bellman_bp.md)
- 文献要約: [`../../references/bellman-backprop-references.md`](../../references/bellman-backprop-references.md)
