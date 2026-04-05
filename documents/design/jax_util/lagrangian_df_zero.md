# `lagrangian_df_zero`

## 対象コードパス

- `python/jax_util/neuralnetwork/lagrangian_df_zero/`

## Goal

`lagrangian_df_zero` は、layer-wise parameter update を reduced problem の停留条件 $DF=0$ として扱う実験サブモジュールです。
既存の `neuralnetwork` 公開面を直接変更せず、別サブモジュール内で定式化、toy validation、solver 境界の確認を進めます。

## Non-Goals

- 既存 `python/jax_util/neuralnetwork/train.py` の直接置き換え
- `sequential_train.py` や `dynamic_programming.py` の unfinished API を即座に復活させること
- 大規模 benchmark や本番向け training loop の実装
- 関数空間の完全一般論を carry-over 初回で実装すること

## Mathematical Setup

`N` 層の network を

$$
\phi_k = f_k(\phi_{k-1}; \theta_k), \quad k = 1, \dots, N
$$

で表します。
終端では

$$
V_N(\phi_N) := J(\phi_N, y)
$$

と置き、Bellman 的には各段の価値関数

$$
V_k(\phi_k) = \min_{f_{k+1}} V_{k+1}(f_{k+1}(\phi_k))
$$

を考えます。

固定した $\phi_{k-1}$ に対して、current layer の出力 $\phi_k$ を変数とみなした reduced functional

$$
\widehat F_k(\phi_k)
$$

をまず定義し、その後で parameterization

$$
\Psi_k(\theta_k) := f_k(\phi_{k-1}; \theta_k)
$$

を通じた pullback

$$
\widetilde F_k(\theta_k) := \widehat F_k(\Psi_k(\theta_k))
$$

を考えます。
ここで layer `k` の update は、単純な

$$
\theta_k^{+} = \theta_k - \eta \nabla_{\theta_k} J
$$

ではなく、まず function-space 側の停留条件

$$
D\widehat F_k(\phi_k)[h] = 0
\qquad
\forall h \in T_{\phi_k}\mathcal M_k(\phi_{k-1})
$$

を書き、その後に誘導計量を用いて $\theta_k$ の変化へ引き戻す形で整理します。

## Bellman-Aligned Notation

この文書の数式は [bellman_bp_chat_summary.md](/workspace/notes/themes/bellman_bp_chat_summary.md) の記法に合わせます。

- $\phi_k$
  - layer `k` の state / output
- $\theta_k$
  - layer `k` の parameter
- $V_k$
  - layer `k` から先を最適化した value function
- $p_k$
  - function-space 側の adjoint / covector
- $\Lambda_k(h)=p_k[h]$
  - $p_k$ を作用として書いた linear functional
- $\widetilde\Lambda_k$
  - $\Lambda_k$ を parameter space へ引き戻した covector
- $g_k$
  - reachable tangent 上の function-space 計量
- $G_k$
  - $\Psi_k$ から誘導される parameter-space 計量

## Canonical Equality-Constraint Formulation

実装の主線は equality-constraint を持つ Lagrangian とします。
suffix state と multiplier を明示して、

$$
\mathcal{L}_k =
V_N(\phi_N) + \sum_{i=k}^{N} \langle p_i, \phi_i - f_i(\phi_{i-1}; \theta_i) \rangle
$$

を考えます。

この formulation を canonical とし、penalty 法や lifted relaxation は comparison baseline として別扱いにします。

Stage 0 の reference solve では、suffix 開始層を `k` とし、prefix state $\phi_{k-1}$ を固定したうえで

$$
u_k = (\theta_k,\dots,\theta_N,\; \phi_k,\dots,\phi_N,\; p_k,\dots,p_N)
$$

をまとめて解く設計にします。
ここで $\phi_i$ と $p_i$ は各層出力と同じ shape の `Matrix` とし、$\theta_i$ は layer ごとに flatten した `Vector` とします。

## From $D_f \widetilde F = 0$ To The Stage 0 Residual

### Which $\widetilde F$ we mean

過去の Bellman 系ノートでは、

$$
D_f \widetilde F_k = 0
$$

という形で、current layer の map $f_k$ 自体に対する停留条件を書いていました。
この worktree では、まず固定した prefix state $\phi_{k-1}$ に対して evaluation map

$$
E_{k,\phi_{k-1}}(f) := f(\phi_{k-1})
$$

を入れ、function-level reduced functional を

$$
\widetilde F_k^{\mathrm{func}}(f_k)
:=
\widehat F_k(E_{k,\phi_{k-1}}(f_k))
=
\widehat F_k(f_k(\phi_{k-1}))
$$

と読みます。

一方、この文書の主記法である

$$
\widetilde F_k(\theta_k) = \widehat F_k(\Psi_k(\theta_k))
$$

は parameter pullback です。
したがって、旧ノートの

$$
D_f \widetilde F_k = 0
$$

は、正確には

$$
D_f \widetilde F_k^{\mathrm{func}}(f_k) = 0
$$

を意味し、さらにそれを parameter family へ制限したものが

$$
D_{\theta_k}\widetilde F_k(\theta_k) = 0
$$

です。

### Function-level expansion

admissible variation を

$$
h_k \in T_{f_k}\mathcal F_k
$$

とします。
ここで $\mathcal F_k$ は fixed prefix $\phi_{k-1}$ のもとで許される layer map の空間です。
evaluation map の微分は

$$
DE_{k,\phi_{k-1}}(f_k)[h_k] = h_k(\phi_{k-1})
$$

なので、chain rule から

$$
\begin{aligned}
D_f \widetilde F_k^{\mathrm{func}}(f_k)[h_k]
&=
D\widehat F_k(\phi_k)[DE_{k,\phi_{k-1}}(f_k)[h_k]] \\
&=
D\widehat F_k(\phi_k)[h_k(\phi_{k-1})]
\end{aligned}
$$

を得ます。

suffix Lagrangian で後段の primal / adjoint stationarity を満たすと、後段変数の一次変分は envelope-type に消え、current layer に対してだけ

$$
D\widehat F_k(\phi_k)[\delta \phi_k] = -\Lambda_k(\delta \phi_k) = -\langle p_k, \delta \phi_k \rangle
$$

が残ります。
よって

$$
D_f \widetilde F_k^{\mathrm{func}}(f_k)[h_k]
=
-\Lambda_k(h_k(\phi_{k-1}))
=
-\langle p_k, h_k(\phi_{k-1}) \rangle
$$

です。

ここで重要なのは、この微分が「関数 $f_k$ 全体」ではなく、「固定された $\phi_{k-1}$ で評価した fiber」に沿って書かれていることです。
この worktree の reduced problem は prefix state を freeze するので、$D_f \widetilde F_k = 0$ は global operator identity ではなく

$$
\langle p_k, h_k(\phi_{k-1}) \rangle = 0
\qquad
\forall h_k \in T_{f_k}\mathcal F_k
$$

という fiberwise stationarity を意味します。

### Reachable tangent interpretation

current layer map の admissible variation が作る reachable variation space を

$$
\mathcal H_k(\phi_{k-1}; f_k)
:=
\{ h_k(\phi_{k-1}) \mid h_k \in T_{f_k}\mathcal F_k \}
\subset
T_{\phi_k}\mathcal M_k(\phi_{k-1})
$$

と置くと、

$$
D_f \widetilde F_k^{\mathrm{func}}(f_k) = 0
\iff
p_k \text{ が } \mathcal H_k(\phi_{k-1}; f_k) \text{ を annihilate する}
$$

と読めます。
これは「reduced differential が 0 である」という主張を、adjoint $p_k$ による可到達接空間の消去条件へ言い換えたものです。

### Parameter pullback expansion

parameterized layer family

$$
f_k(\cdot; \theta_k)
$$

に restricted すると、admissible variation は

$$
h_k = D_\theta f_k(\cdot; \theta_k)[\delta \theta_k]
$$

の形に限られます。
したがって

$$
\begin{aligned}
D_{\theta_k}\widetilde F_k(\theta_k)[\delta \theta_k]
&=
D_f \widetilde F_k^{\mathrm{func}}(f_k)
\bigl[
D_\theta f_k(\cdot; \theta_k)[\delta \theta_k]
\bigr] \\
&=
-\left\langle
p_k,
D_\theta f_k(\phi_{k-1}; \theta_k)[\delta \theta_k]
\right\rangle \\
&=
-\widetilde \Lambda_k(\delta \theta_k)
\end{aligned}
$$

です。
よって parameter pullback の停留条件は

$$
D_{\theta_k}\widetilde F_k(\theta_k) = 0
\iff
\widetilde \Lambda_k = 0
\iff
(D_{\theta_k} f_k(\phi_{k-1}; \theta_k))^* p_k = 0
$$

となります。

### Why Stage 0 uses $r^\theta_k$

Stage 0 実装が使う parameter residual

$$
r^\theta_k = (D_{\theta_k} f_k(\phi_{k-1}; \theta_k))^* p_k
$$

は、上の stationarity equation の座標表示です。
Lagrangian をどう符号規約で書くかによって $D_{\theta_k}\widetilde F_k$ には全体の符号が付きますが、零点集合は変わりません。
このため Stage 0 では、実装と debugging の都合を優先して

$$
r^\theta_k = 0
$$

を canonical な solver target としています。

### Which equation $\theta_k$ solves

ここまでで「covector が 0 になるべき」という形は出ましたが、unknown を $\theta_k$ だけにしたときに何を解くのかを明示しておきます。

まず exact reduced problem の意味では、$\theta_k^\star$ は

$$
D_{\theta_k}\widetilde F_k(\theta_k^\star)[\delta \theta_k] = 0
\qquad
\forall \delta \theta_k
$$

を満たす stationary point です。
これは covector 条件として

$$
(D_{\theta_k} f_k(\phi_{k-1}; \theta_k^\star))^* p_k^\star = 0
$$

と同値です。
ただし $p_k^\star$ は独立定数ではなく、$\theta_k^\star$ を通じて決まる suffix optimality の結果です。

その依存関係を明示すると、

$$
\phi_k(\theta_k) := f_k(\phi_{k-1}; \theta_k)
$$

として、後段の最適応答

$$
(\theta_{k+1:N}^\star(\theta_k), \phi_{k+1:N}^\star(\theta_k), p_{k:N}^\star(\theta_k))
$$

が suffix KKT 系を満たすなら、reduced stationarity は 1 つの nonlinear root equation

$$
R_k^{\mathrm{red}}(\theta_k)
:=
(D_{\theta_k} f_k(\phi_{k-1}; \theta_k))^*
p_k^\star(\theta_k)
= 0
$$

として書けます。
この意味で、exact な $\theta_k^\star$ は「pullback covector が消える nonlinear equation の解」です。

一方、current worktree の Stage 0 solver は $\theta_k$ だけを直接消去しているのではなく、

$$
u_k = (\theta_{k:N}, \phi_{k:N}, p_{k:N})
$$

全体に対して

$$
R_k(u_k)
:=
\begin{bmatrix}
r_k^\phi \\
r_k^p \\
r_k^\theta
\end{bmatrix}
= 0
$$

を解いています。
このとき $\theta_k^\star$ は full root $u_k^\star$ の先頭 block です。
Stage 0 が joint solve を採るのは、$p_k^\star(\theta_k)$ を明示的に消去して reduced equation 1 本へ潰す前に、primal / adjoint / parameter residual の整合を直接検証したいからです。

さらに、exact root そのものではなく「そこへ向かう 1 ステップ」を定めたい場合は、stationarity equation を直接解く代わりに Riesz 方程式

$$
G_k(\Delta \theta_k, \delta \eta_k)
=
-\widetilde \Lambda_k(\delta \eta_k)
\qquad
\forall \delta \eta_k
$$

を解きます。
座標化すれば

$$
G_k \Delta \theta_k = -\tilde p_k
$$

であり、

$$
\theta_k^{+} = \theta_k + \Delta \theta_k
$$

です。
こちらは stationary point の defining equation ではなく、stationarity へ向かう local update equation です。

### How to solve the $\theta_k$ equation

上の 3 種類の式は、数値的には別物です。
「勾配法でやるのか」「CG で終わるのか」「解析解があるのか」は、どの式を採るかで変わります。

#### 1. Exact reduced stationarity: generally a nonlinear root

exact reduced stationarity

$$
R_k^{\mathrm{red}}(\theta_k) = 0
$$

は、一般には nonlinear root です。
理由は、$(D_{\theta_k} f_k)^*$ だけでなく $p_k^\star(\theta_k)$ も $\theta_k$ に依存するからです。
したがって、これをそのまま解きたいなら第一候補は

$$
J_{R_k^{\mathrm{red}}}(\theta_k)\Delta \theta_k
=
-R_k^{\mathrm{red}}(\theta_k)
$$

という Newton 型、あるいは Gauss-Newton / Levenberg-Marquardt 型です。

この文脈での plain gradient descent は、root equation を直接解いているわけではありません。
勾配法を使うなら、例えば

$$
\min_{\theta_k} \widetilde F_k(\theta_k)
\qquad\text{or}\qquad
\min_{\theta_k} \frac12 \|R_k^{\mathrm{red}}(\theta_k)\|^2
$$

という minimization problem へ言い換えてから使うことになります。
ただし後者は stationary root を least-squares residual minimization に変えているので、解く問題自体が少し変わります。

#### 2. CG can finish the induced-metric step, but not the whole nonlinear problem

一方で

$$
G_k \Delta \theta_k = -\tilde p_k
$$

は linear system です。
$G_k$ が対称正定値なら、ここは CG で解いてよいですし、むしろそれが自然です。
この意味では「CG で終わる」という言い方が成立するのは、exact stationary root そのものではなく Bellman 的な local update equation を採る場合です。

ただし注意点が 2 つあります。

1. $G_k$ は induced metric なので、parameterization に冗長性があると半正定値になりうる
1. exact reduced root $R_k^{\mathrm{red}}(\theta_k)=0$ 自体は nonlinear なので、CG 単体では終わらない

前者では damping を入れて

$$
(G_k + \lambda I)\Delta \theta_k = -\tilde p_k
$$

とするか、pseudoinverse / MINRES / LSQR 系へ逃がす必要があります。
後者では CG を使うとしても、Newton-Krylov の内側で linearized step を解くために使う、という位置づけになります。

#### 3. Closed form exists only in special linear-quadratic regimes

解析解があるのは、toy な special case に限られます。
典型的には

- $f_k(\phi_{k-1}; \theta_k)$ が $\theta_k$ に線形
- $\widehat F_k(\phi_k)$ あるいは suffix value が $\phi_k$ に二次
- したがって $\widetilde F_k(\theta_k)$ が $\theta_k$ に二次

という linear-quadratic regime です。
このとき stationary condition は

$$
H_k \theta_k + b_k = 0
$$

という linear equation に落ち、

$$
\theta_k^\star = -H_k^{-1} b_k
$$

あるいは特異なときは Moore-Penrose pseudoinverse で

$$
\theta_k^\star = -H_k^\dagger b_k
$$

と書けます。

同様に local update equation でも、small dense case なら

$$
\Delta \theta_k = -G_k^{-1}\tilde p_k
$$

がそのまま closed form です。
ただし nonlinearity が入った一般の `standard` network では、$p_k^\star(\theta_k)$ の依存まで含めて closed form を期待するのは難しいです。

#### 4. Recommended interpretation for this worktree

この worktree では次の使い分けを canonical とします。

1. Stage 0
   `u_k = (\theta_{k:N}, \phi_{k:N}, p_{k:N})` に対する full KKT root を dense Newton で解く
1. Stage 1
   Bellman 的な layer-local update として $G_k \Delta \theta_k = -\tilde p_k$ を解く
1. solver choice
   small dense toy なら direct solve、large case なら CG あるいは preconditioned CG を第一候補にする
1. exact reduced root
   必要なら Newton-Krylov / Gauss-Newton を検討するが、plain gradient descent は主線にしない
1. analytic solution
   2-layer linear toy や linear-quadratic verification case に限定して使う

## Detailed Residual Design

### Primal residual

各 suffix layer $i = k, \dots, N$ について

$$
r^{\phi}_i = \phi_i - f_i(\phi_{i-1}; \theta_i)
$$

を置きます。

### Adjoint residual

terminal layer では

$$
r^{p}_N = D_{\phi_N} V_N(\phi_N) + p_N
$$

中間 layer では

$$
r^{p}_i = p_i - (D_{\phi_i} f_{i+1}(\phi_i; \theta_{i+1}))^* p_{i+1}
$$

を使います。

### Function-Space Reduced Differential

reachable tangent $h_i \in T_{\phi_i}\mathcal M_i(\phi_{i-1})$ に対して

$$
D\widehat F_i(\phi_i)[h_i] = -\Lambda_i(h_i) = -\langle p_i, h_i \rangle
$$

です。
停留条件は、$p_i$ が reachable tangent を annihilate することとして読めます。

### Pullback Covector

parameterization $\Psi_i(\theta_i)=f_i(\phi_{i-1};\theta_i)$ に沿う変分 $\delta\theta_i$ に対して

$$
\widetilde\Lambda_i(\delta\theta_i)
:=
\Lambda_i(D\Psi_i(\theta_i)[\delta\theta_i])
=
\langle p_i, D_\theta f_i(\phi_{i-1}; \theta_i)[\delta\theta_i]\rangle
$$

を得ます。

### Induced Metric And Steepest Direction

function-space 側の計量 $g_i$ から parameter space 側の induced metric

$$
G_i(\delta\theta_i, \delta\eta_i)
:=
g_i\!\left(
D\Psi_i(\theta_i)[\delta\theta_i],
D\Psi_i(\theta_i)[\delta\eta_i]
\right)
$$

を定めます。
Bellman 的な最急方向は

$$
G_i(\Delta\theta_i, \delta\eta_i) = -\widetilde\Lambda_i(\delta\eta_i)
$$

で定まり、座標化すれば

$$
G_i \Delta\theta_i = -\tilde p_i
$$

です。

### Residual Used In Stage 0 Code

Stage 0 実装では、$\widetilde\Lambda_i$ の座標表示として

$$
r^{\theta}_i = (D_{\theta_i} f_i(\phi_{i-1}; \theta_i))^* p_i
$$

を residual block に使います。
Lagrangian の微分では符号が付きますが、零点集合は同じなので Stage 0 ではこの形に固定します。

### Packed residual ordering

solver に渡す residual は次の順で pack します。

1. $vec(r^\phi_k), \dots, vec(r^\phi_N)$
1. $vec(r^p_k), \dots, vec(r^p_N)$
1. $r^\theta_k, \dots, r^\theta_N$

この順序は debug log、finite-difference 検証、Jacobian block の読みやすさを優先したものです。

## Solver Strategy By Stage

### Stage 0: dense toy reference solve

最初の実装では、小さい問題だけを対象にします。
packed residual に対して `jax.jacobian` で dense Jacobian を作り、`jnp.linalg.solve` か least-squares で Newton step を解きます。

この段階では scalability よりも

- residual 実装の正しさ
- block ordering の一貫性
- autodiff gradient との対応

を優先します。

### Stage 1: induced-metric layer update

Bellman 的な 1-layer 更新則では、まず $\Lambda_k$ と $G_k$ を構成し、`layer k` のみについて

$$
G_k \Delta\theta_k = -\tilde p_k
$$

を解く方針に進みます。
これが current design の本命です。
Stage 0 の suffix reference solve は、この layer-local 版を検証するための比較基準と位置付けます。

### Stage 2: operator-form linearization

dense Jacobian が正しいと確認できたら、`base.linearize` を使って operator-form の Jacobian matvec を作ります。
ここで packed residual $R(u)$ に対して

$$
J_R(u)\Delta u = -R(u)
$$

を解く Newton-Krylov 形へ移ります。

### Stage 3: KKT / Schur reuse

suffix problem が大きくなったら、`solvers/kkt_solver.py` のブロック構造を再利用します。
ただし carry-over 初回ではここまで実装しません。
設計上の依存候補としてだけ固定します。

## Research Decisions Fixed In This Worktree

### 1. Exact equality-constraint line を主線にする

`Carreira-Perpinan_Wang_2014_*` を structural precedent とします。
`Askari_*` と `Gu_*` は有力な comparison baseline ですが、canonical objective にはしません。

### 2. 数式の正本は function-space first で書く

関数空間での Riesz / Gram 計量は理論上重要です。
このため数式の正本は、まず $D\widehat F_k$ を $\phi_k$ 側で書き、その後に pullback metric を通して $\theta_k$ の更新へ落とす流れに固定します。

### 3. Inner solve を持つ場合は implicit differentiation を前提にする

`OptNet` と `DEQ` を参照点にし、solve 後の outer differentiation を ad hoc にせず、KKT / root residual を通した微分境界を切ります。

## From Math To Code

### Mapping To Current Code Names

数式と実装名の対応は次です。

- $\phi_*$ ↔ code の `z_*`
- fixed prefix state $\phi_{k-1}$ ↔ `PrefixTape.z_prefix`
- suffix states $\phi_{k:N}$ ↔ `SuffixVariables.z_tail`
- suffix multipliers $p_{k:N}$ ↔ `SuffixVariables.p_tail`
- terminal value $V_N$ ↔ `loss_from_output`
- pullback covector の座標表示 ↔ `theta_residual`

数学節では code 名を使いません。
以後の節では、実装上必要な場合にだけ code 名を使います。

### Existing Constraints From Current Code

詳細設計は、既存 `neuralnetwork` の次の境界を前提にします。

- `protocols.py` は `Params`、`Static`、`Carry`、`Ctx`、`PyTreeOptimizationProblem`、`ConstrainedPyTreeOptimizationProblem` の契約面を提供します。
- `layer_utils.py` には `module_to_vector` / `vector_to_module` があり、layer 単位の flatten / rebuild を再利用できます。
- `neuralnetwork.py` は `state_initializer` と `forward_with_cache` を持ち、prefix forward state を取る入口として使えます。
- `base.nonlinearoperator` は `linearize` と `adjoint` を持ち、JVP / VJP を `LinearOperator` として扱えます。
- `kkt_solver.py` と `pdipm.py` は、将来の operator-form Newton / KKT solve の再利用候補です。

この worktree では、これらを上書きせず、`lagrangian_df_zero` 側で薄い adapter を追加する方針にします。

### Detailed Package Layout

### `types.py`

- `LayerVectorization`
  - 1 layer 分の `RebuildState`、parameter 次元、出力 shape を保持する static metadata
- `PrefixTape`
  - `layer_index`、`ctx`、`z_prefix` を保持する
- `SuffixVariables`
  - `theta_tail: tuple[Vector, ...]`
  - `z_tail: tuple[Matrix, ...]`
  - `p_tail: tuple[Matrix, ...]`
- `BlockSlice`
  - pack / unpack 用の slice と shape metadata
- `VariableLayout`
  - `theta_slices`、`z_slices`、`p_slices` をまとめる
- `StationaryProblem`
  - suffix 問題を評価する closure 群と metadata を束ねる
- `StationarySolveState`
  - 現在の packed iterate、residual norm、反復回数を保持する
- `StationarySolveInfo`
  - `converged`、`iterations`、`residual_norm`、`residual_breakdown` を保持する

### `parameterization.py`

- `build_layer_vectorizations(layers)`
  - 各 layer を vector 化し、`LayerVectorization` を列挙する
- `flatten_layer_params(layer)`
  - 既存 `module_to_vector` の薄い wrapper
- `rebuild_layer(vector, layer_vectorization)`
  - 既存 `vector_to_module` の薄い wrapper
- `pack_variables(variables, layout)`
  - `SuffixVariables` を 1 本の `Vector` へ詰める
- `unpack_variables(vector, layout)`
  - packed vector を `SuffixVariables` へ戻す

### `prefix.py`

- `build_prefix_tape(network, x, layer_index)`
  - `forward_with_cache` か同等の rollout を使い、`z_prefix` と `ctx` を取得する
- `suffix_input(prefix_tape)`
  - suffix 最初の layer に渡す入力 state を返す

### `constraints.py`

- `rollout_suffix(prefix_tape, theta_tail, vectorizations)`
  - suffix layer 群を順に rebuild して `z_tail` を計算する
- `primal_residual(variables, prefix_tape, vectorizations)`
  - primal residual を返す
- `adjoint_residual(variables, prefix_tape, vectorizations, loss_from_output)`
  - adjoint residual を返す
- `theta_residual(variables, prefix_tape, vectorizations)`
  - pullback covector の座標表示を返す

### `lagrangian.py`

- `local_lagrangian(variables, problem)`
  - suffix Lagrangian を評価する
- `kkt_residual_terms(variables, problem)`
  - split residual blocks を返す
- `kkt_residual(variables, problem)`
  - packed residual を返す
- `packed_kkt_residual(vector, problem)`
  - packed iterate から packed residual を返す

### `stationary_solve.py`

- `build_stationary_problem(network, x, y, layer_index, loss_from_output)`
  - suffix stationary problem を組み立てる
- `initialize_suffix_variables(problem)`
  - current network 由来の warm start を返す
- `solve_stationary_suffix(problem, init_variables, *, tol, maxiter, method)`
  - Stage 0 の dense Newton reference solve を回す
- `extract_layer_update(solution, layer_offset=0)`
  - 解いた suffix から layer `k` の updated parameter を取り出す

### `toy.py`

- MSE output loss を定義する
- identity activation の `standard` network builder を定義する
- solver / residual の toy validation 入口を置く

## 公開 API

- 既存 trainer には未接続です。
- ただし `python/jax_util/neuralnetwork/lagrangian_df_zero/__init__.py` から、次の experimental internal API を公開します。
  - `build_stationary_problem`
  - `initialize_suffix_variables`
  - `kkt_residual`
  - `solve_stationary_suffix`
  - `extract_layer_update`
  - `mean_squared_output_loss`
  - residual / pack-unpack helper 群

## Detailed API Draft

Stage 0 で実装した internal API は次です。

<pre><code class="language-python">
def build_stationary_problem(
    network: NeuralNetwork,
    x: Matrix,
    y: Matrix,
    layer_index: int,
    loss_from_output: Callable[[Matrix, Matrix], Scalar],
) -> StationaryProblem: ...

def initialize_suffix_variables(
    problem: StationaryProblem,
) -> SuffixVariables: ...

def kkt_residual(
    variables: SuffixVariables,
    problem: StationaryProblem,
) -> Vector: ...

def solve_stationary_suffix(
    problem: StationaryProblem,
    init_variables: SuffixVariables,
    *,
    tol: Scalar,
    maxiter: int,
    method: str = "dense_newton",
) -> tuple[SuffixVariables, StationarySolveInfo]: ...

def extract_layer_update(
    solution: SuffixVariables,
    layer_offset: int = 0,
) -> Vector: ...
</code></pre>

`loss_from_output` は output `Matrix` と target `Matrix` を受ける薄い関数とし、prototype 段階では MSE に固定してもよいですが、設計では callable を残します。

## Integration Boundary

### Existing `train.py`

`train.py` は最小の end-to-end gradient loop として残します。
`lagrangian_df_zero` はここへ直接パッチしません。

### Existing `sequential_train.py`

将来的な integration point は `SingleLayerBackprop` と `IncrementalTrainer` です。
ただし初回 carry-over では `lagrangian_df_zero` からこの protocol をまだ実装しません。
まずは standalone な reference solver と toy validator を作ります。

### Existing `neuralnetwork.py`

初回は `network_type == "standard"` のみを対象にします。
`icnn` を最初から含めると、nonnegativity と入力 skip の扱いが混ざって residual 検証が難しくなります。

## Implementation Responsibilities

Stage 0 実装では、次の責務を分離しました。

- `parameterization`
  - layer parameter の flatten / unflatten
  - update 方向を pytree へ戻す変換
- `constraints`
  - $\phi_k = f_k(\phi_{k-1}; \theta_k)$ の residual
  - 必要な JVP / VJP の薄いラッパ
- `lagrangian`
  - local Lagrangian と stationarity residual の組み立て
- `stationary_solve`
  - $DF=0$ または等価な KKT residual を解く入口
- `toy`
  - MSE loss と identity network builder
  - toy validation の入口

## Current Implementation Status

この worktree では、Stage 0 の toy 実装まで完了しています。

- `types.py`
  - `LayerVectorization`、`PrefixTape`、`VariableLayout`、`StationaryProblem` などの内部データモデルを追加済み
- `parameterization.py`
  - layer flatten / rebuild、packed variable の layout、pack / unpack を追加済み
- `prefix.py`
  - `standard` network 限定で prefix state を固定する入口を追加済み
- `constraints.py`
  - suffix rollout、primal / adjoint / theta residual、warm-start multiplier を追加済み
- `lagrangian.py`
  - local Lagrangian、split residual、packed KKT residual、residual breakdown を追加済み
- `stationary_solve.py`
  - problem builder、warm start、dense Newton toy solver、layer update extraction を追加済み
- `toy.py`
  - MSE output loss と identity-activation network builder を追加済み
- `python/tests/neuralnetwork/`
  - parameterization / constraints / toy solver 向けの targeted tests を追加済み

## Validation Status

### Completed in this worktree

- `pack_variables` / `unpack_variables` の roundtrip test を追加した
- forward rollout 由来の warm start で primal residual が 0 になることを確認した
- backward multiplier warm start で adjoint residual が 0 になることを確認した
- zero network + zero target の trivial case で KKT residual が 0 になることを確認した
- dense Newton toy solver が KKT residual norm を減らすことを確認した
- `pyright`、`ruff check`、targeted `pytest` を通した

### Remaining before integration

- 2-layer linear reduced problem の hand-derived equation との照合
- finite-difference による residual / Jacobian の追加検証
- backprop、MAC / penalty、lifted 系との baseline comparison
- operator-form linearization と `kkt_solver.py` 再利用の検証
- `standard` 以外の network type への拡張

### Current solver boundary

- inner solve の warm-start
- damping / line search
- ill-conditioning 時の fallback

## Test File Plan

Stage 0 で追加した targeted tests は次です。

- `python/tests/neuralnetwork/test_lagrangian_df_zero_parameterization.py`
  - layer vectorization
  - pack / unpack
  - rebuild roundtrip
- `python/tests/neuralnetwork/test_lagrangian_df_zero_constraints.py`
  - primal residual
  - warm-start adjoint residual
- `python/tests/neuralnetwork/test_lagrangian_df_zero_toy.py`
  - trivial zero-residual case
  - dense Newton residual decrease

## Dependency Rules

- `lagrangian_df_zero` から既存 `train.py` を import しません。
- `lagrangian_df_zero` から `optimizers/pdipm.py` を直結で呼びません。
- 初回 carry-over で solver 実装は dense toy に限定し、`kkt_solver.py` は reference dependency としてのみ扱います。
- flatten / rebuild は `layer_utils.py` の既存関数を流用し、同じ責務の重複実装を禁止します。

## Risks

- $\theta_{k+1:N}^{*}(\theta_k)$ が一意でない場合、reduced objective 自体が不安定になる
- Jacobian を陽に作ると memory が急増する
- Euclidean metric と pullback metric で update の意味が変わる
- lifted relaxation と exact formulation を混ぜると比較が曖昧になる
- `Carry` / `Ctx` が protocol のままだと generic 実装が広すぎるため、v1 では `StandardCarry` / `StandardCtx` を既定に絞る必要がある
- batch 軸を含む `Matrix` residual を layer parameter residual と同列に pack するため、slice layout が壊れると debug が難しい

## Exit Criteria Before Integration

既存 trainer や benchmark へ持ち込む前に、次を満たさなければなりません。

1. toy problem 上で exact reduced stationary equation を紙とコードの両方で確認する
1. baseline を `backprop`、`MAC / penalty`、`lifted`、$DF=0$ の 4 系統で固定する
1. residual decrease と loss decrease の関係を別々に観測する
1. `notes/themes/06_lagrangian_df_zero.md` と `references/lagrangian_df_zero/README.md` の両方に根拠を残す
1. package layout、data model、residual ordering、dense toy solver の 4 点が引き続き整合することを確認する

## References

- [06_lagrangian_df_zero.md](/workspace/notes/themes/06_lagrangian_df_zero.md)
- [Reference pack](/workspace/references/lagrangian_df_zero/README.md)
- [Bellman / adjoint note](/workspace/notes/themes/bellman_bp.md)
