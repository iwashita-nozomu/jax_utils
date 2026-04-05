# 06. ラグランジアン `DF=0` に基づく層別更新の文献サーベイ

## 要旨

本ノートは、標準的な feedforward network の layer update を、通常の backpropagation に基づく勾配降下ではなく、reduced problem の停留条件 `DF=0` として与えられるか、という問いに対する文献サーベイである。結論を先に述べると、現在の文献群はこの問いをそのまま解いた survey paper を提供していない。一方で、補助変数による等式制約化、lifted / relaxed training、inner solve を含む最適化層の微分、stationary learning という近接した研究線はすでに存在し、そこから本 worktree が取るべき設計上の判断をかなり明確に引き出せる。

最も近い構造的先行研究は Carreira-Perpinan and Wang (2014) の auxiliary coordinates であり、nested objective を明示的な状態変数つきの制約付き問題へ持ち上げるという点で、suffix state と multiplier を導入する本 worktree の方針に最も近い。これに対し、Askari et al.、Gu et al.、Wang and Benning の lifted / relaxed line は強力な比較対象ではあるが、元の問題をそのまま解く exact reduced problem とは区別して扱う必要がある。また、inner solve を残すなら Amos and Kolter (OptNet) および Bai et al. (DEQ) が示す implicit differentiation の境界を無視できない。さらに、Lau et al. の proximal BCD 系は solver choice の比較対象として重要だが、exact reduced stationary equation そのものを定める文献ではない。ここでいう「最も近い」や「最も重要」は、あくまで本 worktree の数式的問いに照らした相対評価であって、deep learning 全体の related work を網羅的に順位付けするものではない。

以上を踏まえると、本 worktree に対する最も自然な整理は次の通りである。数式の正本は exact equality-constraint plus Lagrangian とし、lifted / penalty / Bregman / proximal BCD 系は baseline として扱う。exact reduced equation は一般に非線形 root とみなすべきであり、Bellman 的な local update equation と混同しない。local update としての induced-metric system は CG を含む線形ソルバに適し、exact root は Newton 型または Newton-Krylov 型の対象である。解析解は linear-quadratic toy に限って期待すべきである。

## 1. 問いと射程

### 1.1 問い

標準的な feedforward network の layer update を、単純な backpropagation の gradient step ではなく、上位層の最適性を織り込んだ reduced problem の停留条件 `DF=0` として与えられるか。

### 1.2 調査の射程

- 対象は `python/jax_util/neuralnetwork/lagrangian_df_zero/` の前提になる文献である。
- 主題は、exact equality-constraint formulation、lifted / penalty 系の比較対象、implicit differentiation 境界、stationary learning の対照線を整理することである。
- 読みの中心は、`standard` な feedforward network を layer-wise / block-wise に扱う一次資料である。
- 実装詳細、数式展開、solver interface の正本は [lagrangian_df_zero.md](/workspace/documents/design/jax_util/lagrangian_df_zero.md) に置く。本ノートは読者向けのサーベイ正本とする。

### 1.3 このノートの立場

このテーマをそのまま扱う survey paper は、現在の reference pack には見当たらない。したがって本ノートは、近接する一次資料を比較し、本 worktree の問いに引き寄せて再構成した合成的サーベイである。文献から直接言えることと、本 worktree の解釈や設計判断は意識的に分けて書く。

## 2. 調査方法

### 2.1 調査日とソース

- `Survey Date:`
  - 2026-04-04 から 2026-04-05
- `Primary pool:`
  - [references/lagrangian_df_zero/README.md](/workspace/references/lagrangian_df_zero/README.md)
  - [references/lagrangian_df_zero/catalog.md](/workspace/references/lagrangian_df_zero/catalog.md)
- `Metadata / abstract confirmation:`
  - official paper pages and official landing pages when local extraction was inconvenient

### 2.2 クエリ方針

今回の調査では、次の query pack を中心に検索語を束ねた。

- `layer-wise training reduced objective deep network`
- `auxiliary coordinates deeply nested systems`
- `lifted neural networks activation argmin penalty`
- `Fenchel lifted networks Lagrange relaxation`
- `optimization as a layer implicit differentiation`
- `deep equilibrium models root finding implicit differentiation`
- `equilibrium propagation stationary learning`
- `proximal block coordinate descent deep neural network training`

### 2.3 採用基準と除外基準

採用基準は次の通りである。

1. layer-wise、block-wise、lifted のいずれかで deep network training を扱っていること
1. exact equality constraints、auxiliary states、あるいは明示的な multiplier structure を含むこと
1. inner solve / root solve の differentiation 境界に関係すること
1. 本 worktree の `DF=0` 仮説を支持するか、少なくともその主張を狭める材料を与えること

除外基準は次の通りである。

1. deep-network lifting や layer-wise structure を持たない一般的 optimizer 論文
1. formulation の比較に効かない hardware-only report
1. constrained / lifted / implicit solve の観点を持たない architecture paper

### 2.4 選定した主な文献群

- 歴史的背景
  - Hinton et al. (2006)
  - Bengio et al. (2007)
- 構造的先行研究
  - Carreira-Perpinan and Wang (2014)
- lifted / relaxation baseline
  - Askari et al.
  - Gu et al.
  - Wang and Benning
- solver / differentiation boundary
  - Amos and Kolter
  - Bai et al.
- stationary learning の contrast line
  - Scellier and Bengio
- solver choice を狭める optimizer baseline
  - Lau et al.

### 2.5 調査方法の限界

本調査は systematic review ではなく、設計志向の focused survey である。対象は `lagrangian_df_zero` の数式設計に直接効く文献へ意図的に絞っており、deep learning optimization 全体の完全な網羅を目指していない。したがって、「この分野で最も重要な文献」の列挙ではなく、「本 worktree の問いに最も近い文献」の再構成として読むべきである。

## 3. 文献地図

本調査で整理した研究線を、まず高いレベルでまとめる。

| 研究線                                | 代表文献                        | 本 worktree にとっての意味                                                 | 直接には与えないもの                                 |
| ------------------------------------- | ------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------- |
| 層別学習の歴史的背景                  | Hinton 2006, Bengio 2007        | layer-wise structure が最適化を助けうるという歴史的文脈                    | exact reduced problem の定式化                       |
| 補助変数による等式制約化              | Carreira-Perpinan and Wang 2014 | suffix state と multiplier を持つ constrained formulation の直接的先行研究 | Bellman 的な reduced differential と pullback metric |
| lifted / relaxed training             | Askari, Gu, Wang-Benning        | 比較対象として最重要な基準線                                               | original objective の exact reduced stationarity     |
| proximal / block-coordinate optimizer | Lau et al.                      | gradient-free / block-wise optimization の比較対象                         | `DF=0` の exact equation そのもの                    |
| optimization layer / equilibrium root | Amos-Kolter, Bai et al.         | inner solve を残したときの differentiation 境界                            | finite suffix reduced problem の直接解               |
| stationary learning の contrast line  | Scellier-Bengio                 | 「停留条件で学習する」近傍概念の対照例                                     | feedforward layer-local Bellman reduced stationarity |

## 4. 主要研究線の整理

### 4.1 歴史的背景としての層別学習

Hinton et al. (2006) と Bengio et al. (2007) は、deep network の optimization が単純ではなく、layer-wise な構造を使うことに実利があるという歴史的背景を与える。特に Bengio et al. は greedy layer-wise training が optimization にどう寄与するかを経験的に整理しており、「深いモデルに対して層別の考え方が有効だった」という事実を押さえるうえで重要である。

しかし、これらの文献を今回の `DF=0` 仮説の直接的支持文献として読むのは適切でない。理由は二つある。第一に、これらの主線は unsupervised pretraining や surrogate local objective にあり、上位層の最適性を value-function 的に織り込んだ reduced problem そのものを扱っていない。第二に、function-space の停留条件や、その parameter pullback を正面から論じていない。したがって、これらは「layer-wise という発想の歴史的背景」を与える文献であって、「exact reduced stationarity の理論的根拠」を与える文献ではない。

### 4.2 最も近い先行研究としての auxiliary coordinates

Carreira-Perpinan and Wang (2014) は、本調査で最も重要な構造的先行研究である。この論文の本質は、deeply nested objective を auxiliary coordinates を用いた constrained problem へ持ち上げる点にある。nested function をそのまま微分し続けるのではなく、中間状態を明示変数として外へ出し、制約付き最適化として扱うという戦略は、suffix state と multiplier を導入する本 worktree の構図と極めて近い。

もっとも、この近さは「構造の近さ」であって「数式の細部の一致」ではない。本 worktree が重視しているのは、Bellman 的な reduced functional、reachable tangent 上の differential、さらに pullback metric による parameter-space 側の更新である。Carreira-Perpinan and Wang は constrained reformulation と alternating optimization の道筋を与えるが、`D\widehat F_k` や `\widetilde \Lambda_k` を現在の記法で直接与えてくれるわけではない。したがって、この文献は canonical line の出発点ではあるが、そのまま本 worktree の数式正本にはならない。

### 4.3 Lifted / relaxed line は baseline として読むべきである

Askari et al.、Gu et al.、Wang and Benning は、いずれも通常の backpropagation とは異なる training line を提示しているが、その意味合いは本 worktree の exact line と異なる。Askari et al. は activation を convex subproblem の argmin として記述し、block-coordinate descent に向いた lifted formulation を与える。Gu et al. は multiplier を含む Fenchel lifted line を与え、relaxation / lower bound として定式化する。Wang and Benning は Bregman distance を通じて proximal-map activation を扱い、nonsmooth first-order solver を組み込んだ別系統の lifted formulation を与える。

これらの文献が重要なのは、どれも「layer-wise or block-wise な別解法」が現実に有力であることを示しているからである。しかし同時に、これらを exact reduced problem と同一視してはいけない。activation の持ち上げ方、制約の扱い方、lower-bound 化、Bregman 化といった違いは、最適化問題そのものの変更を含む。したがって、本 worktree においては、これらは最良の比較対象ではあるが、canonical formulation の直接的正当化には使えない。

### 4.4 Proximal BCD line は solver choice を狭めるが、問題設定そのものは与えない

Lau et al. の proximal block coordinate descent 系文献は、deep neural network training に対して gradient-free あるいは block-wise な optimization route が成立しうることを示す。これは、`DF=0` の exact line を考える際に、「gradient descent 以外の主線が必要か」という問いを狭めるうえで非常に有用である。

ただし、この文献が与えるのは optimizer design と収束の議論であって、本 worktree が扱いたい exact reduced stationary equation の定義そのものではない。言い換えると、Lau et al. は「どのような solver family が候補になるか」を教えてくれるが、「何を root equation として解くべきか」を定めてはくれない。この区別は、後で Newton 系と CG 系の役割を分けるうえでも重要になる。

### 4.5 Inner solve を持つなら implicit differentiation 境界は避けられない

Amos and Kolter (OptNet) と Bai et al. (DEQ) は、本 worktree が inner solve を持ち続けるなら必ず参照すべき文献である。OptNet は KKT solve を layer とみなし、その layer を exact differentiation 可能なモジュールとして扱う。DEQ は fixed point / root-finding を forward computation とし、その equilibrium point を implicit differentiation で backward へ接続する。

本 worktree の `lagrangian_df_zero` は、少なくとも Stage 0 において suffix stationary problem を inner solve として持つ。したがって、その solve を「forward で何とか回せばよいもの」として扱うのでは不十分であり、将来的に end-to-end な training loop と接続するなら、その solve をどう微分境界に置くかを設計しなければならない。OptNet と DEQ は model class こそ異なるが、この境界条件を厳密に意識すべきことを強く示している。

### 4.6 Equilibrium propagation は近いが、同じ問題ではない

Scellier and Bengio (2017) の equilibrium propagation は、「停留状態を通じて学習する」という意味で本 worktree の問題意識に近い。しかし、そこから直ちに本 worktree の理論的支持を引き出すのは危険である。扱うモデルは energy-based であり、二相の局所摂動と平衡状態の差から学習信号を取り出すという構図は、有限層 feedforward network に対する Bellman-style reduced stationarity とは異なる。

この文献の役割は、むしろ contrast line として「stationary learning に似た概念が既に存在するが、今回の問題はそれとは別である」と示すことにある。stationary learning という語感だけで equivalence を匂わせないためにも、contrast line として明示的に保持するのがよい。

## 5. このプロジェクトの記法による式展開

ここまでの文献比較だけでは、本 worktree が実際に何を「`DF=0`」と呼んでいるかは十分に明確ではない。そこで本節では、[lagrangian_df_zero.md](/workspace/documents/design/jax_util/lagrangian_df_zero.md) の数式設計を踏まえつつ、Bellman 記法に沿って `D_f \widetilde F = 0` から `\theta_k` の方程式までを一続きに書き下す。

### 5.1 Reduced functional と parameter pullback

`N` 層の network を

$$
\phi_i = f_i(\phi_{i-1}; \theta_i),
\qquad
i = 1, \dots, N
$$

で表し、終端の損失を

$$
V_N(\phi_N) = J(\phi_N, y)
$$

と書く。`k` 層に注目するとき、本 project では固定した prefix state `\phi_{k-1}` のもとで、current layer の出力 `\phi_k` を変数とする reduced functional

$$
\widehat F_k(\phi_k)
$$

を先に考える。そのうえで current layer の parameterization

$$
\Psi_k(\theta_k) := f_k(\phi_{k-1}; \theta_k)
$$

を通して parameter space へ引き戻した

$$
\widetilde F_k(\theta_k) := \widehat F_k(\Psi_k(\theta_k))
$$

を layer `k` の reduced objective とみなす。ここで重要なのは、`\widetilde F_k` が終端損失 `J` の単純な局所近似ではなく、「後段を最適化したあとに current layer の admissible 方向だけを見る」ための pullback だという点である。

### 5.2 `D_f \widetilde F = 0` の意味

過去の Bellman 系ノートでは、

$$
D_f \widetilde F_k = 0
$$

という記号が使われていた。この worktree では、これを global な関数恒等式ではなく、固定した `\phi_{k-1}` 上の fiberwise stationarity として読む。具体的には evaluation map

$$
E_{k,\phi_{k-1}}(f) := f(\phi_{k-1})
$$

を導入し、function-level reduced functional を

$$
\widetilde F_k^{\mathrm{func}}(f_k)
:=
\widehat F_k(E_{k,\phi_{k-1}}(f_k))
=
\widehat F_k(f_k(\phi_{k-1}))
$$

と置く。すると admissible variation `h_k \in T_{f_k}\mathcal F_k` に対して

$$
DE_{k,\phi_{k-1}}(f_k)[h_k] = h_k(\phi_{k-1})
$$

であるから、chain rule により

$$
\begin{aligned}
D_f \widetilde F_k^{\mathrm{func}}(f_k)[h_k]
&=
D\widehat F_k(\phi_k)[h_k(\phi_{k-1})].
\end{aligned}
$$

さらに suffix 側の primal / adjoint stationarity を満たすと、後段変数の一次変分は envelope 型に消え、

$$
D\widehat F_k(\phi_k)[\delta \phi_k]
=
-\langle p_k, \delta \phi_k \rangle
$$

が残る。したがって

$$
D_f \widetilde F_k^{\mathrm{func}}(f_k)[h_k]
=
-\langle p_k, h_k(\phi_{k-1}) \rangle
$$

である。すなわち `D_f \widetilde F_k = 0` とは、

$$
\langle p_k, h_k(\phi_{k-1}) \rangle = 0
\qquad
\forall h_k \in T_{f_k}\mathcal F_k
$$

という意味であり、adjoint `p_k` が current layer の可到達変分をすべて打ち消す、という条件にほかならない。

このとき reachable variation space を

$$
\mathcal H_k(\phi_{k-1}; f_k)
:=
\{h_k(\phi_{k-1}) \mid h_k \in T_{f_k}\mathcal F_k\}
$$

と書けば、上の条件は `p_k` が `\mathcal H_k(\phi_{k-1}; f_k)` を annihilate する、と言い換えられる。

### 5.3 `\theta_k` が解く方程式

current layer を parameter family `f_k(\cdot; \theta_k)` に制限すると、admissible variation は

$$
h_k = D_\theta f_k(\cdot; \theta_k)[\delta \theta_k]
$$

の形に限られる。このとき parameter pullback の微分は

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

となる。よって stationary point `\theta_k^\star` は

$$
D_{\theta_k}\widetilde F_k(\theta_k^\star)[\delta \theta_k] = 0
\qquad
\forall \delta \theta_k
$$

を満たし、同値に

$$
\bigl(D_{\theta_k} f_k(\phi_{k-1}; \theta_k^\star)\bigr)^* p_k^\star = 0
$$

を満たす。これが本 worktree の `theta_residual` の数学的意味である。

ただし `p_k^\star` は独立変数ではなく、`\theta_k^\star` を通じて決まる suffix 最適応答の結果である。したがって `\theta_k` は一般に単独の線形方程式を解くのではなく、

$$
R_k^{\mathrm{red}}(\theta_k)
:=
\bigl(D_{\theta_k} f_k(\phi_{k-1}; \theta_k)\bigr)^*
p_k^\star(\theta_k)
=
0
$$

という非線形の reduced root equation の解になる。

### 5.4 Full KKT root と局所更新方程式

Stage 0 実装では、上の reduced root を `\theta_k` だけで直接解く代わりに、

$$
u_k = (\theta_{k:N}, \phi_{k:N}, p_{k:N})
$$

を unknown とする full KKT root

$$
R_k(u_k)
:=
\begin{bmatrix}
r_k^\phi \\
r_k^p \\
r_k^\theta
\end{bmatrix}
=
0
$$

を解く。ここで `\theta_k^\star` は full root の先頭 block として得られる。これは reduced equation を明示的に潰し切る前に、primal residual、adjoint residual、parameter residual の整合を toy 問題で直接確かめるためである。

他方で、本 project が Bellman 的な更新則として重視しているのは、stationary point の定義方程式そのものではなく、そこへ向かう局所更新方程式

$$
G_k(\Delta \theta_k, \delta \eta_k)
=
-\widetilde \Lambda_k(\delta \eta_k)
\qquad
\forall \delta \eta_k
$$

である。座標表示では

$$
G_k \Delta \theta_k = -\tilde p_k,
\qquad
\theta_k^+ = \theta_k + \Delta \theta_k
$$

となる。この式は exact reduced stationarity と同じではない。前者は stationary point の定義であり、後者は induced metric を通して parameter space に引き戻した局所最急方向である。

この区別を保つと、solver の役割も自然に分かれる。exact reduced root `R_k^{\mathrm{red}}(\theta_k)=0` は一般に非線形なので Newton 型、Gauss-Newton 型、あるいは Newton-Krylov 型の対象になる。一方で `G_k \Delta \theta_k = -\tilde p_k` は線形系なので、`G_k` が対称正定値であれば CG が自然である。解析解を期待できるのは、`\theta_k` に関して `f_k` が線形で、`\widehat F_k` あるいは suffix value が二次になる線形二次 toy にほぼ限られる。

## 6. 批判的比較

### 6.1 現時点で最も近い文献は何か

構造の近さだけを問うなら、Carreira-Perpinan and Wang (2014) が最も近い。理由は、nested objective を explicit state と constraint に持ち上げる点が、suffix state、primal residual、adjoint residual を持つ本 worktree の設計と最もよく対応しているからである。

### 6.2 同一視してはいけないものは何か

本調査の最大の注意点は、次の三つを混同しないことである。

1. exact equality-constraint line
1. lifted / relaxed / Bregman line
1. optimizer line としての proximal BCD

これらは multiplier や block-coordinate update を共有しうるが、解いている optimization problem は同一ではない。したがって、「近い」という事実から「同じ」と結論してはいけない。

### 6.3 歴史的 layer-wise 文献は、どこまで使ってよいか

Hinton 2006 と Bengio 2007 は、layer-wise thinking が deep optimization に役立ったという文脈を与える。しかし、そこから「exact reduced problem の停留条件はすでに支持されている」と読むのは過剰である。今回の survey では、これらを historical motivation に限定し、exact line の理論的根拠に流用しないことを明示的な方針とする。

### 6.4 Solver claim は equation class に従って分けるべきである

文献の読みを solver claim に直すと、次のような切り分けが必要になる。

- exact reduced stationarity は一般に nonlinear root とみなすべきである
- induced-metric local update は linear system とみなせる
- proximal BCD は別系統の optimizer family であって exact reduced equation と同一ではない
- analytic closed form は linear-quadratic toy の範囲に限る

この切り分けを怠ると、「exact `DF=0` は CG で終わる」「gradient descent で十分」「lifted baseline と同じだ」といった誤った短絡を招く。

### 6.5 強い主張として避けるべきもの

本調査から導けない、あるいは導くべきでない主張は次の通りである。

- Bengio 2007 が exact layer-local reduced stationarity を証明した
- lifted baseline が exact `DF=0` objective と同値である
- current toy solver が scalability や backprop に対する優位性を示した
- current line が既存 trainer API を置き換える準備ができている

## 7. 現時点で確立できること

### 7.1 構造的には exact equality-constraint line を主線に置くのが自然である

文献全体を比較すると、本 worktree が canonical line として exact equality-constraint plus Lagrangian を採る判断は妥当である。これは novelty の主張ではなく、構造の最も近い先行研究がその線上にある、という意味での妥当性である。

### 7.2 Baseline は lifted / relaxed / BCD line を含めて設計すべきである

exact line を主線にしても、比較対象は単なる backprop だけでは足りない。Askari、Gu、Wang-Benning、Lau の系統を baseline 設計に含めることで、はじめて「exact line を選ぶ理由」と「その対価」を定量的に議論できる。

### 7.3 Inner solve を導入するなら微分境界も同時に設計すべきである

OptNet と DEQ が示すのは、inner solve を持つモデルでは forward solver と backward differentiation を分離できないという点である。本 worktree でも、toy implementation の次段でこの設計を避けることはできない。

## 8. 争点と未解決点

### 8.1 exact reduced value function は実用上 tractable か

文献は constrained reformulation の可能性を支持しているが、suffix-optimal value function 自体の regularity、uniqueness、numerical tractability を直接保証してはいない。exact line を本当に主線に据えるなら、ここが最大の open question である。

### 8.2 exactness と tractability の trade-off をどこで切るべきか

lifted / relaxed line は、問題を変える代わりに block structure や solver friendliness を得る。exact line は original problem への fidelity を保つ代わりに、inner solve と nonlinear root の負担を抱える。この trade-off をどこで切るべきかは、現時点ではまだ開いている。

### 8.3 実用的な update operator は何であるべきか

parameter-space の Euclidean metric、function-space からの pullback metric、あるいはその近似 preconditioner のどれを practical update operator とみなすべきかは、まだ決着していない。今回の survey は、この争点が重要であることは示すが、決定的な答えまでは与えない。

### 8.4 どこまでが toy prototype の主張範囲か

本 survey は toy `standard` network と small dense solve までを正当化するには十分である。しかし、`icnn` や既存 trainer 置換まで同じ論理で拡張できると主張するには、文献と evidence の両方が不足している。

## 9. 本 worktree への含意

以下は文献から機械的に導かれる唯一の結論ではなく、本調査で整理した支持線・反証候補・制約条件を踏まえて、本 worktree が採るのが自然だと判断した設計方針である。

### 9.1 採用すべき canonical line

- exact equality-constraint plus Lagrangian を canonical formulation とする
- Bellman と function-space の記法は [bellman_bp_chat_summary.md](/workspace/notes/themes/bellman_bp_chat_summary.md) に合わせる
- lifted / relaxed / Bregman / proximal BCD は comparison baseline として保持する

### 9.2 期待すべき solver policy

- Stage 0 では full KKT root に対する dense Newton を toy verification 用に使う
- exact reduced equation in `\theta_k` は一般に nonlinear root とみなす
- Bellman 的な local update equation `G_k \Delta \theta_k = -\tilde p_k` は CG を含む線形ソルバの対象とみなす
- analytic closed form は linear-quadratic toy に限定して使う

### 9.3 次に必要な証明義務

1. 2-layer linear toy で `\widehat F_k`、`\widetilde \Lambda_k`、reduced root が解析的に一致することを示す
1. exact equality-constraint toy と lifted / BCD baseline を少なくとも 1 本ずつ比較する
1. inner-solve differentiation boundary を、実装統合の前に明文化する

## 10. 結論

本調査の結論は単純である。`DF=0` 型の layer update という問いに対して、現在の文献は直接の完成形を与えていない。しかし、問題を置くべき場所、比較対象として何を並べるべきか、inner solve を導入したときに何が新たな責務になるかについては、十分に強いヒントを与えている。最も近い先行研究は auxiliary coordinates による constrained reformulation であり、最も強い baseline は lifted / relaxed / proximal BCD の系統であり、最も重要な solver-boundary reference は OptNet と DEQ である。

したがって本 worktree では、exact equality-constraint line を canonical に保ちつつ、baseline と solver choice を文献に即して慎重に切り分けるのが正しい。ここで必要なのは、都合のよい引用の積み上げではなく、「どこまでなら文献が支え、どこから先はまだ自分たちで検証しなければならないか」を明確に保つことである。その意味で、本 survey は実装の直前に置くための背景説明ではなく、今後の claim を縛るための制約文書でもある。

## 付録 A. 調査ログ

- `Question:`
  - 標準的な feedforward network の layer update を、Bellman 的な reduced problem の停留条件として与えられるか。
- `Search Log Summary:`
  - layer-wise training
  - auxiliary coordinates
  - lifted neural networks
  - Fenchel lifted networks
  - Bregman training
  - optimization as a layer
  - deep equilibrium models
  - equilibrium propagation
  - proximal block coordinate descent
- `Included:`
  - Hinton 2006
  - Bengio 2007
  - Carreira-Perpinan and Wang 2014
  - Askari et al.
  - Gu et al.
  - Wang and Benning
  - Lau et al.
  - Amos and Kolter
  - Bai et al.
  - Scellier and Bengio
- `Excluded:`
  - generic optimizer papers without layer-wise or lifted structure
  - hardware-only results without formulation consequences
  - architecture papers unrelated to constrained, lifted, or implicit-solve views

## 付録 B. 文献対応表

| 役割                                       | 文献                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 歴史的背景                                 | [Hinton_Osindero_Teh_2006_A_Fast_Learning_Algorithm_for_Deep_Belief_Nets.pdf](/workspace/references/lagrangian_df_zero/Hinton_Osindero_Teh_2006_A_Fast_Learning_Algorithm_for_Deep_Belief_Nets.pdf), [Bengio_2007_Greedy_Layer_Wise_Training_of_Deep_Networks.pdf](/workspace/references/lagrangian_df_zero/Bengio_2007_Greedy_Layer_Wise_Training_of_Deep_Networks.pdf)                                                                                                                                            |
| exact equality-constraint の直接的先行研究 | [Carreira-Perpinan_Wang_2014_Distributed_Optimization_of_Deeply_Nested_Systems.pdf](/workspace/references/lagrangian_df_zero/Carreira-Perpinan_Wang_2014_Distributed_Optimization_of_Deeply_Nested_Systems.pdf)                                                                                                                                                                                                                                                                                                     |
| lifted / relaxed baseline                  | [Askari_Negiar_Sambharya_ElGhaoui_2021_Lifted_Neural_Networks.pdf](/workspace/references/lagrangian_df_zero/Askari_Negiar_Sambharya_ElGhaoui_2021_Lifted_Neural_Networks.pdf), [Gu_Askari_ElGhaoui_2020_Fenchel_Lifted_Networks.pdf](/workspace/references/lagrangian_df_zero/Gu_Askari_ElGhaoui_2020_Fenchel_Lifted_Networks.pdf), [Wang_Benning_2023_Lifted_Bregman_Training_of_Neural_Networks.html](/workspace/references/lagrangian_df_zero/Wang_Benning_2023_Lifted_Bregman_Training_of_Neural_Networks.html) |
| optimizer baseline                         | [Lau et al. (official arXiv)](https://arxiv.org/abs/1803.09082)                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| root-solve / implicit differentiation      | [Amos_Kolter_2017_OptNet_Differentiable_Optimization_as_a_Layer_in_Neural_Networks.pdf](/workspace/references/lagrangian_df_zero/Amos_Kolter_2017_OptNet_Differentiable_Optimization_as_a_Layer_in_Neural_Networks.pdf), [Bai_Kolter_Koltun_2019_Deep_Equilibrium_Models.pdf](/workspace/references/lagrangian_df_zero/Bai_Kolter_Koltun_2019_Deep_Equilibrium_Models.pdf)                                                                                                                                          |
| contrast line                              | [Scellier_Bengio_2017_Equilibrium_Propagation.pdf](/workspace/references/lagrangian_df_zero/Scellier_Bengio_2017_Equilibrium_Propagation.pdf)                                                                                                                                                                                                                                                                                                                                                                       |

## 関連文書

- [Reference pack overview](/workspace/references/lagrangian_df_zero/README.md)
- [Reference catalog](/workspace/references/lagrangian_df_zero/catalog.md)
- [Design note](/workspace/documents/design/jax_util/lagrangian_df_zero.md)
- [Bellman notation note](/workspace/notes/themes/bellman_bp_chat_summary.md)
