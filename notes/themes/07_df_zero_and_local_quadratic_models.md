# 07. `DF=0` と局所二次モデルの比較

## 問い

`DF=0` を exact reduced stationarity として扱うとき、`\widetilde F_k` の二次近似は定義上必要なのか。
必要でないなら、二次近似を入れるのは何のためか。
逆に二次近似を入れたとき、解いている方程式と数理的性質はどう変わるのか。

## 結論

短く言えば、`DF=0` の定義そのものに二次近似は不要である。
exact `DF=0` は

$$
D_{\theta_k}\widetilde F_k(\theta_k) = 0
$$

あるいは等価な reduced root

$$
R_k^{\mathrm{red}}(\theta_k) = 0
$$

として first-order に定義できる。

ただし、これだけで得られるのは stationary point の条件であって、local minimizer の判定や、1 ステップで使う数値更新則までは決まらない。
そこで局所二次モデルが役に立つ。
二次近似は「何を stationary point と呼ぶか」を決めるためではなく、

- Newton 型や Gauss-Newton 型の step を作る
- local minimizer と saddle / maximizer を分ける
- trust-region や damping を入れた安定化方針を書く
- linear-quadratic toy で closed form を得る

ための追加構造である。

## 設定

この project の記法では、固定した prefix state $\phi_{k-1}$ のもとで

$$
\widetilde F_k(\theta_k) := \widehat F_k(\Psi_k(\theta_k)),
\qquad
\Psi_k(\theta_k) = f_k(\phi_{k-1}; \theta_k)
$$

と書く。
ここで exact `DF=0` は

$$
D_{\theta_k}\widetilde F_k(\theta_k^\star)[\delta \theta_k] = 0
\qquad
\forall \delta \theta_k
$$

である。

この式は「stationary point の定義」であって、まだ更新式ではない。
一方で現在の設計には、Bellman 的な局所更新方程式

$$
G_k \Delta \theta_k = -\tilde p_k
$$

もある。
これは induced metric を通して covector を vector に変換する式であり、exact `DF=0` の定義そのものではない。

## 1. 二次近似を入れない場合

### 1.1 解いている問題

二次近似を入れない場合の主問題は、exact reduced stationarity そのものである。
代表的には次の 2 通りがある。

1. `\widetilde F_k` の微分を 0 にする
1. suffix 変数も含めた full KKT root `R_k(u_k)=0` を解き、その解から `\theta_k` の stationarity を読む

どちらも「exact な stationary condition」を扱っている。

### 1.2 必要な滑らかさ

stationarity を書くだけなら、基本的には `\widetilde F_k` の Fréchet 微分または Gateaux 微分が存在すればよい。
少なくとも `C^1` があれば、`D_{\theta_k}\widetilde F_k(\theta_k)=0` は意味を持つ。

### 1.3 得られる解の意味

この場合に得られるのは exact stationary point である。
良い点は、解いた問題がそのまま本来の `DF=0` であることだ。
悪い点は、stationary point には local minimum だけでなく saddle や local maximum も含まれることだ。

したがって、

- `DF=0` を満たした
- reduced problem の exact stationarity を解いた

とは言えても、

- loss が局所的に最小である
- descent step として常に良い方向を返す

までは言えない。

### 1.4 数値解法の class

exact root は一般に nonlinear である。
理由は、`D_{\theta_k}f_k` だけでなく、後段を最適化した結果として現れる adjoint や suffix state も `\theta_k` に依存するからだ。

したがって自然な solver family は

- Newton
- Newton-Krylov
- full KKT root に対する Newton

である。

CG 単体で「問題全体が終わる」とは通常言えない。
CG は使えても、Newton 線形化の内側で step を解くための線形 solver としてである。

## 2. 局所二次モデルを入れる場合

### 2.1 典型形

現在点 `\theta_k` のまわりで、`\widetilde F_k` の局所二次モデル

$$
m_k(\delta \theta_k)
=
\widetilde F_k(\theta_k)
\quad + \langle g_k, \delta \theta_k \rangle
\quad + \frac12 \langle \delta \theta_k, B_k \delta \theta_k \rangle
$$

を考える。
ここで

$$
g_k = \nabla \widetilde F_k(\theta_k)
$$

であり、`B_k` は次のいずれかである。

- exact Hessian `\nabla^2 \widetilde F_k(\theta_k)`
- Gauss-Newton 近似
- Levenberg-Marquardt のような damped 近似

このとき step は

$$
B_k \delta \theta_k = -g_k
$$

あるいは trust-region / damping を入れた変形になる。

### 2.2 何が変わるか

ここで解いているのは、もはや exact `DF=0` そのものではなく、現在点のまわりで作った local model の minimization である。
`B_k = \nabla^2 \widetilde F_k` を使う Newton step であっても、

- step 自体は local model の最小化
- stationary point の定義は元の `D_{\theta_k}\widetilde F_k = 0`

という二層構造になっている。

つまり、二次近似は「定義」ではなく「数値 step の作り方」である。

### 2.3 必要な滑らかさ

この場合は少なくとも `C^2` が欲しい。
Hessian をそのまま使うなら二階微分可能性が必要である。
Gauss-Newton や近似 Hessian を使う場合でも、最低限 local linearization の正当化が必要になる。

### 2.4 得られるもの

局所二次モデルを入れると、stationarity そのものではなく

- local minimizer に向かう step
- curvature を使った step quality
- local convergence rate の議論

が書けるようになる。

たとえば Hessian が local minimizer で正定値なら、Newton 型は局所二次収束を期待できる。
一方で Hessian が不定なら、二次モデルも saddle 方向や負曲率方向を持ちうる。
その場合は trust region や damping が必要になる。

## 3. 「二次近似あり」と「なし」の比較

| 観点            | 二次近似なし                                                | 二次近似あり                                                  |
| --------------- | ----------------------------------------------------------- | ------------------------------------------------------------- |
| 主問題          | exact `D_{\theta_k}\widetilde F_k = 0` または full KKT root | local quadratic model `m_k` の最小化                          |
| 役割            | stationary point の定義                                     | local step の構成、curvature 利用                             |
| 必要な滑らかさ  | まずは `C^1`                                                | 通常は `C^2` 以上                                             |
| 解の意味        | exact stationary point                                      | 現在点まわりの model-based step                               |
| minimizer 判定  | これだけでは弱い                                            | Hessian / 近似 Hessian で二次十分条件を議論しやすい           |
| 方程式の class  | 一般に nonlinear root                                       | 1 回ごとの subproblem は linear system または trust-region QP |
| 自然な solver   | Newton, Newton-Krylov, full KKT Newton                      | Newton, Gauss-Newton, LM, trust-region                        |
| `CG` の位置づけ | root 全体の solver ではなく内側線形 solver                  | `B_k` が SPD なら自然に使える                                 |
| claim の強さ    | 「exact `DF=0` を解いた」と言える                           | 「exact `DF=0` の local quadratic step を使った」と言うべき   |
| 主な壊れ方      | stationary point が saddle かもしれない                     | model mismatch、indefinite Hessian、過大 step                 |

## 4. linear-quadratic regime では区別が消える

二次近似が exact `DF=0` と一致する特別な場合がある。
それは

- `f_k(\phi_{k-1}; \theta_k)` が `\theta_k` に線形
- `\widehat F_k` が `\phi_k` に二次

で、したがって `\widetilde F_k(\theta_k)` が `\theta_k` に二次になる場合である。

この regime では

$$
\widetilde F_k(\theta_k)
=
\frac12 \theta_k^\top H_k \theta_k + b_k^\top \theta_k + c
$$

となるので、exact stationarity も局所二次モデルも同じ線形方程式

$$
H_k \theta_k + b_k = 0
$$

に落ちる。
このときだけ、「二次近似で十分か」という問いへの答えは「十分であり、しかも exact」である。

現在の `standard` network 一般論はこの regime ではない。
したがって、linear-quadratic toy での closed form を一般の nonlinear layer へそのまま拡張してはいけない。

## 5. induced-metric step は二次 Taylor 近似とは別物

この project で使っている

$$
G_k \Delta \theta_k = -\tilde p_k
$$

は、見た目には二次形式を含むので局所二次モデルに似ている。
しかしこれは通常、`\widetilde F_k` の Hessian そのものではない。
ここで使っている `G_k` は reachable tangent 上の計量を parameter space へ引き戻したものであり、Riesz map を通して covector を更新方向へ変換している。

したがってこの式は

- exact `DF=0` そのものでもない
- `\widetilde F_k` の二次 Taylor 近似そのものでもない

という中間の立場にある。

これは「stationarity へ向かう局所更新式」と読むのがよい。
`G_k` が SPD なら CG が自然だが、それはあくまで metric step に対する話であり、exact reduced root 全体に対する話ではない。

## 6. この project での使い分け

現時点の project では、次の三段階を分けて扱うのが妥当である。

### 6.1 exact `DF=0` を主張したいとき

このとき二次近似は不要である。
正本はあくまで

$$
D_{\theta_k}\widetilde F_k(\theta_k) = 0
$$

またはそれと等価な reduced root / full KKT root である。
report や design でも、「exact root を解いた」のか「その local model を解いた」のかを混ぜない。

### 6.2 安定な局所 step が欲しいとき

このとき二次近似は有用である。
ただし主張は

- Newton step
- Gauss-Newton step
- damped local quadratic model

のように書くべきで、exact `DF=0` と同義にはしない。

### 6.3 local minimum を区別したいとき

このとき二次情報はほぼ必須である。
`DF=0` だけでは stationary point の class が弱いからだ。
少なくとも Hessian の正定性、あるいはそれに相当する二次十分条件を別に確認する必要がある。

## 7. 実務上の判断

この問いに対する project-local な判断は次である。

1. exact `DF=0` の定義に `\widetilde F_k` の二次近似は要らない
1. ただし minimizer 判定、Newton 型 step、trust-region、closed form toy には二次情報が要る
1. induced-metric step `G_k \Delta \theta_k = -\tilde p_k` は Hessian model ではなく、metric による first-order update として扱う
1. 文書では `exact root`, `local quadratic model`, `metric step` を常に別名で呼ぶ

## References

- [lagrangian_df_zero.md](/workspace/documents/design/jax_util/lagrangian_df_zero.md)
- [06_lagrangian_df_zero.md](/workspace/notes/themes/06_lagrangian_df_zero.md)
- [bellman_bp.md](/workspace/notes/themes/bellman_bp.md)
- [bellman_bp_chat_summary.md](/workspace/notes/themes/bellman_bp_chat_summary.md)
