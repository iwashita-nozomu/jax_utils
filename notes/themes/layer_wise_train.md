# レイヤー毎のBP

## NNの構成

$\phi \in \Phi$ をレイヤー毎の関数の合成 $$\phi = f_N \circ ... \circ f_1$$ とする。
これに対し、無制約の最適化問題 $J[\phi]$ を考える。$f_k$ の更新時、$f_l, l<k$ は固定されると考えて、$l>k$のみ影響する。最適解$f_l^*$ 近傍で線形化されていると考えると、$$\frac{\partial J}{\partial f_l} = 0$$ であり情報を持たないので困ってしまう。
そこで、これを制約にしてしまうことを考えると、ラグランジアン$$L[f_N,...,f_k] = J_k + \sum_{i=k+1}^N {\left \langle \frac{\partial J}{\partial f_i}, \lambda_i \right\rangle}$$ である。すると $$\frac{\partial J}{\partial f_k} = 0 \\[4pt] \frac{\partial J}{\partial f_i} = 0 \\[4pt] \frac{\partial L}{\partial f_i} = 0 $$ が導出できる。$f_i$に関する微分作用を$\partial_{f_i}$ と定義すると、 より、

再帰的に考えると、$$\frac {\partial J_{k+1}}{\partial f_{k+1}} =0$$ を制約として持ち、$J_k$ を最小化する問題ととらえることができる。すなわち、
$$
\begin{aligned}
\min_{f_{k+1},f_k} \quad & J_{k}[f_{k+1},f_k] \\
\text{s.t.} \quad & \frac {\partial J_{k+1}}{\partial f_{k+1}} =0
\end{aligned}
$$
ただし、

## NNの構造

自己写像 $\phi_0: X \to X$ を与える。NNの第 $k$ 層の出力空間を$Z_k$とする。
$\phi_N: X \to Z_N$ がNNの表現する関数であり、各層を関数の変換 $$f_k: \Phi_{k-1} \to \Phi_{k}$$ とみなせる。
損失(目的)関数を
$$J:\Phi_N \to \mathbb{R}$$
とする。
各層の学習時に現れる目的関数を
$$
L_{k-1}[f_{k-1}] := \inf_{f_k}{L_k}|_{f_{k-1}}
$$
と定義する。
$$
L_N[f_N] = J[f_N(\phi_{N-1})]
$$
と定義すれば、すべての層における目的関数 $L_k$ が定まり、$$\min_{\Phi} J \leq L_N $$ である。
この解法は2通り考えられる。

1. 上位問題の全構築、保持

    $f_{k-1}$ の更新後、$f_k$を更新する。純粋な動的計画法

2. 一次情報のみを保持する方法
    $$\frac{d L_k}{d f_{k-1}}$$
    を保持し、$L_N$を見積もりながら、最適化を進める方法
