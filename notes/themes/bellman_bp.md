# 動的計画法

## 定式化

状態 $\phi_k$ 、制御 $f_k$ , 状態遷移 $T$ 、各段のコスト $ c_k (\phi_k)$

$$\phi_{k+1} = T(\phi_{k}, f_{k+1}) \\[4pt]
V_k(\phi_{k}) = c(\phi_k) + \min_{f_{k+1}} V_{k+1}({T(\phi_{k},f_{k+1})})
$$

コストは最終段における目的関数 $J[\phi_N]$ のみが有効なので再帰は

$$
\left\{
    \begin{aligned}
    V_N(\phi_N) &= J[\phi_N] = J[f_N \circ \phi_{N-1}] 　\\[4pt]
    V_k(\phi_k) &= \min_{f_{k+1}} V_{k+1}(T(\phi_k, f_{k+1})) \quad 1<k<N
    \end{aligned}
\right.
$$

最適化問題はminの部分のことなので、ここの構築がキモ

各段で解くべき最適化問題は $\min_{f_k} V_{k+1}(T(\phi_k, f_{k+1}))$ なので、ここを実装すればいい。
