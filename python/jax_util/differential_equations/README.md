# Differential Equations

この package は、微分方程式の問題定義と problem set を置く入口です。

当面の責務は次です。

- ODE / PDE / DAE などの問題メタデータを表す
- 問題群を 1 つの problem set としてまとめる
- 後続の solver / benchmark / neuralnetwork 実験から参照しやすい置き場を用意する

この段階では、個別問題の大量実装までは行いません。
まずは import-safe な problem catalog の土台だけを用意します。
