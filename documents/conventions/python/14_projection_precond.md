# 投影・前処理の記法

## 要約
- 投影は `proj @ v` の形に統一します。
- 前処理は `precond @ r` を使います。

## 規約
- 投影は `proj @ v` の形で統一し、`proj` は `LinOp` とします。
- 前処理は `precond @ r` の形で統一し、合成は `proj * precond * proj` を基本とします。
- 恒等写像の既定引数は避け、必要な場所で `LinOp(lambda x: x)` を定義します。
