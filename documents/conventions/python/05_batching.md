# バッチ対応

この章は、`LinOp` と `Matrix` のバッチ扱いを整理します。

## 要約

- `LinOp` で 1D/2D を内部で分岐します。
- `Matrix` はバッチとして扱う場合を明示します。

## 規約

- バッチ軸（`axis=-1`）などの共通定義は `documents/coding-conventions-project.md` の **5.2** に集約します。
- 本章は `LinOp` の **入出力が `Vector` / `Matrix` に揃う**ことだけを扱います。
- `LinOp` は `Vector` / `Matrix` を受け、内部で 1D/2D の分岐を行います。
- `Matrix` を「ベクトルのバッチ」として扱う場合は、意図をコメントで明示します。
