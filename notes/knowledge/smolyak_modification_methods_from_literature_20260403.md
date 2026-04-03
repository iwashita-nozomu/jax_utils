# Smolyak Modification Methods From Literature

## Purpose

現行の [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py) を、論文と既存実装を踏まえてどう改造するのが筋かをまとめるメモ。

ここでの主眼は

- 数学を変えずに速くする案
- 数学を変えるが term 数を減らせる案
- JAX 実装として現実的な案

を分けて整理すること。

## Current State

現行実装の特徴:

- 1D rule は差分則 `Δ_l` を使っている。
- term 配列の全保持はやめ、term 自体は動的生成している。
- ただし term ごとの point decode は毎回 mixed-radix の `div/rem + gather` で行っている。
- outer `vmap` に対しては wrapper batching を入れ始めているが、速度はまだ不十分。

したがって次に効くのは、

- `difference rule` を使っていること自体

ではなく、

- `difference rule` の「隣の点・隣の term との差分更新しやすさ」
- tensor product の「shared prefix / suffix」

を runtime で利用すること。

## Literature And Existing Implementations

### 1. Sparse grids as lower sets and surplus operators

Tasmanian の数理マニュアルは、global / sequence grid を

- 1D operators `U_l`, `Q_l`
- surplus operators `Δ_l = U_l - U_{l-1}` または `Q_l - Q_{l-1}`
- lower set `Θ`

で整理している。さらに sequence grid は repeated evaluation が速いが、係数構築は高コストで、特に vector-valued output では重いとしている。  
Sources:

- [Tasmanian Math Manual](https://mkstoyanov.github.io/tasmanian_aux_files/docs/TasmanianMathManual.pdf)
- [Tasmanian docs](https://ornl.github.io/TASMANIAN/rolling/group__TasmanianSG.html)

今のコードへの含意:

- 現在の差分則利用は数学的には正しい。
- ただし sequence-grid 的な「一度 surpluses / coefficients を作って repeated evaluation を速くする」道は、今の `integrate(f)` 単発カーネルにはそのままは向かない。
- ただし「多出力 / 多回数 evaluation では coefficient path が有利になる」こと自体は重要な比較対象。

### 2. Dimension-adaptive / adaptive sparse grids

Gerstner–Griebel の dimension-adaptive quadrature と、Bungartz–Dirnstorfer の adaptive sparse-grid quadrature は、

- admissible lower set
- surplus / contribution に基づく refinement

で term 数そのものを減らす方向の代表。  
Sources:

- Gerstner, Griebel, *Dimension-Adaptive Tensor-Product Quadrature*, Computing 71(1), 65–87 (2003), DOI: <https://doi.org/10.1007/s00607-003-0015-5>
- Bungartz, Dirnstorfer, *Multivariate Quadrature on Adaptive Sparse Grids*, Computing 71(1), 89–114 (2003), DOI: <https://doi.org/10.1007/s00607-003-0016-4>

今のコードへの含意:

- 高次元高 level の execution frontier を押し上げる本命ではある。
- ただし canonical same-budget baseline からは外れる。
- baseline 測定中の今すぐの改造対象ではなく、「math-changing optimization」として分離して扱うべき。

### 3. Clustered evaluation by discrete Fubini / dimension reduction

MDI-SG は sparse-grid の点選択自体は維持しつつ、function evaluation を「次元を落としながら cluster で再利用する」方向に書き換える。論文の主張は、

- sparse-grid の理論収束は変えない
- 実装上の repeated computation を減らす
- function evaluation を collectively 処理する

というもの。  
Source:

- [An Efficient and Fast Sparse Grid Algorithm for High-Dimensional Numerical Integration](https://www.mdpi.com/2227-7390/11/19/4191)

今のコードへの含意:

- これは今の pain point にかなり近い。
- ただし論文の実装は symbolic function formation をかなり使うので、generic JAX callable にそのままは移植しにくい。
- そのままコピーするのではなく、「shared prefix / suffix の再利用」「tile-local partial sum」「小さい carry だけ持つ」という構造を取り出すのがよい。

### 4. Dynamic memory layout and compact storage

Jacob の regular sparse-grid hierarchization の仕事は、

- compact contiguous storage
- phase ごとに適した memory layout
- generalized counting で定数時間の row access

を使う。SG++ 本体も、

- default hash-based storage
- performance critical parts を C++
- tensor product structure を使う template support

と明記している。  
Sources:

- [Efficient Regular Sparse Grid Hierarchization by a Dynamic Memory Layout](https://studwww.itu.dk/~rikj/Publications/SGAProceedings.pdf)
- [SG++ docs](https://sgpp.sparsegrids.org/docs/)

今のコードへの含意:

- 今の JAX 実装でも、small state carry と contiguous tiny buffers に寄せるのが筋。
- 大きい tensor を carry するより、小さい digit / index / weight state を持つべき。
- metadata を array-of-struct 的にばらまくより、contiguous で phase 特化な表現のほうが有利。

### 5. Sequential-over-dimensions to avoid repeated calculations

Smolyak の量子ダイナミクス実装では、直積 grid の和をそのまま全部回すのではなく、各次元を順に処理して repeated computation を避ける、という実装経験が報告されている。  
Source:

- [Efficient Implementation of a Smolyak Sparse-grid Scheme with Non-nested Grids](https://open.library.ubc.ca/cIRcle/collections/48630/items/1.0307375)

今のコードへの含意:

- まさに current `_smolyak_plan_integral(...)` の「term ごとに毎回 decode / assemble」をそのままやり続けるのは避けたい。
- ただし generic JAX callable に対しては「完全な symbolic reduction」より、「prefix / suffix ごとの shared work を reuse する」形に落とすのが現実的。

### 6. Mixed-radix Gray-code / loopless generation

mixed-radix の Gray-code あるいは loopless reflected mixed-radix generation は、

- consecutive states differ in one digit
- per-step update can be `O(1)`

を狙う。  
Representative sources:

- [Dense Gray Codes in Mixed Radices](https://digitalcommons.dartmouth.edu/senior_theses/117/)
- [Loopless reflected mixed-radix Gray generation notes](https://www.inf.fu-berlin.de/lehre/WS15/AAA/Loopless-Gray.pdf)

今のコードへの含意:

- これは `_decode_points_and_weights(...)` の置換候補として非常に相性がよい。
- point ごとの full decode `O(d)` を、償却 `O(1)` に近づけられる可能性がある。
- JAX では large tensor carry ではなく、
  - digits
  - direction bits
  - per-axis current node / weight
  - current weight product
  - partial sum
 だけを carry にするのが王道。

## What To Change First

### A. Gray-style incremental point traversal inside each tensor term

これは最優先候補。

狙い:

- 各 point で `div/rem + gather` を全部やり直さない
- 次の point では変わる軸だけ更新する

実装イメージ:

1. term ごとに current digits `a_k` を持つ
2. reflected mixed-radix Gray-like orderで次点へ進む
3. 変化した軸 `j` だけ
   - `x_j`
   - `w_j`
   - `weight_product`
   を更新
4. tile 内で部分和へ即加算

期待効果:

- current decode は `Theta(d * points)`
- Gray traversal では平均変更軸数 `r` に対して `Theta(r * points)`
- `r` が小さければ大きい改善余地がある

注意:

- point 完全逐次 `scan` にすると GPU 並列性を潰しうる
- したがって「tile 内は小さい並列 block」「tile 間は scan/map」がよい

### B. Tile-local partial sums instead of full chunk materialization

これは Gray traversal と組み合わせる本命。

狙い:

- `values = vmap(f)(points)` の chunk 全体 materialization を避ける
- `weights` ベクトルを大きく持ち回らない
- 部分和をもっと内側で取る

実装イメージ:

- chunk をさらに small tile に分ける
- tile でだけ point block を assemble
- `vmap(f)` は tile に対して使う
- `tile_partial_sum += values @ weights`
- term acc にすぐ加える

期待効果:

- `batch x chunk` temp を `batch x tile` へ落とせる
- decode と reduction の間の大きな中間を減らせる

批判的コメント:

- これは以前の完全逐次化とは違う
- point 単位 `scan` は rejected だったが、tile 単位 partial sum はまだ有望

### C. Shared prefix / suffix reuse inspired by MDI-SG

これは B の次。

狙い:

- shared prefix を持つ point 群の work をまとめる
- term ごとに毎回 full assemble しない

現実的な落とし込み:

- symbolic function formation までは行かない
- 代わりに「prefix は固定、suffix だけ sweep」またはその逆を使う
- current `batched_suffix_ndim` は already available な hook なので、それを mode ではなく internal planning に使う

批判的コメント:

- generic `f` では論文そのままの MDI-SG にはならない
- ただし「shared work を cluster で reuse」という核心はそのまま使える

### D. Compact carry and layout cleanup

これは上の 3 つを支える基盤。

方針:

- carry は small state だけ
- large point/value tensor は carry に入れない
- metadata は contiguous small buffer にまとめる

特に避けるもの:

- `dynamic_update_slice` で巨大 block を毎回書く
- scan carry に `dim x tile` の大配列を抱える
- suffix 展開を早い段階で dense 化する

## What To Defer

### Sequence-grid style coefficient path

Tasmanian は repeated evaluation に sequence grid が効くとしているが、同時に

- coefficient construction は高い
- vector-valued output で特に高い

とも述べている。今の `integrate(f)` kernel の主課題とは少しずれる。

したがって:

- repeated evaluation of the same sparse-grid surrogate が本題になるまでは defer

でよい。

### Adaptive / anisotropic lower-set selection

これは term 数削減には本質的だが、

- baseline 比較条件を変える
- canonical Smolyak から外れる

ので、execution-speed 改善の段階では defer が自然。

## Repository-Specific Plan

現状コードへ落とす順序はこれがよい。

1. [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py) に phase annotation を入れる  
   まず `decode`, `integrand_eval`, `reduction`, `term_update`

2. `_decode_points_and_weights(...)` を置き換えるのではなく、term-local traversal helper を新設する  
   ここに Gray-style incremental update を入れる

3. current chunk path を「tile-local partial sum」へ切る  
   `chunk -> tiles -> partial sums`

4. wrapper batching は維持し、tile size は internal heuristic のまま  
   mode 切替は増やさない

5. 改善が確認できた後にだけ、shared prefix / suffix reuse を強める

## Bottom Line

文献と既存実装から見ると、今もっとも筋が良い修正は次の 2 本。

- mixed-radix / Gray-like incremental traversalで point decode を差分更新化する
- tile-local partial sums で large temporary を減らしつつ GPU 並列性を残す

逆に、今すぐではないものは次の 2 本。

- sequence-grid 的な coefficient path
- dimension-adaptive / anisotropic lower-set

つまり、次の改造方針は

- 数学はそのまま
- state は小さく
- 更新は差分で
- 部分和は tile 単位で早めに取る

がよい。

## References

- Bungartz, Griebel, *Sparse grids*, Acta Numerica 13 (2004), DOI: <https://doi.org/10.1017/S0962492904000182>
- Gerstner, Griebel, *Dimension-Adaptive Tensor-Product Quadrature*, Computing 71(1), 65–87 (2003), DOI: <https://doi.org/10.1007/s00607-003-0015-5>
- Bungartz, Dirnstorfer, *Multivariate Quadrature on Adaptive Sparse Grids*, Computing 71(1), 89–114 (2003), DOI: <https://doi.org/10.1007/s00607-003-0016-4>
- Tasmanian Math Manual: <https://mkstoyanov.github.io/tasmanian_aux_files/docs/TasmanianMathManual.pdf>
- Tasmanian docs: <https://ornl.github.io/TASMANIAN/rolling/group__TasmanianSG.html>
- SG++ docs: <https://sgpp.sparsegrids.org/docs/>
- Jacob, *Efficient Regular Sparse Grid Hierarchization by a Dynamic Memory Layout*: <https://studwww.itu.dk/~rikj/Publications/SGAProceedings.pdf>
- *An Efficient and Fast Sparse Grid Algorithm for High-Dimensional Numerical Integration* (MDI-SG): <https://www.mdpi.com/2227-7390/11/19/4191>
- Lauvergnat, *Efficient Implementation of a Smolyak Sparse-grid Scheme with Non-nested Grids*: <https://open.library.ubc.ca/cIRcle/collections/48630/items/1.0307375>
- Fan, *Dense Gray Codes in Mixed Radices*: <https://digitalcommons.dartmouth.edu/senior_theses/117/>
- *Loopless reflected mixed-radix Gray generation notes*: <https://www.inf.fu-berlin.de/lehre/WS15/AAA/Loopless-Gray.pdf>
