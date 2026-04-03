# Smolyak Dynamic Generation Literature Note

Date: 2026-04-02

## Question

`term plan` を全生成せず、Smolyak の multi-index と term 内 tensor 点を動的生成できるか。

## Short Answer

できる。文献から見ると、実装方針は次の 2 段に分けるのが自然である。

1. Smolyak の multi-index 集合は、全列挙ではなく `down-set frontier` か `rank/unrank` で動的生成する。
2. 各 term の tensor 点は、全 index table を持たずに `mixed-radix` の `scan` で動的生成する。

## Relevant Literature

1. Bungartz, H.-J., Griebel, M. "Sparse grids", Acta Numerica 13, 2004.
   Link: https://doi.org/10.1017/S0962492904000182
   Accessible summary: https://www.cambridge.org/core/journals/acta-numerica/article/abs/sparse-grids/47EA2993DB84C9D231BB96ECB26F615C

2. Hegland, M. "Adaptive sparse grids", ANZIAM Journal 44(E), 2003.
   Link: https://doi.org/10.21914/anziamj.v44i0.685
   Accessible page: https://journal.austms.org.au/ojs/index.php/ANZIAMJ/article/view/685

3. Hegland, M., Leopardi, P. "The rate of convergence of sparse grid quadrature on the torus", ANZIAM Journal 52, 2011.
   Link: https://doi.org/10.21914/anziamj.v52i0.3952
   PDF: https://maths-people.anu.edu.au/~leopardi/CTAC-2010-HL-Sparse-Grid-Torus-revised.pdf

4. Stoyanov, M. K., Webster, C. G. "A Dynamically Adaptive Sparse Grid Method for Quasi-Optimal Interpolation of Multidimensional Analytic Functions", 2015.
   Link: https://doi.org/10.48550/arXiv.1508.01125
   Page: https://arxiv.org/abs/1508.01125

5. Jakeman, J. D., Roberts, S. G. "Local and Dimension Adaptive Sparse Grid Interpolation and Quadrature", 2011.
   Link: https://doi.org/10.48550/arXiv.1110.0010
   Page: https://arxiv.org/abs/1110.0010

6. van Zanten, A. J. "The Ranking Problem of a Gray Code for Compositions", Ars Combinatoria 41, 1995.
   Page: https://combinatorialpress.com/ars-articles/volume-041-ars-articles/the-ranking-problem-of-a-gray-code-for-compositions/

7. Genitrini, A., Pepin, M. "Lexicographic Unranking of Combinations Revisited", Algorithms 14(3):97, 2021.
   Link: https://doi.org/10.3390/a14030097
   Page: https://www.mdpi.com/1999-4893/14/3/97

8. Vajnovszki, V., Vernay, J.-F. "Restricted compositions and permutations: From old to new Gray codes", Information Processing Letters 111(13), 2011.
   Link: https://doi.org/10.1016/j.ipl.2011.03.022
   Abstract page: https://www.sciencedirect.com/science/article/abs/pii/S0020019011000913

9. Bouyuklieva, S., Bouyukliev, I., Bakoev, V., Pashinska-Gadzheva, M. "Generating m-Ary Gray Codes and Related Algorithms", Algorithms 17(7):311, 2024.
   Link: https://doi.org/10.3390/a17070311
   PDF: https://pdfs.semanticscholar.org/d95b/f6eebf0031358eeb85d74f40cc14f23a0bb7.pdf

10. Murarasu, A. F., Weidendorfer, J., Buse, G., Butnaru, D., Pfluger, D. "Compact data structure and scalable algorithms for the sparse grid technique", PPOPP 2011.
    Summary page: https://www.sciweavers.org/publications/compact-data-structure-and-scalable-algorithms-sparse-grid-technique
    DBLP entry: https://dblp.org/rec/conf/ppopp/MurarasuWBBP11

## What The Literature Supports

### 1. Multi-index は down-set と active frontier で扱うのが正道

Hegland, Gerstner-Griebel 系では、Smolyak の index set を `down-set` として扱い、全列挙ではなく admissible frontier を更新する形が中心である。Hegland-Leopardi の解説では、index set は `down-set` として扱うべきで、適応アルゴリズムは admissible な候補の中から profit/cost が最も高い index を追加する形になっている。

実装含意:

- `term_levels` 全体を最初に配列化しない。
- 保持するのは現在の `accepted down-set` と `active frontier` のみ。
- 等方 regular Smolyak の場合でも、`frontier` を lexicographic 順または profit 順で順に吐けばよい。

### 2. isotropic regular Smolyak なら rank/unrank で term を直接復元できる

regular isotropic Smolyak の term 集合は

`Lambda(d, l) = { i in N^d : |i|_1 <= d + l - 1 }`

であり、これは stars-and-bars により組合せと同一視できる。したがって term を `rank -> multi-index` で直接復元できる。van Zanten は composition Gray code の ranking / unranking を、Genitrini-Pepin は combination unranking を整理している。

実装含意:

- `term_levels` を `(T, d)` で持たなくても、`term_rank` からその場で multi-index を復元できる。
- isotropic case では `down-set frontier` すら不要で、単純な rank/unrank iterator に落とせる。
- weighted case では単純な closed form が崩れるので、frontier 方式のほうが素直。

### 3. term 内 tensor 点は mixed-radix の Gray / scan で回せる

Vajnovszki-Vernay と 2024 の m-ary Gray code 論文は、restricted compositions や m-ary tuples を小さな差分で列挙できることを示している。Smolyak の term 内点列挙は、まさに各軸ごとに長さが違う mixed-radix tuple 列挙なので、ここは `carry-based odometer` または `reflected mixed-radix Gray code` で動的生成できる。

実装含意:

- `local_point_index -> div/mod` を毎回やる代わりに、chunk 内では `lax.scan` の状態更新で suffix index を進められる。
- suffix batched 実装では、prefix だけ rank/unrank し、suffix は `scan` で連続生成するのがよい。
- Gray code を使えば隣接点の差分は 1 軸だけなので、node/weight gather の再利用余地がある。

### 4. full permutation group ではなく mixed-radix 更新規則が本体

これは文献と現在の数理整理の両方から言える。任意の term では各軸の rule length が一致しないため、`S_n` の単純な群作用だけで全点をきれいに走査するのは一般には無理である。等長軸の入れ替えは使えても、本体は permutation ではなく mixed-radix の更新規則である。

これは文献からの直接引用ではなく、上記文献と現行実装構造からの推論である。

### 5. GPU 向けには「自然数との bijection」が重要

Murarasu らは regular sparse grid 上の点集合と連続自然数の間の bijection を軸に、GPU に向く compact data structure を作っている。対象は hierarchical sparse grid storage で本実装と完全一致ではないが、

- 全点を木や hash で持たない
- rank/unrank で point を復元する
- 連続整数区間を GPU 上で並列処理する

という設計原理は今回の `lazy-indexed` に非常に近い。

## Recommended Rule Set For This Codebase

文献を踏まえると、次の順で改造するのが最も自然である。

### Stage 1: term metadata の全生成をやめる

regular isotropic case では `term_rank -> multi-index` の unranking を実装し、`term_levels`, `term_rule_offsets`, `term_rule_lengths`, `term_axis_strides`, `term_num_points`, `term_point_offsets` の全生成をやめる。

必要最低限の保持:

- 1D rule storage
- `d`, `level`
- `batched_suffix_ndim`

### Stage 2: term ごとの point 列挙を `scan` 化する

各 term について

- prefix: rank/unrank or odometer
- suffix: `lax.scan` による mixed-radix update

で点を生成する。これで現在の `searchsorted + div/mod` を chunk 全点に対して打つ形よりも GPU 向きになる。

### Stage 3: weighted / adaptive case は active frontier のみ保持

weighted Smolyak や dimension-adaptive 化では、Hegland / Gerstner-Griebel 型の admissible frontier を採用し、`down-set` を保ちながら候補だけを伸ばす。

## Bottom Line

文献上の本命は `term plan full materialization` ではない。

- regular isotropic: `combinadics/stars-and-bars` による term unranking
- weighted/adaptive: `down-set frontier`
- term 内 tensor 点: `mixed-radix scan` または `mixed-radix Gray code`

この 3 つを合わせると、現在の O(`num_terms * dimension`) term metadata を O(`frontier_size * dimension`) か O(`dimension`) に落とせる見込みがある。
