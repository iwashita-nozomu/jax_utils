# Smolyak Permutation-Group Note

## Goal

Smolyak の各 tensor-product term で現れる局所格子点を、「全インデックス列挙」ではなく、

- 1 回の小さな更新を繰り返す経路として走査できるか
- そのとき「完全な置換群」をどこまで厳密に使えるか
- trailing suffix を batched にすると何が群論的に正当で、何がそうでないか

を整理する。

このノートの結論は次の 2 点である。

1. 全局所点の厳密な網羅には、suffix 軸の単なる permutation だけでは不十分で、mixed-radix Gray code のような「値も動く」経路が必要である。
2. suffix 軸に対する full symmetric group `S_s` は、suffix 各軸の rule length がすべて等しいときにだけ厳密な自己同型として使える。一般の weighted / anisotropic term では、許されるのは等しい長さの軸どうしを入れ替える stabilizer subgroup に限られる。

## Sources

- Vajnovszki and Vernay, "Restricted compositions and permutations: From old to new Gray codes", Information Processing Letters 111 (2011), DOI: <https://doi.org/10.1016/j.ipl.2011.03.022>
- Greppi, Bosio, Arato, "Efficient computation paths for the systematic analysis of sensitivities", Computer Physics Communications 184 (2013), DOI: <https://doi.org/10.1016/j.cpc.2012.08.018>
- Even, "On the generation of permutations", Information Processing Letters 3 (1975), DOI: <https://doi.org/10.1016/0020-0190(75)90037-X>

これらをどう使うか:

- Vajnovszki-Vernay は product set に対する reflected Gray code の再帰定義と、隣接 2 点が 1 座標だけ `±1` だけ変化する事実の直接の根拠になる。
- Greppi らは mixed-radix reflected Gray code を grid graph 上の well-behaved path として解釈しており、Smolyak term の局所 tensor grid を Hamiltonian path として舐める見方に対応する。
- Even は permutation 自体を adjacent transposition で全列挙する古典で、suffix 軸の並べ替えに permutation group を持ち込む際の基礎として使う。

## Setting

1 つの Smolyak term を固定し、その各軸の 1 次元差分則の長さを

`L = (L_1, ..., L_d)`, `L_i >= 1`

とする。term 内の局所点集合は

`T(L) = {0, ..., L_1-1} x ... x {0, ..., L_d-1}`

であり、その要素 `u = (u_1, ..., u_d)` は各軸の local rule index を表す。

このとき必要なのは

- `T(L)` を重複なく全列挙すること
- できれば隣接する 2 点の差分が小さいこと
- できればその列挙を群作用や checkpoint + update で圧縮表現できること

である。

## Proposition 1: Reflected Mixed-Radix Gray Code Covers Every Local Point Exactly Once

### Claim

`b_i = L_i - 1` とおく。Vajnovszki-Vernay が用いる product-set reflected Gray code `G_d(b)` は `T(L)` の全要素をちょうど 1 回ずつ列挙し、隣接 2 要素はただ 1 つの座標だけ `+1` あるいは `-1` だけ変化する。

### Recursive Definition

`b' = (b_2, ..., b_d)` とする。`G_d(b)` を

- `d = 1` なら `0, 1, ..., b_1`
- `d > 1` なら、prefix `a = 0, 1, ..., b_1` ごとに
  - `a` が偶数なら `a G_{d-1}(b')`
  - `a` が奇数なら `a rev(G_{d-1}(b'))`

を連結した列

として定義する。

### Proof

帰納法で示す。

- 基底 `d = 1`:
  `G_1(b_1)` は `0, 1, ..., b_1` そのものであり、`{0, ..., b_1}` をちょうど 1 回ずつ列挙する。隣接項は 1 だけ変化する。
- 帰納段:
  `G_{d-1}(b')` が主張を満たすと仮定する。固定した `a` についてブロック `a G_{d-1}(b')` または `a rev(G_{d-1}(b'))` は、prefix が `a` の全 suffix をちょうど 1 回ずつ列挙する。異なる `a` のブロックは第 1 座標が異なるので互いに素である。したがって全ブロックの連結は `T(L)` 全体をちょうど 1 回ずつ覆う。

隣接差分については 2 種類だけ見ればよい。

- 同一ブロック内:
  帰納法の仮定より suffix 側のただ 1 座標だけが `±1` だけ変化する。
- 隣接ブロック境界:
  偶奇で順序を反転しているので、ブロック `a` の最後の suffix とブロック `a+1` の最初の suffix は一致する。よって変わるのは第 1 座標だけで、その変化量は `+1` である。

以上より、列全体は `T(L)` の Gray code である。`QED`

### Consequence

この命題から、各 chunk について

- 先頭 1 点の完全な index vector
- 以後の各 step で「どの軸が変わったか」
- その変化量 `±1`

だけ持てば、全 local point を復元できる。これは dense な `(dimension, num_points)` index table よりずっと軽い表現である。

## Proposition 2: Full Coordinate Permutations Do Not By Themselves Enumerate The Grid

### Claim

suffix 軸数を `s` とし、suffix の長さ profile を

`M = (M_1, ..., M_s)`

とする。full symmetric group `S_s` は suffix 座標の並べ替えを与えるが、これだけでは `T(M)` の全点列挙にはならない。さらに、`S_s` の各元が `T(M)` の自己同型になるのは `M_1 = ... = M_s` のときだけである。一般には、自己同型として使える軸 permutation は「等しい長さを持つ軸どうしだけを入れ替える」部分群に限られる。

### Proof

まず、軸 permutation `sigma in S_s` の作用を

`sigma · (u_1, ..., u_s) = (u_{sigma^{-1}(1)}, ..., u_{sigma^{-1}(s)})`

とする。

これが `T(M)` を自分自身へ写すには、任意の `u in T(M)` について像の各成分が対応する範囲

`0 <= v_i <= M_i - 1`

に入らなければならない。

もし `M_i != M_{sigma(i)}` なら、たとえば `u_i = M_i - 1` かつ他成分 0 の点を取ると、その像の `sigma(i)` 成分は `M_i - 1` になる。`M_i > M_{sigma(i)}` の場合、これは許容範囲を越える。したがって `sigma` が自己同型であるための必要条件は

`M_i = M_{sigma(i)}` for all `i`

である。

逆にこの条件が成り立つなら、各成分の取りうる範囲は permutation 後も一致するので、`sigma` は `T(M)` 上の全単射になる。よって自己同型群として使える座標 permutation は

`G_M = prod_r S_{m_r}`

である。ここで `m_r` は長さ `r` を持つ軸の本数である。

特に `M_1 = ... = M_s` の isotropic suffix なら `G_M = S_s`、そうでなければ full `S_s` は使えない。`QED`

### Consequence

「完全な置換群で suffix 軸を回せば全部舐められる」という直感は、一般の anisotropic term では正しくない。full `S_s` が厳密に使えるのは、suffix 各軸の rule length が完全に一致している場合だけである。

## Proposition 3: What The Permutation Group Is Good For

Permutation group に役割がないわけではない。役割は「全点の生成」そのものではなく、**既に正しい Gray-path / tensor-path の上にある軸順序を変えること** にある。

具体的には:

- local point の完全列挙:
  Proposition 1 の mixed-radix Gray path が担当する。
- 軸順序の並べ替え:
  Proposition 2 の stabilizer subgroup が担当する。

したがって、実装上の正しい分離は

1. local tuple を Gray code で生成する
2. 必要なら、その各 tuple に stabilizer subgroup の元を作用させて suffix 軸順を並べ替える

である。

Permutation だけでは点の値 `u_i` が動かないので、`T(L)` 全点は生成できない。

## Batched Suffix Complication

後ろ `s` 軸をまとめて batched にすると、上の議論はさらに 1 段複雑になる。

### 1. 実装上の対象は suffix tensor block

batched suffix で実際に materialize したいのは

- suffix local indices
- suffix nodes
- suffix weight products

である。このとき suffix 軸 permutation を使うなら、

- `rule_offsets`
- `rule_lengths`
- `term_axis_strides`
- node/weight gather の軸順

をすべて同じ permutation で同時に動かさなければならない。

### 2. 有効な permutation は term ごとに変わりうる

weighted Smolyak では term ごとに `L_i` が違うので、suffix 側の長さ profile `M` も term ごとに変わる。したがって Proposition 2 の stabilizer subgroup `G_M` も term ごとに変わる。

つまり、「suffix 5 軸に full `S_5` を 1 つ固定して全 term に使う」という設計は一般には正しくない。

### 3. 安全な設計

安全なのは次の 2 通りである。

- 設計 A:
  permutation を使わず、各 term の suffix をそのまま mixed-radix Gray で生成する。
- 設計 B:
  term ごとに suffix 軸を length profile で標準形へ並べ替え、その permutation を metadata として保持する。このとき使える群作用は、その標準形の stabilizer subgroup に限る。

今の高次元 weighted case では、まず A が安全で、その上で「等長 block の内部だけ」B を加えるのが自然である。

## Implementation Implications For This Repository

現在の codebase では、次の順で進めるのが妥当である。

1. `lazy-indexed` の次として、mixed-radix Gray path に基づく `transition-indexed` / `gray-indexed` mode を入れる。
   持つべき情報は
   - chunk 先頭の full rule-index vector
   - chunk 内の changed-axis 列
   - chunk 内の `±1` delta 列
   でよい。
2. `max_vectorized_suffix_ndim` を `1..5` で sweep して、50D weighted case でどの suffix 幅が最も throughput / memory / Pstate を改善するかを測る。
3. suffix permutation を使うなら、最初は「suffix 長が等しい軸だけ」を対象にする。
4. full `S_s` をそのまま使うのは、suffix が isotropic な term に限定する。

## What Is Actually Proven Here

このノートで証明したのは次のことだけである。

- reflected mixed-radix Gray code は local tensor grid を重複なく全部列挙する
- coordinate permutation の full symmetric group は一般の anisotropic suffix では grid の自己同型にならない

したがって、「完全な置換群だけで Smolyak term の全点を舐める」方針は一般 case では成立しない。成立するのは、

- permutation を **軸の並べ替え** として使う
- 点の全列挙そのものは Gray-path が担う

という分担のもとである。

## Immediate Next Step

実装タスクとしては、

- `gray-indexed` mode の追加
- `max_vectorized_suffix_ndim in {1,2,3,4,5}` sweep
- 50D weighted frontier の再計測

が次の自然な 1 手である。
