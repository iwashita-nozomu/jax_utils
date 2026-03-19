# Legacy Smolyak Results

## 0. このメモの目的

このメモは、

旧版 Smolyak 実装の大規模実験結果を、

`main` から参照可能な形で残すためのものである。

結果ファイル本体は results branch に置き、

`main` には

- branch 名
- 結果の位置づけ
- 定性的な考察
- 旧実装から得た教訓

を残す。

ここでいう「旧版」は、

explicit な grid / weight 保持を強く前提にした Smolyak 実装を指す。

改訂版は別 branch で継続している。

## 1. 関連 branch

### 1.1 旧版結果 branch

- `results/functional-smolyak-scaling`

### 1.2 改訂版の継続実験 branch

- `results/functional-smolyak-scaling-tuned`

この二つは、

単に commit の前後ではなく、

設計思想が少し違う。

旧版は

「grid を構成して重み付き和を取る」

色が強い。

改訂版は

「grid の明示展開を避け、積分器の評価構造を軽くする」

方向へ寄っている。

## 2. 旧版で残している主要結果

対象として重要なのは次の二系統である。

### 2.1 完了済み GPU run

- `smolyak_scaling_gpu_20260315T140215Z.json`
- 対応 report ディレクトリ

この run は、

比較的小さいレンジではあるが、

複数 dtype を最後まで回している。

したがって、

精度比較の基準として価値が高い。

### 2.2 長時間 run の partial 集計

- `smolyak_scaling_gpu_20260316T061620Z.jsonl`
- `run_smolyak_scaling_20260316T061619Z.log`
- `smolyak_scaling_gpu_20260316T061620Z_partial.json`
- partial report ディレクトリ

こちらは途中停止している。

しかし、

JSONL の逐次保存があるため、

停止時点までの frontier と failure の傾向を読める。

この partial run は、

精度比較というより、

旧版の運用限界とメモリ挙動を見る材料として重要である。

## 3. 積分問題の整理

旧版の実験では、

主に解析解が分かる積分問題で精度を測っている。

基本領域は

$$
\Omega_d = [-1/2, 1/2]^d
$$

で、

体積は

$$
|\Omega_d| = 1
$$

である。

この正規化はかなり重要で、

Monte Carlo と Smolyak の両方で

余分な体積係数を別に持たずに済む。

## 4. 解析解を持つ基準問題

### 4.1 指数関数型

係数ベクトル $a \in \mathbb{R}^d$ に対し

$$
f_a(x) = \exp(a^\top x)
$$

を考える。

このとき、

積分は各軸に分離し、

$$
\int_{\Omega_d} \exp(a^\top x)\,dx
=
\prod_{k=1}^d
\int_{-1/2}^{1/2} e^{a_k t}\,dt
$$

となる。

したがって

$$
\int_{\Omega_d} \exp(a^\top x)\,dx
=
\prod_{k=1}^d
\frac{2\sinh(a_k/2)}{a_k}
$$

である。

ただし $a_k = 0$ の極限では

$$
\frac{2\sinh(a_k/2)}{a_k} \to 1
$$

である。

この問題は、

- 各次元に分離できる
- 係数を連続的に変えられる
- 高次元化しやすい

ので、

大規模実験の基準問題として扱いやすい。

### 4.2 ガウス型

非線形性を見るために、

$$
g_\alpha(x) = \exp(-\alpha \|x\|_2^2)
$$

のような問題も考える。

これも分離可能で、

$$
\int_{\Omega_d} g_\alpha(x)\,dx
=
\left(
\int_{-1/2}^{1/2} e^{-\alpha t^2}\,dt
\right)^d
$$

となる。

1 次元積分は誤差関数を使って

$$
\int_{-1/2}^{1/2} e^{-\alpha t^2}\,dt
=
\sqrt{\frac{\pi}{\alpha}}
\operatorname{erf}\!\left(\frac{\sqrt{\alpha}}{2}\right)
$$

である。

したがって

$$
\int_{\Omega_d} g_\alpha(x)\,dx
=
\left(
\sqrt{\frac{\pi}{\alpha}}
\operatorname{erf}\!\left(\frac{\sqrt{\alpha}}{2}\right)
\right)^d
$$

となる。

旧版ではこの種の問題により、

単なる多項式正確性ではなく、

非線形関数に対する実用的な挙動も見ていた。

## 5. 旧版 Smolyak 実装の概念図

旧版の概念をかなり粗く書くと、

1. 1D rule 列を作る
1. multi-index を列挙する
1. 各 term の tensor product point を作る
1. 重複点を統合する
1. global points / weights を作る
1. 最後に重み付き和を取る

という流れである。

数式上は、

Smolyak 近似はたとえば

$$
A(q,d)
=
\sum_{|i|_1 \le q + d - 1}
\Delta_{i_1} \otimes \cdots \otimes \Delta_{i_d}
$$

と書ける。

問題は、

この数式上の和を、

実装上

「一度すべての点に展開してから整理する」

方向で書いていたことにある。

この設計は、

小さい問題では分かりやすい。

しかし高次元化すると、

host memory に強く効いてしまう。

## 6. 旧版で観測した主な現象

旧版の結果から見えたことを、

ここでは分解して残す。

### 6.1 `float64` は安定

完了済みの旧版 run では、

`float64` がもっとも安定だった。

level を上げたときに、

少なくとも観測したレンジでは、

誤差の低下が自然に見えた。

このことは重要である。

なぜなら、

Smolyak の構成そのものが壊れているなら、

`float64` でも改善しないはずだからである。

したがって、

旧版の主問題は

- 近似理論の破綻

ではなく、

- 実装上のメモリ構造
- 低精度における数値相殺

であると解釈できる。

### 6.2 `float16`, `bfloat16` は高次元・高 level で急激に悪化

`float16` と `bfloat16` は、

低い level ではそこまで悪く見えなくても、

level や dimension を上げると急速に苦しくなる。

その理由は、

Smolyak が差分則の和だからである。

差分則では正負の重みが混ざる。

したがって、

最終的な積分値が moderate でも、

途中の絶対値和

$$
\sum_{k=1}^n |a_k|
$$

はかなり大きくなりうる。

逐次加算では、

$$
\hat s_n = (((a_1+a_2)+a_3)+\cdots)+a_n
$$

であり、

真の和 $s_n$ が cancellation により小さくなると、

相対誤差が一気に悪化する。

旧版ではまさにこの現象が見えている。

### 6.3 `float32` は中間

`float32` は `float16` / `bfloat16` よりかなり良い。

しかし、

大きなケースになると、

やはり誤差悪化が出る。

したがって、

旧版においては

`float32` も

「十分に高次元・高 level なら安全ではない」

という評価になる。

## 7. なぜ低精度が苦しいのか

ここは少し丁寧に書いておく。

単純な重み付き和

$$
S = \sum_{k=1}^n w_k f(x_k)
$$

だけを見ると、

低精度でも問題なく見えることがある。

しかし Smolyak では、

実際には

$$
S =
\sum_{i \in \mathcal{I}}
c_i
\sum_{j \in J_i}
w_{i,j} f(x_{i,j})
$$

のように、

term ごとの和のさらに外側に

係数付き和がある。

しかも

- $c_i$ の符号が混ざる
- $w_{i,j}$ の符号も混ざる

となる。

このとき問題になるのは、

小さい $u$ だけではない。

むしろ、

$$
\kappa_{\mathrm{sum}}
=
\frac{\sum |a_k|}{|\sum a_k|}
$$

に近い量が大きくなることが本質である。

Smolyak の低精度実験は、

この条件数の悪化をかなり素直に映している。

## 8. 旧版の host memory 問題

旧版のもう一つの核心は、

host memory が支配的だったことである。

格子構築時のメモリ量をかなり粗く見積もると、

$$
M
\approx
N_{\mathrm{pts}} d b_x
+
N_{\mathrm{pts}} b_w
+
M_{\mathrm{tmp}}
$$

である。

ここで一番厄介なのは

$$
M_{\mathrm{tmp}}
$$

である。

`meshgrid`

`reshape`

`concatenate`

`unique`

`inverse indices`

などの一時領域が、

本体より重くなる。

旧版の大規模 run では、

実際に

- RSS の急増
- swap の利用
- `host_oom`
- `worker_terminated`

が見えた。

この時点で、

ボトルネックは GPU の FLOP ではなく、

host 側の grid materialization だと見てよい。

## 9. GPU 利用率が低い理由

旧版の議論で、

「GPU がフルに使われていない」

という問題意識があった。

この点は旧版結果を読むうえで重要である。

直感的には、

積分計算なのだから GPU が忙しそうに見えるはずだ、

と思いやすい。

しかし、

旧版で実際に重いのは

- point cloud 構築
- 重複統合
- host-device 転送前の準備

である。

そのため、

GPU 上の最終的な

`vmap(f)` や `tensordot`

より前に時間が吸われる。

この構造では、

GPU 使用率が低く見えるのは自然である。

## 10. 途中停止 run の意味

途中停止 run は、

完成 run よりむしろ多くのことを教えてくれた。

完了した benchmark はもちろん大事だが、

途中で壊れた run は

実装の限界を直接示す。

今回の partial run では、

failure kind として

- `error`
- `worker_terminated`
- `host_oom`
- `oom`

が見えた。

この分類を残せたのは大きい。

とくに

`host_oom`

と

`worker_terminated`

が多いことは、

問題が単なる関数評価の重さではないことを示している。

## 11. partial run に `float16` しか無い理由

これは誤解を招きやすいので、

丁寧に残しておく。

partial run に

`float16`

しか残っていないのは、

renderer の抽出漏れではない。

また、

`float16` だけが特別に選ばれたわけでもない。

旧 runner のケース順が

$$
\text{dtype} \to \text{level} \to \text{dimension}
$$

だったため、

run が途中で止まると、

最初の dtype ブロックしか残らない。

この設計は実験としてよくない。

途中停止しても各 dtype を比較できるよう、

本来は

$$
\text{level} \to \text{dimension} \to \text{dtype}
$$

あるいは

$$
\text{dimension} \to \text{level} \to \text{dtype}
$$

のように interleave すべきだった。

この教訓は改訂版で重要である。

## 12. partial frontier から読めること

partial run から読む限り、

旧版 `float16` の frontier はおおむね

- level 1..5: dimension 32
- level 6: dimension 28
- level 7: dimension 18
- level 8: dimension 13

付近だった。

これは、

厳密な「理論限界」ではない。

あくまで、

旧版実装、

旧 runner の順序、

当時の resource 条件、

当時の failure policy

の下で観測された frontier である。

しかし傾向としては重要で、

高 level 側で急速に成功領域が縮むことを示している。

## 13. 旧版結果の価値

旧版結果は、

古い実装だから価値がない、

というものではない。

むしろ価値は三つある。

### 13.1 `float64` では Smolyak 自体が成立していることの確認

これは土台になる。

### 13.2 低精度で何が壊れるかの観測

実装改善の方向を決める材料になる。

### 13.3 host memory 問題の可視化

GPU 最適化より先に何を直すべきかを教えてくれる。

つまり旧版結果は、

旧版の墓標ではなく、

改訂版の設計資料である。

## 14. 旧版から直接つながる改善方針

旧版結果から直結する改善方針は次の通りである。

### 14.1 explicit grid を避ける

全点 materialize を減らす。

### 14.2 term ごとの逐次和へ寄せる

Smolyak の定義に近づける。

### 14.3 低精度では加算順に気を配る

少なくとも評価設計を見直す。

### 14.4 実験順序を interleave する

途中停止しても比較できるようにする。

### 14.5 JSONL を標準化する

停止しても partial recovery できるようにする。

## 15. 文献調査との接続

今回収集した文献とも、

旧版結果はよく整合する。

文献の流れを大雑把に言えば、

- hash / tree based storage
- compact index representation
- combination technique
- adaptive frontier

の方向が主流である。

それに対して旧版は、

実装の見通しは良いが、

大規模運用では重い explicit grid 側に寄っていた。

だからこそ、

文献で自然に感じられる改善方針が、

今回の実験結果とも噛み合って見える。

## 16. 旧版 report をどう読むべきか

旧版 report を見るときには、

単に色の濃淡だけを見るのではなく、

次の 4 つを同時に見るのがよい。

### 16.1 `mean_abs_err`

数値精度の表面。

### 16.2 `avg_integral_seconds`

実行時間の表面。

### 16.3 `storage_bytes`

保持量の表面。

### 16.4 `frontier`

成功／失敗境界の表面。

これらを重ねると、

「どこから精度が崩れたか」

だけでなく、

「どこから host 側の重さが支配的になったか」

も見えてくる。

## 17. なぜ `main` にはメモだけ残すのか

この点も意図を残しておく。

旧版結果の JSON、HTML、SVG、ログは、

参照価値は高い。

しかし `main` に常設すると、

ノイズも大きい。

したがって、

結果本体は results branch に置き、

`main` には

- どの branch を見ればよいか
- 何が読めるか
- どう解釈すべきか

だけを残す方針を取っている。

この分離はかなり良いと思う。

## 18. このメモの役割

このメモは、

単なる「結果がありました」という告知ではない。

役割は次の三つである。

1. 旧版結果の存在を `main` から辿れるようにする
1. 旧版結果から何を学んだかを残す
1. 改訂版の設計判断の背景を残す

とくに三つ目が重要である。

実験結果は、

次の設計を支えるときに初めて生きる。

旧版結果の意味は、

改訂版の方針を支えるところにある。

## 19. 今後このメモに追記するなら

今後このメモに追記するなら、

次のような情報が有用だと思う。

- 旧版と改訂版の定量比較
- 同じ条件での throughput 比較
- host RSS の差
- HLO の構造差
- 低精度での誤差傾向の違い

つまり、

このメモは完結済みの報告書というより、

旧版結果を基準線として使うためのハブである。

## 20. 要約

最後に短くまとめる。

- 旧版 `float64` は安定していた
- 低精度では cancellation が強く出た
- 旧版の主ボトルネックは GPU ではなく host 側格子構築だった
- partial run が `float16` しか含まないのは case ordering の設計による
- 旧版結果は改訂版の設計資料として有用である

この五点が、

旧版 Smolyak 実験から得た主要な教訓である。
