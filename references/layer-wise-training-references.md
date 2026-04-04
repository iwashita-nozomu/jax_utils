# レイヤー別訓練（Layer-wise Training）の参考文献

## 概要

本ドキュメントは、`neuralnetwork` モジュールの設計基盤となる**層疎結合（layer-wise decoupled）** アプローチの理論的根拠を示す参考文献をまとめたものです。

実装は `python/jax_util/neuralnetwork/sequential_train.py` に、設計方針は `notes/themes/desigin.md` に記載されています。

---

## 主要参考文献

### [1] 層別訓練の創始：GMDH (1965)

**文献ファイル**: [`1965-Ivakhnenko-Lapa-Cybernetics-Forecasting-Techniques.pdf`](./1965-Ivakhnenko-Lapa-Cybernetics-Forecasting-Techniques.pdf)
*(注: ファイルが利用可能な場合、上記リンクから参照可能)*

**文献情報**
- 著者: Alexey G. Ivakhnenko, Valentin G. Lapa
- 著作: *"Cybernetics and Forecasting Techniques"*
- 出版: American Elsevier Publishing Co.
- 出版年: 1965 (revised 1967)
- 関連論文: Ivakhnenko (1970). "Heuristic self-organization in problems of engineering cybernetics". *Automatica*, 6(2), 207-219.

**重要な発見**
- 世界初の動作する深層学習アルゴリズム「Group Method of Data Handling (GMDH)」を提案
- **1971年の実験** で8層のニューラルネットワークが層ごと訓練（layer-by-layer training via polynomial regression）で適切に学習できることを実証
- 層ごと訓練が数学的に機能することを最初に証明

**数学的貢献**
- 層ごと多項式回帰を用いた逐次的なパラメータ最適化
- 不要なニューロンを削減するバリデーションセット活用

**注記**: この手法は計算能力不足から普及しませんでしたが、理論的には最初の成功事例です。

---

### [2] 深層信念ネットワークの事前訓練 (2006)

**文献ファイル**: [`2006-Hinton-Osindero-Teh-Deep-Belief-Nets.pdf`](./2006-Hinton-Osindero-Teh-Deep-Belief-Nets.pdf)
*(注: ファイルが利用可能な場合、上記リンクから参照可能)*

**文献情報**
- 著者: Geoffrey E. Hinton, Simon Osindero, Yee-Whye Teh
- 論文: "A Fast Learning Algorithm for Deep Belief Nets"
- 掲載誌: *Neural Computation*, Vol. 18, No. 7, pp. 1527–1554
- 出版年: 2006年7月
- DOI: https://doi.org/10.1162/neco.2006.18.7.1527

**重要な発見**
- Restricted Boltzmann Machine (RBM) を積重ねた教師なし事前訓練アプローチ
- 各層をfreeze（固定）して上層を訓練する **sequential freezing strategy** を導入
- **重要**: 層ごと訓練は収束が遅いが、複雑な確率分布を学習可能であることを実証

**アルゴリズム**概要
```
For each layer l = 1 to L:
  1. Freeze weights θ₁, ..., θ₍ₗ₋₁₎
  2. Train RBM layer l on hidden representations from layer l-1
  3. Use learned representations as input to layer l+1
```

**実用的インパクト**
- ImageNetで2012年のAlexNet成功に先行する深層学習の実全例
- 事前訓練なしでは学習不可能だった深さ（当時）を達成

---

### [3] 貪欲層別訓練の収束解析 (2007) ⭐ **最重要**

**文献ファイル**: [`2007-Bengio-Lamblin-Popovici-Larochelle-Greedy-Layer-wise-Training.pdf`](./2007-Bengio-Lamblin-Popovici-Larochelle-Greedy-Layer-wise-Training.pdf)
*(注: ファイルが利用可能な場合、上記リンクから参照可能)*

**文献情報**
- 著者: Yoshua Bengio, Pascal Lamblin, Dan Popovici, Hugo Larochelle
- 論文: "Greedy layer-wise training of deep networks"
- 掲載: *Advances in Neural Information Processing Systems (NIPS)*, pp. 153–160
- 出版年: 2007
- PDF: http://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks

**重要な発見**
- **正式な定式化**: 貪欲層別訓練（Greedy Layer-wise Training, GLWD）の数学的定義
- **収束解析**: 層ごと最適化が全体の逆伝播より収束が速く、局所解に陥る傾向が低いことを理論的に示した
- **実証**: 複数のデータセットでGLWDが標準的なバックプロパゲーション初期化より優れていることを確認

**理論的貢献**
層別訓練の目的関数：
$$
\mathcal{L}_\ell = \mathbb{E}_{x} [L(h_\ell(h_{\ell-1}(\cdots h_1(x))))]
$$

ここで $h_\ell = f_\ell(W_\ell h_{\ell-1} + b_\ell)$ であり、各層は独立に訓練されます：

$$
\theta_\ell^* = \arg\min_{\theta_\ell} \mathbb{E}[L(\phi(\theta_\ell, a_\ell^-))]
$$

（$a_\ell^-$ は前層の固定された活性化）

**本プロジェクトへの直接関連性**
- sequential_train.py の GradientTrainer が実装する層ごと更新の理論的正当性をもつ

---

### [4] 並列化と分散学習の通信効率分析 (2018)

**文献ファイル**: [`2018-Ben-Nun-Hoefler-Parallel-Distributed-Deep-Learning.pdf`](./2018-Ben-Nun-Hoefler-Parallel-Distributed-Deep-Learning.pdf)
*(注: arXiv PDF から入手可能: https://arxiv.org/pdf/1802.09941.pdf)*

**文献情報**
- 著者: Tal Ben-Nun, Torsten Hoefler
- 論文: "Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis"
- 出版年: 2018年2月（v1）, 2018年9月更新（v2）
- arXiv: https://arxiv.org/abs/1802.09941
- DOI: https://doi.org/10.48550/arXiv.1802.09941

**調査対象**
- DNN訓練の異なる並列化戦略を6,000行以上の文献レビューで分類分析
- 単一オペレータ、ネットワーク推論、分散訓練まで多層的に考察

**重要な洞察（本設計に関連する部分）**
- **層ごと訓練（layer-wise update）は通信効率が良い**
  - 勾配計算を層単位で独立させられる
  - ノード間通信がバッチ単位で集約可能

- **データ並列vs モデル並列**
  - 層別訓練はデータ並列に親和的
  - 勾配集約のオーバーヘッドが削減可能

**実装最適化**
- JAX/equinoxの分散学習において層単位の独立実行が通信ボトルネック削減に寄与

---

### [5] ブロック座標降下法の理論 (2011)

**文献ファイル**: [`2011-Boyd-Parikh-ADMM-Foundations-Trends.pdf`](./2011-Boyd-Parikh-ADMM-Foundations-Trends.pdf)
*(注: ファイルが利用可能な場合、上記リンクから参照可能。出版社サイト: https://www.nowpublishers.com/article/Details/MAL-016)*

**文献情報**
- 著者: Stephen Boyd, Neal Parikh
- 著作: *"Distributed Optimization and Statistical Learning via the ADMM"*
- 掲載誌: *Foundations and Trends in Machine Learning*, Vol. 3, No. 1, pp. 1–122
- 出版年: 2011
- URL: https://www.nowpublishers.com/article/Details/MAL-016

**理論的適用性**
層別訓練はブロック座標降下法（BCD）の特殊例として解釈可能：

パラメータ空間をブロック分割：
$$
\Theta = \{\Theta_1, \Theta_2, \ldots, \Theta_L\} \quad (\text{各層に対応})
$$

各反復で1つのブロックを最適化：
$$
\Theta_\ell^{(t+1)} = \arg\min_{\Theta_\ell} \mathcal{L}(\Theta_1^{(t+1)}, \ldots, \Theta_{\ell-1}^{(t+1)}, \Theta_\ell, \Theta_{\ell+1}^{(t)}, \ldots)
$$

**収束保証**
- **強凸目的関数**: 線形収束速度 $O(1/k)$
- **非凸設定** (ニューラルネット): 実用上、安定な収束パターンを示す

---

## 実装との対応

| 論文 | 実装箇所 | 適用関連性 |
|-----|--------|----------|
| Ivakhnenko (1965) | - | 層別訓練概念の原点 |
| Hinton et al. (2006) | - | sequential freezing戦略の参考 |
| **Bengio et al. (2007)** | `sequential_train.py: GradientTrainer` | 🔴 **直接的根拠** |
| Ben-Nun & Hoefler (2018) | 将来の分散実装 | 並列化設計指針 |
| Boyd & Parikh (2011) | 最適化理論基盤 | 収束性証明 |

---

## 設計方針との関連

`notes/themes/desigin.md` に記載されている **2つのアプローチの比較**：

| 項目 | 層別訓練（Approach 2）✓選択 | Mask-based全体最適化（Approach 1） |
|-----|---------------------------|------------------------------|
| 理論的根拠 | Bengio et al. (2007) | 標準逆伝播（1986年以降） |
| 初期提案 | GMDH (1965), DBN (2006) | Backprop Era |
| 計算効率 | 層単位デコード（Ben-Nun 2018） | 全体フローによるオーバーヘッド |
| JAX親和性 | 非常に高い（lax.scan適合） | 中程度 |

---

## 入手方法と文献ファイル

### ローカル保存されているファイル

本ドキュメントは以下のファイルを `references/` ディレクトリに参照しています：

| 論文 | ファイル名 | ステータス |
|-----|-----------|----------|
| [1] Ivakhnenko & Lapa (1965) | [1965-Ivakhnenko-...](./1965-Ivakhnenko-Lapa-Cybernetics-Forecasting-Techniques.pdf) | ⏳ 入手待ち |
| [2] Hinton et al. (2006) | [2006-Hinton-...](./2006-Hinton-Osindero-Teh-Deep-Belief-Nets.pdf) | ⏳ 入手待ち |
| [3] Bengio et al. (2007) ⭐ | [2007-Bengio-...](./2007-Bengio-Lamblin-Popovici-Larochelle-Greedy-Layer-wise-Training.pdf) | ⏳ 入手待ち |
| [4] Ben-Nun & Hoefler (2018) | [2018-Ben-Nun-...](./2018-Ben-Nun-Hoefler-Parallel-Distributed-Deep-Learning.pdf) | ⏳ 入手待ち |
| [5] Boyd & Parikh (2011) | [2011-Boyd-Parikh-...](./2011-Boyd-Parikh-ADMM-Foundations-Trends.pdf) | ⏳ 入手待ち |

### オンラインでの入手方法

各論文が手元に無い場合、以下の方法で入手可能です：

#### [2] Hinton et al. (2006): Neural Computation誌（有料）
- 出版社: MIT Press
- 多くの大学図書館で購読中
- **代替**: arXiv版が無い場合、ResearchGate で検索

#### [3] Bengio et al. (2007): NIPS Proceedings（**無料**）
- ✓ NIPS Official Archive: http://papers.nips.cc/paper/3048-greedy-layer-wise-training-of-deep-networks
- ✓ PDF直接DL可能

#### [4] Ben-Nun & Hoefler (2018): arXiv（**無料**）
- ✓ PDF直接DL: https://arxiv.org/pdf/1802.09941.pdf
- ✓ トラッキング情報: https://arxiv.org/abs/1802.09941

#### [5] Boyd & Parikh (2011): 出版社（有償、ドラフト版公開）
- 公式: https://www.nowpublishers.com/article/Details/MAL-016
- **代替**: ドラフト版をGoogle Scholarで検索

#### [1] Ivakhnenko & Lapa (1965): 古文献（入手困難）
- 出版社絶版
- **参照方法**: Schmidhuber (2015)の"Annotated History"で詳しく解説
  - Schmidhuber, J. (2015). "Deep Learning in Neural Networks: An Overview". *Neural Networks*, 61(1), 85-117.
  - https://arxiv.org/abs/1404.7828 _(無料、Section 5.1参照)_

---

## ファイル管理手順

### ローカル保存ファイルの追加方法

本ドキュメントで参照するPDFファイルを `references/` ディレクトリに保存する場合、以下のファイル命名規則に従ってください：

#### ファイル命名規則
```
YYYY-FirstAuthor-...-Title.pdf
```

**例**：
- `2007-Bengio-Lamblin-Popovici-Larochelle-Greedy-Layer-wise-Training.pdf`
- `2006-Hinton-Osindero-Teh-Deep-Belief-Nets.pdf`

#### ファイルサイズ目安
- 単一論文：1-5 MB
- スキャン資料：10-50 MB

#### 保存場所
```
references/
├── layer-wise-training-references.md    ← 本ドキュメント
├── 2007-Bengio-Lamblin-...pdf           ← 新規保存するファイル
├── 2006-Hinton-Osindero-...pdf          ← 新規保存するファイル
└── sparse_grid/                         ← その他研究領域
```

### ドキュメント内でのリンク貼り方

相対パスでリンクしてください：

```markdown
**文献ファイル**: [`2007-Bengio-...pdf`](./2007-Bengio-Lamblin-Popovici-Larochelle-Greedy-Layer-wise-Training.pdf)
```

**結果表示**:
> **文献ファイル**: [`2007-Bengio-...pdf`](./2007-Bengio-Lamblin-Popovici-Larochelle-Greedy-Layer-wise-Training.pdf)

### チェックリスト

ファイル追加時の確認項目：

- [ ] ファイル名が規則に従っている
- [ ] 相対パスが正しい（`./` で始まる）
- [ ] ファイルサイズが合理的（< 100 MB）
- [ ] 複数著者の場合、first 3 authors を記載
- [ ] 表のステータスを「⏳ 入手待ち」から「✓ 保存済み」に更新

---

## 参考文献（セカンダリ）

本ドキュメントが参照する他の主要文献：

- **Schmidhuber, J. (2015)**. "Deep Learning in Neural Networks: An Overview". *Neural Networks*, 61(1), 85-117.
  - https://arxiv.org/abs/1404.7828
  - *Ivakhnenko (1965)の詳細な歴史的解説を含む*

- **LeCun, Y., Bengio, Y., & Hinton, G. (2015)**. "Deep Learning". *Nature*, 521, 436–444.
  - https://doi.org/10.1038/nature14539
  - *深層学習全体の総括文献*

1. **desigin.md** を読んで設計方針を理解（5分）
2. **Bengio et al. (2007)** を通読（15-20分、アルゴリズムボックス中心）
3. **sequential_train.py** のコード確認、Bengio論文との対応を確認（10分）
4. 必要に応じて **Hinton et al. (2006)** または **Boyd & Parikh (2011)** で理論を深掘り

---

## 今後の拡張

このドキュメントは以下の論文が参照リストに加わる可能性があります：

- **強化学習との関連**: Policy gradient が層別学習と互換性を持つか
- **分散訓練**: 層別訓練の分散設定での収束性
- **転移学習**: 事前訓練（pretraining）との関連性
- **動的ネットワーク**: 層数がタスク中に変化する設定

---

*最終更新: 2026-03-19*
*担当: Project NN Architecture Design*
