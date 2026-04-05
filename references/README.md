# References

実装や実験で参照した論文・書籍・資料をこのディレクトリに置きます。

## ディレクトリ・ファイル構成

### レイヤー別訓練の理論基盤

**`layer-wise-training-references.md`**

- **管理者**: Project NN Architecture Design
- **対象モジュール**: `python/jax_util/neuralnetwork/sequential_train.py`
- **設計ドキュメント**: `notes/themes/desigin.md`
- **内容**:
  - 層別訓練の歴史的背景（GMDH 1965 → DBN 2006 → Bengio 2007）
  - 5つの主要論文の詳細分析
  - 実装との対応関係
  - 入手方法と推奨読了順序

**主要参考文献**:

1. Ivakhnenko & Lapa (1965) - GMDH の創始
1. Hinton et al. (2006) - 深層信念ネット
1. **Bengio et al. (2007)** - 貪欲層別訓練の収束解析 ⭐
1. Ben-Nun & Hoefler (2018) - 並列化分析
1. Boyd & Parikh (2011) - ブロック座標降下法

### Bellman 型逆伝播と双対・計量

**`bellman-backprop-references.md`**

- **管理者**: Bellman / Backprop note
- **対象ノート**: `notes/themes/bellman_bp.md`
- **内容**:
  - reverse-mode AD を dual / cotangent の pullback として読むための一次資料
  - Riesz 写像と質量行列が必要になる段階の整理
  - 関数空間の計量をパラメータ空間へ引き戻した Gram / mass matrix の解釈
  - Bellman 型の記法に引き寄せた短い要約

**主要参考文献**:

1. Baydin et al. (2018) - reverse-mode AD survey
1. Frostig et al. (2021) - linearization + transposition
1. Schwedes, Funke, Ham (2016) - Riesz 表現と mesh dependence
1. dolfin-adjoint docs - dual と Riesz representation
1. van Erp et al. (2022) - 引き戻し計量の Gram 行列

### `lagrangian_df_zero/`

- **overview**: `lagrangian_df_zero/README.md`
- **detailed catalog**: `lagrangian_df_zero/catalog.md`
- **対象ノート**: `notes/themes/06_lagrangian_df_zero.md`
- **対象設計**: `documents/design/jax_util/lagrangian_df_zero.md`
- **内容**:
  - $DF=0$ 型の layer update を調べるための一次資料パック
  - overview と detailed catalog を分けた索引
  - exact equality-constraint、lifted baseline、implicit differentiation の research line 整理
  - local PDF、landing page、pending review artifact の status 整理

### その他の研究領域

- `sparse_grid/`
  - Smolyak 積分、sparse-grid combination technique、adaptive sparse grid に関する一次資料です。

## アクセスガイド

- **desigin.md から**: `notes/themes/desigin.md` → `参考文献` セクション → [参考文献詳細へ](layer-wise-training-references.md)
- **sequential_train.py の背景理論を知りたい**: [Bengio et al. (2007)の詳細解説](layer-wise-training-references.md#3-%E8%B2%AA%E6%AC%B2%E5%B1%A4%E5%88%A5%E8%A8%93%E7%B7%B4%E3%81%AE%E5%8F%8E%E6%9D%9F%E8%A7%A3%E6%9E%90-2007-%E2%AD%90-%E6%9C%80%E9%87%8D%E8%A6%81)
- **層別訓練の歴史を学ぶ**: [層別訓練の歴史セクション](layer-wise-training-references.md#%E5%B1%A4%E5%88%A5%E8%A8%93%E7%B7%B4%E3%81%AE%E6%AD%B4%E5%8F%B2)
- **Bellman 型逆伝播の双対・質量行列を確認したい**: [Bellman 型逆伝播と双対・計量の参考文献](bellman-backprop-references.md)
- **$DF=0$ 型の更新則の一次資料を見たい**: [lagrangian_df_zero reference pack](lagrangian_df_zero/README.md)
- **`lagrangian_df_zero` の artifact ごとの役割を見たい**: [lagrangian_df_zero catalog](lagrangian_df_zero/catalog.md)
