# 4. Progressive Neural Networks と Meta-Learning：層別訓練の近隣技術

**作成日**： 2026-03-19
**対象**： 層別訓練に関連する最近の手法と理論
**関連文献**： Rusu et al. (2016)、Schmidhuber (2018)、Vezhnevets et al. (2016)

---

## 概要

層別訓練は、modern deep learning では使用頻度が低いが、**転移学習** や **メタ学習** など関連する強力な手法が登場している。本章では、これらの方法と層別訓練の理論的つながりを分析する。

---

## 1. Progressive Neural Networks (PNNs)

### 1.1 基本的な考え方

Progressive Neural Networks は、**段階的にネットワークを拡張** する手法である（Rusu et al., 2016）。

**従来の手法との違い**：
- 層別訓練： 固定されたアーキテクチャの層を順序的に最適化
- PNN： 新しいタスク追加時に **新しい列（column）** を追加し、既存パラメータを保持

### 1.2 PNN のアーキテクチャ

```
タスク 1:  [NN_1] → [出力 1]
タスク 2:  [NN_1] ←→ [NN_2] → [出力 2]
タスク 3:  [NN_1] ← [NN_2] ←→ [NN_3] → [出力 3]

注： ←→ は lateral connection (側方接続)
```

**層構成**：
- **列 (columns)**： 各タスク専用の層スタック
- **側方接続 (lateral connections)**： 前号タスクの出力を新しい列に入力

### 1.3 数学的定式化

タスク $t$ のニューロン層 $l$：
$$h_l^t = f\left( W_l^t h_{l-1}^t + \sum_{t' < t} U_{l}^{t \leftarrow t'} h_l^{t'} \right)$$

ここで：
- $W_l^t$： タスク $t$ 専用の重み
- $U_l^{t \leftarrow t'}$： タスク $t'$ からの側方接続

**学習**：
既存列は **frozen** （$W_l^{t'}, t' < t$ は固定）、新たに追加された列のみ学習。

### 1.4 層別訓練との関連

**共通点**：
- 層/タスク間での **parameter freezing**
- 段階的な最適化

**相違点**：
- 層別訓練： 同一ネットワークの層を順序処理
- PNN： 異なるタスク間での重み共有と転移

---

## 2. 転移学習とファイン チューニング

### 2.1 Pre-training と Fine-tuning

```
段階 1：Pre-training (一般的なタスク)
        ↓ 多数の層を学習

段階 2：Fine-tuning (特定タスク)
        ↓ 最終層を再学習、または全層を小学習率で調整
```

**層別訓練的観点**：
各層が **一般的な特徴** から **タスク特定 特徴** へと段階的に調整。

### 2.2 数学的定式化

Pre-trained 重み： $W^{pre}$
Fine-tuned 重み： $W^{ft}$

微調整での目的関数：
$$L^{ft}(W^{ft}) = L^{task}(W^{ft}) + \lambda \| W^{ft} - W^{pre} \|_2^2$$

第2項は **regularization**: pre-trained 値からの距離を制限。

### 2.3 層別性の観点

各層の学習率を異なるものに設定することで、層別調整が可能：

$$W_i^{ft} = W_i^{pre} - \alpha_i \nabla_{W_i} L^{task}$$

ここで $\alpha_i$ は層 $i$ に対する学習率。

**推奨**：
- **下位層（早期層）**： $\alpha_i$ 小 (一般的な特徴を保持)
- **上位層（後期層）**： $\alpha_i$ 大 (タスク適応)

---

## 3. Model-Agnostic Meta-Learning (MAML)

### 3.1 MAML の基本概念

MAML (Finn et al., 2017) は、**少数のサンプルで素早く適応** できるモデルを学習する方法。

**中心的考え方**：

$$\text{Learn } \theta^* \text{ such that } \theta^* - \beta \nabla L(\theta^*) \text{ is good for new tasks}$$

つまり、**初期値** $\theta^*$ が多くのタスクに対して 1-2 ステップの勾配降下で適応可能。

### 3.2 MAML の学習ループ

```
Meta-training:
  for task_batch:
    for task in task_batch:
      # 1 ステップ適応
      θ_task = θ_meta - α ∇ L_task(θ_meta)

      # 外部ループで評価
      meta_loss += L_task(θ_task)

    # メータパラメータ更新
    θ_meta -= β ∇_θ_meta sum(meta_loss)
```

### 3.3 層別訓練との関連

**相違点**（重要）：
- MAML： **全パラメータを同時** に meta-update
- 層別訓練： 層を **順序的・固定的** に処理

**関連性**：
- 両者とも **段階的な最適化** を目指す
- MAML の内ループ（task-specific）は、層別訓練の局所最適化に類似

### 3.4 数学的比較

| 方法 | パラメータ更新 | 固定条件 |
|------|-------------|---------|
| 層別訓練 | $W_i \leftarrow W_i - \alpha \nabla L_i(W_i)$ | $W_1, \ldots, W_{i-1}$ 固定 |
| MAML | $\theta \leftarrow \theta - \beta \sum \nabla L_{task}(\theta - \alpha \nabla L_{task}(\theta))$ | なし（全パラメータ更新） |

---

## 4. Hierarchical Reinforcement Learning (HRL)

### 4.1 計画階層の構築

層別訓練は、教師あり学習での方法だが、**強化学習** における階層的学習にも類似構造がある。

**Macro-actions の学習** (Vezhnevets et al., 2016):

```
上位層：       〇 → 〇 → 〇  (macro-action policy)
          ↓         ↓         ↓
下位層：   ◆→◆→◆  ◆→◆→◆  ◆→◆→◆  (primitive actions)
```

### 4.2 STRAW (Strategic Attentive Writer)

Vezhnevets et al. (2016) の提案法：

**特徴**：
- 上位：**何を計画するか** (which macro-action)
- 下位：**どのように実行するか** (how to execute)

数学的には：
$$\pi(a | s) = \int_m p(m | s) \pi(a | s, m) dm$$

ここで $m$ = macro-action。

### 4.3 層別訓練との関連

**層別訓練的解釈**：
- 各 macro-action を **1つの層** として見なす
- 下位層（primitive action）を固定して、上位層(macro policy)を学習

---

## 5. Curriculum Learning

### 5.1 基本的考え方

**Curriculum Learning** (Bengio et al., 2009)：

簡単なタスクから難しいタスクへ段階的に学習。

```
難度:  ▁▂▃▄▅▆▇█
タスク: 簡 → 中 → 難
```

### 5.2 層別訓練との関連

**層別訓練の観点**：
- 各層を **難度増加の段階** と見なす
- 下位層（機能が単純）→ 上位層（複雑な抽象化）

**数学的関係**：
下位層が学習する特徴の **複雑度**：
$$C_i = \text{entropy of learned features in layer } i$$

通常、$C_1 < C_2 < \ldots < C_n$ （下位層ほど単純）

---

## 6. 多目標学習 (Multi-Task Learning)

### 6.1 共有層と特殊層

```
共有層 (Shared):  [Dense] → [Dense]
                     ↙        ↘
特殊層 (Task-specific):
                [Dense] → [出力 Task1]
                [Dense] → [出力 Task2]
```

### 6.2 層別的な学習戦略

各層での多目標損失：
$$L_{shared}(W_{shared}) = \sum_t w_t L_t(W_{shared}, W_t^{spec})$$

**層別優化**：
1. 共有層を固定
2. 特殊層を各タスク内で独立学習
3. 共有層を更新（複数タスク情報使用）

---

## 7. Backup Neural Networks (BNNs) / Capsule Networks

### 7.1 基本概念

Capsule Networks (Hinton et al., 2011, Sabour et al., 2017):

```
Neuron:  スカラー出力
Capsule: ベクトル出力（方向・大きさ）
```

$$u_j = \| W_{ij} u_i \|  \text{ (routing by agreement)}$$

### 7.2 層別訓練との関連

**段階的な抽象化**：
- 低層 capsule： 素朴な特徴（エッジなど）
- 高層 capsule： 複雑なエンティティ（顔など）

**段階的学習**：
各層の capsule を順序的に学習することで、hierarchical representation を構築。

---

## 8. 現代的な統合視点

### 8.1 Implicit Layer-wise Computation

Recent works (e.g., Neural ODE) は、implicit な層別計算を含む：

$$h_{t+1} = h_t + f(h_t, t)$$

**層別的解釈**：
- $t = 1, 2, \ldots$ が層インデックス
- 各ステップで層を追加計算

### 8.2 Attention-based Progressive Models

Transformer ベースの段階的学習：

```
Token = "Task1"  → [層1] → [層2] → [層3] → 出力1
Token = "Task2"  → [層1] → [層2] → [層3] → [層4] → 出力2
```

各タスク専用に層が追加される（PNN との類似）。

---

## 9. 統合的な理解

### 9.1 層別訓練の位置づけ

層別訓練は、以下の広い概念の **特殊ケース**：

| 概念 | スコープ |
|-----|---------|
| **Curriculum Learning** | タスク難度の段階化 |
| **Progressive Learning** | モデルの段階的拡張 |
| **Hierarchical Optimization** | 多段階の部分問題 |
| **Transfer & Meta-Learning** | 知識の再利用 |
| **Modular Networks** | 独立モジュールの組み合わせ |

### 9.2 統一的な最適化フレームワーク

```
【多層最適化フレームワーク】

層別訓練 ← { 局所目的関数、固定パラメータ } → PNN
           ↓
      Block Coordinate Descent ← 理論的基盤 → Alternating Optimization
           ↓
      Curriculum Learning ← 難度管理 → Progressive Expansion
```

---

## 10. 実装指針

### 10.1 どの方法を選ぶか

| 状況 | 推奨手法 |
|-----|--------|
| 極深モデル（100+ 層） | 層別訓練 + バックプロパ fine-tuning |
| 複数タスク共存 | Multi-task Learning + PNN |
| 少数サンプル学習 | Meta-Learning (MAML) |
| プログレッシブな学習 | Curriculum Learning |
| 強化学習 | Hierarchical RL |

### 10.2 ハイブリッド戦略

```python
# 実装例：PNN に層別訓練的な固定を組み込む

class ProgressiveLayerWise(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.columns = []
        self.freeze_before_task = 0

    def add_task_column(self, task_id):
        # 新しい列を追加
        new_column = TaskSpecificColumn(...)
        self.columns.append(new_column)

    def train_column(self, task_id, data_loader):
        # 現在の列のみを訓練
        for batch in data_loader:
            # 前のタスクの列は frozen
            for prev_task in range(task_id):
                self.columns[prev_task].freeze()

            # 現在の列のみ学習
            output = self.forward_task(task_id, batch)
            loss = compute_loss(output, batch)
            loss.backward()
            self.optimizer.step()
```

---

## 参考文献

- **Rusu, A. A., Rabinowitz, N. C., Desjardins, G., et al.** (2016). Progressive Neural Networks. arXiv:1606.04671.
- **Finn, C., Abbeel, P., & Levine, S.** (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks. ICML.
- **Vezhnevets, A., Mnih, V., Agapiou, J., Osindero, S., Graves, A., Vinyals, O., & Kavukcuoglu, K.** (2016). Strategic Attentive Writer for Learning Macro-Actions. arXiv:1606.04695.
- **Bengio, Y., Louradour, J., Collobert, R., & Weston, J.** (2009). Curriculum Learning. ICML.
- **Sabour, S., Frosst, N., & Hinton, G. E.** (2017). Dynamic Routing Between Capsules. NIPS.
- **Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. K.** (2018). Neural Ordinary Differential Equations. NIPS.
