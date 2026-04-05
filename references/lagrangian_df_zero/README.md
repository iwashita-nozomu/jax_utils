# Lagrangian DF=0 Reference Pack

このディレクトリは、`lagrangian_df_zero` 系の調査で使う一次資料パックです。
目的は、$DF=0$ 型の layer update を調べるときに、

- どの research line を主線に置くか
- どの論文を比較対象として読むか
- どの local artifact が current design に入っているか

をすぐ辿れるようにすることです。

## Question

- 標準的な feedforward network の layer update を、単なる backprop step ではなく、Bellman 的な reduced problem の停留条件として与えられるか。

## Structure

- `README.md`
  - このファイル
  - pack 全体の overview、reading path、current indexing status を置く
- [`catalog.md`](catalog.md)
  - 各 artifact の role、current use、official link をまとめた詳細索引
- [Survey report](/workspace/notes/themes/06_lagrangian_df_zero.md)
  - reader-facing の literature survey 本体
  - question、search log、comparison、support / narrowing、implementation implication を置く

## Research Lines

### 1. Historical motivation

layer-wise training が optimization を助けうる、という歴史的背景です。
ただし exact reduced problem や Bellman 的な value function をそのまま与える系統ではありません。

- [`Hinton_Osindero_Teh_2006_A_Fast_Learning_Algorithm_for_Deep_Belief_Nets.pdf`](Hinton_Osindero_Teh_2006_A_Fast_Learning_Algorithm_for_Deep_Belief_Nets.pdf)
- [`Bengio_2007_Greedy_Layer_Wise_Training_of_Deep_Networks.pdf`](Bengio_2007_Greedy_Layer_Wise_Training_of_Deep_Networks.pdf)

### 2. Exact equality-constraint line

network を auxiliary state つき constrained problem に持ち上げる主線です。
current design の structural precedent はここに置きます。

- [`Carreira-Perpinan_Wang_2014_Distributed_Optimization_of_Deeply_Nested_Systems.pdf`](Carreira-Perpinan_Wang_2014_Distributed_Optimization_of_Deeply_Nested_Systems.pdf)

### 3. Lifted and relaxation baselines

exact reduced problem そのものではなく、lifted / penalty / relaxation で training を扱う比較対象です。
current design では baseline として読みます。

- [`Askari_Negiar_Sambharya_ElGhaoui_2021_Lifted_Neural_Networks.pdf`](Askari_Negiar_Sambharya_ElGhaoui_2021_Lifted_Neural_Networks.pdf)
- [`Gu_Askari_ElGhaoui_2020_Fenchel_Lifted_Networks.pdf`](Gu_Askari_ElGhaoui_2020_Fenchel_Lifted_Networks.pdf)
- [`Wang_Benning_2023_Lifted_Bregman_Training_of_Neural_Networks.html`](Wang_Benning_2023_Lifted_Bregman_Training_of_Neural_Networks.html)

### 4. Optimizer baseline line

exact reduced problem そのものではないが、gradient-free や block-coordinate による training line を比較したいときの参照線です。
solver choice を詰めるときの narrowing source として使います。

- [Lau et al. (official arXiv)](https://arxiv.org/abs/1803.09082)

### 5. Implicit differentiation and root-solve boundary

inner solve を持つなら、outer differentiation をどう切るかの参照線です。
solver boundary と differentiation policy はここを参照します。

- [`Amos_Kolter_2017_OptNet_Differentiable_Optimization_as_a_Layer_in_Neural_Networks.pdf`](Amos_Kolter_2017_OptNet_Differentiable_Optimization_as_a_Layer_in_Neural_Networks.pdf)
- [`Bai_Kolter_Koltun_2019_Deep_Equilibrium_Models.pdf`](Bai_Kolter_Koltun_2019_Deep_Equilibrium_Models.pdf)

### 6. Contrast line

stationary learning の別系統として読む文献です。
現行 `neuralnetwork` へ直接移す主線ではありませんが、比較のために残します。

- [`Scellier_Bengio_2017_Equilibrium_Propagation.pdf`](Scellier_Bengio_2017_Equilibrium_Propagation.pdf)

## Recommended Reading Paths

### Bellman and function-space path

Bellman 的な $\phi_k / V_k / \Lambda_k$ の整理に寄せて読む経路です。

1. [`Carreira-Perpinan_Wang_2014_Distributed_Optimization_of_Deeply_Nested_Systems.pdf`](Carreira-Perpinan_Wang_2014_Distributed_Optimization_of_Deeply_Nested_Systems.pdf)
1. [`Gu_Askari_ElGhaoui_2020_Fenchel_Lifted_Networks.pdf`](Gu_Askari_ElGhaoui_2020_Fenchel_Lifted_Networks.pdf)
1. [`Amos_Kolter_2017_OptNet_Differentiable_Optimization_as_a_Layer_in_Neural_Networks.pdf`](Amos_Kolter_2017_OptNet_Differentiable_Optimization_as_a_Layer_in_Neural_Networks.pdf)
1. [`Bai_Kolter_Koltun_2019_Deep_Equilibrium_Models.pdf`](Bai_Kolter_Koltun_2019_Deep_Equilibrium_Models.pdf)

### Baseline comparison path

historical motivation、lifted baseline、exact constraint line を比較したいときの経路です。

1. [`Hinton_Osindero_Teh_2006_A_Fast_Learning_Algorithm_for_Deep_Belief_Nets.pdf`](Hinton_Osindero_Teh_2006_A_Fast_Learning_Algorithm_for_Deep_Belief_Nets.pdf)
1. [`Bengio_2007_Greedy_Layer_Wise_Training_of_Deep_Networks.pdf`](Bengio_2007_Greedy_Layer_Wise_Training_of_Deep_Networks.pdf)
1. [`Askari_Negiar_Sambharya_ElGhaoui_2021_Lifted_Neural_Networks.pdf`](Askari_Negiar_Sambharya_ElGhaoui_2021_Lifted_Neural_Networks.pdf)
1. [`Gu_Askari_ElGhaoui_2020_Fenchel_Lifted_Networks.pdf`](Gu_Askari_ElGhaoui_2020_Fenchel_Lifted_Networks.pdf)
1. [Lau et al. (official arXiv)](https://arxiv.org/abs/1803.09082)
1. [`Carreira-Perpinan_Wang_2014_Distributed_Optimization_of_Deeply_Nested_Systems.pdf`](Carreira-Perpinan_Wang_2014_Distributed_Optimization_of_Deeply_Nested_Systems.pdf)

### Implementation boundary path

current repo の design / solver boundary を詰めるときの最短経路です。

1. [`Carreira-Perpinan_Wang_2014_Distributed_Optimization_of_Deeply_Nested_Systems.pdf`](Carreira-Perpinan_Wang_2014_Distributed_Optimization_of_Deeply_Nested_Systems.pdf)
1. [`Amos_Kolter_2017_OptNet_Differentiable_Optimization_as_a_Layer_in_Neural_Networks.pdf`](Amos_Kolter_2017_OptNet_Differentiable_Optimization_as_a_Layer_in_Neural_Networks.pdf)
1. [`Bai_Kolter_Koltun_2019_Deep_Equilibrium_Models.pdf`](Bai_Kolter_Koltun_2019_Deep_Equilibrium_Models.pdf)
1. [Lau et al. (official arXiv)](https://arxiv.org/abs/1803.09082)
1. [`catalog.md`](catalog.md)

## Current Indexing Status

- current canonical line
  - exact equality-constraint plus Bellman/function-space reading
- current baselines
  - lifted / penalty / relaxation line
- current optimizer baseline
  - proximal / block-coordinate descent line
- current solver boundary
  - implicit differentiation and root-solve line
- indexed artifacts
  - Hinton 2006
  - Bengio 2007
  - Carreira-Perpinan and Wang 2014
  - Askari et al. 2021
  - Gu et al. 2020
  - Amos and Kolter 2017
  - Bai et al. 2019
  - Lau et al. 2018
  - Scellier and Bengio 2017
  - Wang and Benning 2023 landing page

## See Also

- [Detailed catalog](catalog.md)
- [Theme note](/workspace/notes/themes/06_lagrangian_df_zero.md)
- [Design note](/workspace/documents/design/jax_util/lagrangian_df_zero.md)
- [Bellman notation note](/workspace/notes/themes/bellman_bp_chat_summary.md)
