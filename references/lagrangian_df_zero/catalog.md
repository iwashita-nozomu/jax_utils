# Lagrangian DF=0 Catalog

`references/lagrangian_df_zero/` に置いた local artifact の詳細索引です。
overview と reading path は [README.md](README.md) に置き、このファイルでは 1 件ごとの役割を固定します。

## Canonical Line

| Artifact                                                                                                                                                               | Status  | Why it matters                                                                            | Current use                            | Official link                                                     |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ----------------------------------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------- |
| [Carreira-Perpinan_Wang_2014_Distributed_Optimization_of_Deeply_Nested_Systems.pdf](Carreira-Perpinan_Wang_2014_Distributed_Optimization_of_Deeply_Nested_Systems.pdf) | primary | auxiliary coordinates により nested objective を equality-constraint problem に持ち上げる | current design の structural precedent | [PMLR](https://proceedings.mlr.press/v33/carreira-perpinan14.pdf) |

## Historical Motivation

| Artifact                                                                                                                                                   | Status  | Why it matters                          | Current use                            | Official link                                                                                            |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | --------------------------------------- | -------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [Hinton_Osindero_Teh_2006_A_Fast_Learning_Algorithm_for_Deep_Belief_Nets.pdf](Hinton_Osindero_Teh_2006_A_Fast_Learning_Algorithm_for_Deep_Belief_Nets.pdf) | context | layer-wise pretraining の歴史的起点     | historical context only                | [official](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf)                                          |
| [Bengio_2007_Greedy_Layer_Wise_Training_of_Deep_Networks.pdf](Bengio_2007_Greedy_Layer_Wise_Training_of_Deep_Networks.pdf)                                 | context | greedy layer-wise training の代表的整理 | historical and comparison context only | [NeurIPS](https://papers.nips.cc/paper_files/paper/2006/file/5da713a690c067105aeb2fae32403405-Paper.pdf) |

## Lifted And Relaxation Baselines

| Artifact                                                                                                                               | Status            | Why it matters                                       | Current use                                      | Official link                                                                        |
| -------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ---------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------ |
| [Askari_Negiar_Sambharya_ElGhaoui_2021_Lifted_Neural_Networks.pdf](Askari_Negiar_Sambharya_ElGhaoui_2021_Lifted_Neural_Networks.pdf)   | baseline          | activation を convex subproblem として lifted 化する | comparison baseline                              | [Berkeley EECS](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2021/EECS-2021-218.pdf) |
| [Gu_Askari_ElGhaoui_2020_Fenchel_Lifted_Networks.pdf](Gu_Askari_ElGhaoui_2020_Fenchel_Lifted_Networks.pdf)                             | baseline          | Lagrange multiplier を含む lifted / relaxation line  | comparison baseline with multiplier structure    | [PMLR](https://proceedings.mlr.press/v108/gu20a/gu20a.pdf)                           |
| [Wang_Benning_2023_Lifted_Bregman_Training_of_Neural_Networks.html](Wang_Benning_2023_Lifted_Bregman_Training_of_Neural_Networks.html) | landing-page-only | lifted 系の発展形                                    | follow-up candidate; local PDF is not stored yet | [JMLR](https://jmlr.org/papers/v24/22-0934.html)                                     |

## Optimizer Baseline Line

| Artifact                         | Status             | Why it matters                                                            | Current use                              | Official link                             |
| -------------------------------- | ------------------ | ------------------------------------------------------------------------- | ---------------------------------------- | ----------------------------------------- |
| Lau et al. (official arXiv only) | optimizer-baseline | gradient-free / block-coordinate optimization line with convergence focus | solver-choice narrowing source, baseline | [arXiv](https://arxiv.org/abs/1803.09082) |

## Implicit Differentiation And Root-Solve Boundary

| Artifact                                                                                                                                                                       | Status           | Why it matters                                       | Current use                        | Official link                                                                                        |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------- | ---------------------------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------------------------- |
| [Amos_Kolter_2017_OptNet_Differentiable_Optimization_as_a_Layer_in_Neural_Networks.pdf](Amos_Kolter_2017_OptNet_Differentiable_Optimization_as_a_Layer_in_Neural_Networks.pdf) | solver-reference | KKT solve を layer とみなす implicit differentiation | differentiation boundary reference | [PMLR](https://proceedings.mlr.press/v70/amos17a/amos17a.pdf)                                        |
| [Bai_Kolter_Koltun_2019_Deep_Equilibrium_Models.pdf](Bai_Kolter_Koltun_2019_Deep_Equilibrium_Models.pdf)                                                                       | solver-reference | root-finding と implicit differentiation の代表例    | root-solve boundary reference      | [NeurIPS](https://proceedings.neurips.cc/paper/2019/file/01386bd6d8e091c2ab4c7c7de644d37b-Paper.pdf) |

## Contrast Line

| Artifact                                                                                             | Status   | Why it matters               | Current use         | Official link                             |
| ---------------------------------------------------------------------------------------------------- | -------- | ---------------------------- | ------------------- | ----------------------------------------- |
| [Scellier_Bengio_2017_Equilibrium_Propagation.pdf](Scellier_Bengio_2017_Equilibrium_Propagation.pdf) | contrast | stationary learning の別系統 | contrast paper only | [arXiv](https://arxiv.org/abs/1602.05179) |

## Current Reading Decision

- canonical design line
  - Carreira-Perpinan and Wang (2014)
- baseline comparison line
  - Askari et al. (2021)
  - Gu et al. (2020)
  - Wang and Benning (2023)
- optimizer baseline line
  - Lau et al. (2018)
- solver and differentiation line
  - Amos and Kolter (2017)
  - Bai et al. (2019)
- historical context only
  - Hinton et al. (2006)
  - Bengio et al. (2007)
- contrast only
  - Scellier and Bengio (2017)
