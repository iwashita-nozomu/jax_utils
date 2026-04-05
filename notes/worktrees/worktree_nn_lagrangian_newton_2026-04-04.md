# NN Lagrangian Newton Worktree Log

## 2026-04-04

- `work/nn-lagrangian-newton-20260404` を current `main` (`6030e1e`) から作成した。
- 目的は、ラグランジアンの二回微分までを使う Newton 型手法の文献調査、数理検証、設計、実装を、既存 `neuralnetwork` 本線と衝突しにくい別サブモジュールとして進めること。
- 作業対象は `python/jax_util/neuralnetwork/lagrangian_newton/` を仮の正本とし、既存の `neuralnetwork.py`、`train.py`、`sequential_train.py`、`dynamic_programming.py` は読み取り中心とする。
- 初期段階では、文献調査、数理的検証、設計文書、reference 整理を優先し、本実装はその後に入る。
- `Workflow Family:` `Research-Driven Change`
- `Question:` ラグランジアンの二階微分まで使う Newton 型更新を、既存 `NeuralNetwork` の forward 実装と分離したサブモジュールとしてどう定式化するか。まずは standard / ICNN の小規模回帰で、層状態とパラメータを含む制約付き定式化へ落とせるかを確認する。
- `Formulation:` 第 1 反復では `layer_utils.module_to_vector` / `vector_to_module` による `eqx.Module -> Vector` 変換を前提にし、状態変数は forward cache、制約は layer transition consistency、目的は loss とする。Newton / KKT 系は dense Hessian を直接作らず、`base.linearize` / `base.adjoint` と `solvers.kkt_solver` の matrix-free 作用素で表現する前提で進める。
- `Comparison Target:` 既存の比較対象は 1) `sequential_train.py` の一次更新系、2) `dynamic_programming.py` の層別学習着想、3) 小規模 MLP に対する autodiff 勾配と loss 減少。現時点では本線の train API は未成熟なので、最初の exit criteria は「同じ toy 問題で KKT residual と loss が一貫して減る設計」が置けることにする。
- `Decision:` `neuralnetwork.py` の forward 実装と layer protocol は読み取り専用に保ち、新規ロジックは `lagrangian_newton/` に閉じる。
- `Decision:` 再利用候補として `neuralnetwork/protocols.py` の `Params` / `Static` / `PyTreeOptimizationProblem`、`layer_utils.py` の flatten / unflatten、`base` の `Vector` / `Matrix` / `linearize` / `adjoint`、`solvers/kkt_solver.py`、`optimizers/pdipm.py` の Hessian / KKT 組み立てパターンを採用候補とする。
- `Decision:` 初回の prototype は `layer transition consistency` を主制約に置く。`stationarity` を直接制約にする案は比較対象として保持し、第 2 段階で再評価する。
- `Decision:` 数式導出は functional-first にする。まず関数空間で随伴と汎関数勾配を導出し、離散化と parameter 制約は後段で入れる。
- `Risk:` 既存 train パスは未実装箇所が多く、そこへ強依存すると scope drift しやすい。第 1 段階では `sequential_train.py` と `dynamic_programming.py` は比較対象に留める。
- `Math Design Entry:` `documents/design/jax_util/lagrangian_newton.md`
- `Implementation Design Entry:` `documents/design/jax_util/lagrangian_newton_impl.md`
- `Reference Note:` 文献シードは `references/lagrangian-newton-references.md` に集約し、Hessian-free / Gauss-Newton / augmented Lagrangian / lifted training の系譜を分けて追う。
- `Theme Note:` 再利用したい知見は `notes/themes/06_lagrangian_newton_and_constrained_training.md` に整理する。
- `Mathematical Draft:` 設計文書に、記号、constrained problem、ラグランジアン、KKT 条件、Gauss-Newton 近似、`\widetilde F_k` の二階微分、2 層 toy 例を書き下ろした。初回の数式正本は `documents/design/jax_util/lagrangian_newton.md` とする。
- `Validation Plan:` 先に文献ノートと設計文書を整え、その後に 2 層程度の toy network で `loss` と分解した `KKT residual`（parameter / state / constraint）を確認する最小 prototype を切る。
- `Review Plan:` 実装前に設計文書レビュー、Python 差分後に `pyright` / `ruff` / `pytest` と `code-review`、`python-review` を実施する。研究主張や report を閉じる段階では `critical-review` と `report-review` を挟む。
- `Decision:` action log の正本はこのファイル 1 本に戻す。補助的に作成した parent-only note の内容はここへ統合し、以後はこの log へ追記する。

## 2026-04-05

- `Change:` 先行研究を、direct second-order、lifted / constrained training、dual / Riesz / pullback metric、反証候補の 4 系統に分けて再調査した。
- `Source Selection:` 2026-04-05 時点で、原則として arXiv、JMLR、PMLR、ICML の一次資料を優先した。supportive な論文だけでなく、Kervadec et al. (2019)、Arbel et al. (2023) のような cautionary source も含めた。
- `Result Summary:` 新しい prior-art report として `notes/themes/07_lagrangian_newton_prior_art_report_2026-04-05.md` を作成した。設計上いちばん近い一次資料は Evens et al. (2021) であり、最新の整理として Wang et al. (2025/2026) を追加した。
- `Interpretation:` 本 worktree の方向は、1) direct Gauss-Newton、2) lifted / constrained training、3) function-space dual / mass matrix の交点にある。特に `f` 微分から始める方針と、質量行列・合成行列で `\widetilde F_k` の二階微分を離散化する整理は、既存 3 系統を接続する独自整理として残る。
- `Limitation:` 今回の survey では「Bellman 型 reduced functional の Hessian をそのまま前面に出した完成形」は見つけられなかった。これは absent proof ではなく、2026-04-05 時点の検索範囲で未確認という意味に留める。
- `Change:` 参照した公開文献 17 本を `references/lagrangian_newton/` に取得し、`references/lagrangian_newton/README.md` から辿れるようにした。
- `Validation:` 保存した各ファイルについて、先頭 4 byte が `%PDF` であることを確認した。
- `Change:` 拡張ラグランジアン以外の近い文献として、auxiliary coordinates、Pontryagin 型 optimal control、Bregman learning の 3 本を追加した。
- `Interpretation:` non-AL の比較対象としては、Wang and Carreira-Perpinan (2012)、Carreira-Perpinan and Wang (2014)、Li et al. (2018)、Wang and Benning (2023)、Hubler et al. (2022) を先に読むのがよい。
- `Change:` 設計文書を数理と実装に分離した。`lagrangian_newton.md` は KKT と `\widetilde F_k` の二階微分に絞り、再利用資産、実装単位、検証方針は `documents/design/jax_util/lagrangian_newton_impl.md` に移した。
- `Change:` `\widetilde F_k` を局所ラグランジアン `\mathcal L_k` から導く流れを数理設計の主導出にした。global KKT は companion lens として残し、`\widetilde F_k` の一次・二次微分を envelope / adjoint 微分の形で先に出す構成へ直した。
- `Change:` 数理設計の正本から `\mathcal R_k` を外し、`bellman_bp.md` の `J`, `V_k`, `p_k`, `\widetilde F_k`, `\mathcal L_k` だけで閉じる形に寄せた。
- `Change:` 数理設計の章立て自体を整理し直した。追記で継ぎ足す形をやめて、`出発点 -> 単段階の部分問題 -> 局所ラグランジアン -> \widetilde F_k の一次・二次微分 -> 離散化 -> companion KKT` の順に並べ替えた。
- `Change:` `lagrangian_newton.md` の前半を関数空間 first に組み直した。parameter vector と vectorization を先に出す形をやめ、まず `f_k`, `\phi_k`, `p_k` で global constrained problem を展開し、その後に離散化と parameter-space への引き戻しを独立の節へ送った。
- `Change:` 随伴変数の記号を `p` に統一した。有限次元の実装記号は `(\theta, z, p)` とし、parameter residual は `r_\theta`、state residual は `r_z`、constraint residual は `r_c` と書くことにした。
- `Change:` 数理設計の正本から、trial basis、質量行列、pullback metric、parameter vector の詳説をいったん外した。`lagrangian_newton.md` では `F_k`, `\widetilde F_k`, `\mathcal L_k`, 局所応答作用素、Gauss-Newton 主要項までを関数空間の中で閉じ、有限次元化は `lagrangian_newton_impl.md` に移した。
