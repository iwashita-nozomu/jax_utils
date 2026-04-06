# Theme Notes

`notes/themes/` には、複数の実験から得た知見を、話題ごとにまとめます。

- `notes/experiments/`
  - 個別実験の report と結果解釈
- `notes/branches/`
  - branch ごとの要約と入口
- `notes/worktrees/`
  - 削除済み worktree から吸い出した判断

に対し、このディレクトリでは「その話題について今何が言えるか」と、「どんな工夫が効き、何がうまくいかなかったか」を topic 単位で整理します。

## Format

- 1 theme 1 file を基本とします。
- 歴史の全記録ではなく、現時点の知識として再利用したい項目を優先します。
- `Known`, `Likely`, `Open` のような知識ラベルに加えて、`Worked`, `Did Not Work`, `Coding Pattern`, `Pitfall` のような実装ラベルを使えます。
- うまくいかなかった案も、「なぜやめたか」「どこで詰まったか」が分かる形で残します。
- 根拠となる report や result JSON へのリンクは末尾に集約できます。
- 図と数式を入れる場合は本文に埋め込みます。図は `main` 側の `notes/assets/` に置きます。
- その topic に標準的な論文や教科書がある場合は、末尾に代表文献を置きます。
- 観測ベースの知見と文献ベースの知見が混ざるときは、本文か reference で区別できるようにします。
- 本文では、branch 名、worktree 名、`旧版`、`改訂版` のような内部の呼び名をできるだけ避けます。
- 本文では、`点群を明示的に作る方法`、`plan を先に作る方法` のように、方法そのものを主語にします。
- 内部の履歴や呼び名が必要な場合は、本文ではなく `References` か `diary/` に回します。
- 文は短くします。1 文で 1 つのことだけを書きます。
- 箇条書きだけで済ませず、短い段落でその意味も書きます。

## Current Themes

- [01_rbm_and_contrastive_divergence.md](/workspace/notes/themes/01_rbm_and_contrastive_divergence.md)
- [02_convergence_and_optimality.md](/workspace/notes/themes/02_convergence_and_optimality.md)
- [03_gpu_and_parallelization.md](/workspace/notes/themes/03_gpu_and_parallelization.md)
- [04_progressive_neural_networks_and_meta_learning.md](/workspace/notes/themes/04_progressive_neural_networks_and_meta_learning.md)
- [05_bellman_equation_and_deep_learning.md](/workspace/notes/themes/05_bellman_equation_and_deep_learning.md)
- [06_lagrangian_df_zero.md](/workspace/notes/themes/06_lagrangian_df_zero.md)
- [07_df_zero_and_local_quadratic_models.md](/workspace/notes/themes/07_df_zero_and_local_quadratic_models.md)
- [bellman_bp.md](/workspace/notes/themes/bellman_bp.md)
- [bellman_bp_chat_summary.md](/workspace/notes/themes/bellman_bp_chat_summary.md)
- [desigin.md](/workspace/notes/themes/desigin.md)
- [experiment_runner.md](/workspace/notes/themes/experiment_runner.md)
- [layer_wise_train.md](/workspace/notes/themes/layer_wise_train.md)
- [smolyak_integrator.md](/workspace/notes/themes/smolyak_integrator.md)
