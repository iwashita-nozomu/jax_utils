# Smolyak Experiment Results Branch Summary

## Branch

- `results/smolyak-experiment-20260321`

## Role

- `experiments/smolyak_experiment/` の簡略化、spawn ベース runner への接続、実験ディレクトリ設計を進めた branch
- 長時間実験を results branch で保持しつつ、code / note / 最小結果を main へ戻す運用を固めた branch

## Primary Knowledge

- 実験ディレクトリの最終方針:
  - [experiment_directory_planning.md](/workspace/notes/knowledge/experiment_directory_planning.md)
- 実験運用の要点:
  - [experiment_operations.md](/workspace/notes/knowledge/experiment_operations.md)

## What Survived

この branch で形になった重要な判断は、`python/experiment/` のような新しい大箱を増やすことではなく、共通 runtime を `python/experiment_runner/` に集約し、topic 固有コードを `experiments/` 配下へ寄せることだった。

また、spawn ベースの fresh child process を軸にして、Smolyak 実験 script を不要に複雑化させない方針もここで強くなった。

## Status

- archived
- Retention: `delete-ok`
- `main` へそのまま merge する対象ではなく、知見は note と規約へ吸収済み

## References

- [work_smolyak_improvement_20260318.md](/workspace/notes/branches/work_smolyak_improvement_20260318.md)
- [worktree_work_smolyak_improvement_2026-03-18.md](/workspace/notes/worktrees/worktree_work_smolyak_improvement_2026-03-18.md)
