# Smolyak Improvement Branch Summary

## Branch

- `work/smolyak-improvement-20260318`

## Role

- Smolyak 実験 script の簡略化、spawn ベース runner 接続、benchmark / experiment の境界整理を進めた branch
- 後続の `results/smolyak-experiment-20260321` と main 側の規約整理の前段になった branch

## Primary Notes

- [worktree_work_smolyak_improvement_2026-03-18.md](/workspace/notes/worktrees/worktree_work_smolyak_improvement_2026-03-18.md)
- [experiment_directory_planning.md](/workspace/notes/knowledge/experiment_directory_planning.md)

## What Was Learned

この branch では、Smolyak 実験を長期間回すには code branch と results branch を分ける必要があること、そして process lifecycle を experiment script に埋め込まず runner 側へ寄せる必要があることが明確になった。

同時に、誤った helper API 前提で Smolyak 本体を改造し始めると merge conflict と責務混線が増える、という教訓も残った。

## Status

- archived
- Retention: `delete-ok`
- branch 上の個別 commit をそのまま再利用するより、main に吸収済みの設計判断を参照する方が安全

## References

- [results_smolyak_experiment_20260321.md](/workspace/notes/branches/results_smolyak_experiment_20260321.md)
- [smolyak_infrastructure_checkpoint_20260320.md](/workspace/notes/experiments/smolyak_infrastructure_checkpoint_20260320.md)
