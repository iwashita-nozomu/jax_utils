# Experiment Runner Module Branch Summary

## Branch

- `work/experiment-runner-module-20260316`

## Role

- 実験 runner を `experiment_runner` へ切り出し、`smolyak_scaling` から共通化した作業 branch

## Primary Note

- [worktree_experiment_runner_module_2026-03-16.md](/workspace/notes/worktrees/worktree_experiment_runner_module_2026-03-16.md)

## What Was Done

この branch では、`run_smolyak_scaling.py` に埋め込まれていた runner 固有の責務を `python/experiment_runner/` へ切り出した。対象は主に、case queue、worker slot 管理、subprocess 起動、JSONL 逐次保存、timeout / worker termination の補足、fallback record 生成といった部分である。`smolyak_scaling` 自体はその上に乗る 1 実験へ下げ、runner 共通部を他の実験にも流用できる構造にした。

設計としては、host が free slot と timeout を見ながら child を dispatch し、child はケース実行と JSONL 追記を担当したうえで completion record を明示的に返す形にした。これにより、`Popen.wait()` だけでは区別しにくい異常終了と正常終了を切り分けやすくなった。また、child command builder と failure policy を分離したことで、experiment code 側では「何を実行するか」に集中しやすくなった。

## Validation

この branch では multi-GPU child probe を含む test を追加した。GPU test は child が 2.5 秒程度 repeated matmul を回すようにしてあり、`nvidia-smi` でも観測しやすい。`selected_gpu_indices=[0,1]`, `worker_labels=['gpu-0-w0','gpu-1-w0']` が得られ、各 child が 1 GPU だけを見ていること、host scheduler が slot 単位で dispatch できていることを確認した。さらに `pyright` の対象コードも clean にし、`PYTHONPATH` なしの直接実行でも test file 自身が動くように path 補助を入れた。

## Interpretation

- この branch は、Smolyak 固有の改善というより、今後の experiment code 全体の土台を作った branch として読むのがよい。特に「長時間実験を途中停止しても JSONL が残る」「child の completion を host が明示的に受け取る」という運用基盤は、この branch で形になった。

## Status

- archived
- 成果は `main` と `results/functional-smolyak-scaling-tuned` に取り込まれている
