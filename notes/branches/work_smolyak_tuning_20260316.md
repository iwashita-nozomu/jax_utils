# Smolyak Tuning Branch Summary

## Branch

- `work/smolyak-tuning-20260316`

## Role

- tuned Smolyak 積分器の設計変更、HLO 解析、GPU 可視性の切り分けを行った作業 branch

## Primary Note

- [worktree_smolyak_tuning_2026-03-16.md](/workspace/notes/worktrees/worktree_smolyak_tuning_2026-03-16.md)

## What Was Done

この branch では、Smolyak 積分器の評価構造そのものをかなり整理した。explicit な tensor grid と重複点集約に依存する公開経路を外し、`integrate(f, integrator)` に評価 API を寄せたうえで、`initialize_smolyak_integrator(...)` が plan を持つ構造へ直した。また、`prepared_level` を持つ積分器として、最大 level までの rule を準備し、その範囲で `refine()` できる形へ寄せた。単一精度実行、明示 grid の削減、unused code の削除もこの branch の仕事だった。

加えて、HLO 解析系を大きく拡充した。`jax_util.hlo.dump` と `scripts/hlo/summarize_hlo_jsonl.py` を伸ばし、`stablehlo` / `hlo` の text size、proto size、compiled memory stats、cost analysis を JSONL と summary に残せるようにした。その上で `experiments/functional/smolyak_hlo/` を作り、単一ケースの HLO から bottleneck を見る実験を追加した。

GPU が 1 枚しか使われていないように見える問題についても、この branch で切り分けを行った。`debug_gpu_visibility.py` により、`ProcessPoolExecutor` と subprocess の両方で child ごとの `CUDA_VISIBLE_DEVICES` が効いており、各 child は local には `cuda:0` しか見ていないことを確認した。これにより、GPU 可視性バグではなく、CPU 側初期化支配が主因だと判断できた。

## What Was Learned

HLO 解析からは、算術 kernel より `stablehlo.while`, `func.call`, `stablehlo.gather` が目立つことが分かった。小さい case でもこの傾向は安定しており、点数爆発より前に制御フローと index 処理のコストが前面に出ていると読めた。さらに tuned 実装の大規模 run では、`level=1`, `num_points=1` の時点で次元とともに `integrator_init_seconds` と RSS が強く増えていた。したがって、この branch の最大の成果は「explicit grid を減らせばすぐ GPU bound になる」という期待を外し、実際には初期化・lowering・compile が主因候補だと絞り込んだことにある。

## Validation

この branch では、Smolyak 周りの functional test、HLO 実験の CPU/GPU 実行、GPU 可視性 probe、小さい CPU smoke run を通している。追跡用の細かい JSON/JSONL は持ち込んでいないが、結論は [worktree_smolyak_tuning_2026-03-16.md](/workspace/notes/worktrees/worktree_smolyak_tuning_2026-03-16.md) に吸い出してある。

## Status

- archived
- worktree 削除済み
- tracked 成果は `results/functional-smolyak-scaling-tuned` に包含済み
