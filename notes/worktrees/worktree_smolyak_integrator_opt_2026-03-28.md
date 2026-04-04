# Smolyak Integrator Optimization Worktree

## Summary

- branch: `work/smolyak-integrator-opt-20260328`
- worktree: `/workspace/.worktrees/work-smolyak-integrator-opt-20260328`
- role: `SmolyakIntegrator` の最適化と実験 README の整理
- base commit: `12dee48`

## Action Log

- 2026-03-28: `12dee48` を起点に worktree を作成し、Smolyak integrator の最適化作業を開始。
- 2026-03-31: `origin/main` を fast-forward で取り込み、`8c0cdef` と `9233674` を反映。
- 2026-03-31: 当初の Smolyak 実験 README の stash conflict を解消し、fresh-run rule と local の再現手順を両立させた。
- 2026-03-31: `WORKTREE_SCOPE.md` と本 action log を追加し、branch summary から実在する入口を辿れる状態へ戻した。
- 2026-04-04: `work/smolyak-integrator-mainready-20260404` を切り、main へ持ち込むための統合 branch として code / experiment / notes を束ねる方針にした。

## Observations

- 現在の主要変更対象は `python/jax_util/functional/smolyak.py` と `experiments/functional/smolyak_scaling/README.md`。
- `python/jax_util/functional/smolyak.py` は `work/smolyak-integrator-lead-20260328` と重なるため、API や test 契約の変更は branch 間の整合が必要。
- raw 実験結果の保持先は `results/smolyak-validation-20260328` とし、この worktree ではコードと入口文書の整理に寄せる。

## Carry-Over Targets

- `notes/branches/work_smolyak_integrator_opt_20260328.md`
- `notes/experiments/smolyak_scaling_experiment.md`
- `python/jax_util/functional/smolyak.py`
- `experiments/functional/smolyak_scaling/README.md`

## Interpretation

- 実験を回しながら改造する運用は維持しつつ、この worktree は「最適化と再現入口」に責務を絞ると衝突を減らしやすい。

## Cleanup Prep

- 2026-04-04 時点の branch head: `bdcbfa6` (`chore: sync experiment runner contract from main`)
- 直近の戻り先候補:
  - `7338c4a` `chore(results): refresh latest smolyak scaling snapshot`
  - `25f2345` `perf(smolyak): reduce term loop and rule build overhead`
  - `d41b43c` `perf(smolyak): improve decode and rule storage init`
  - `57b406a` `chore: checkpoint smolyak runtime and experiment tooling`

現在の主な未整理差分:

- integrator:
  - [smolyak.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/smolyak.py)
  - [integrate.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/integrate.py)
  - [monte_carlo.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/python/jax_util/functional/monte_carlo.py)
- experiment:
  - [run_smolyak_scaling.py](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_scaling/run_smolyak_scaling.py)
  - [README.md](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/experiments/functional/smolyak_scaling/README.md)
- notes:
  - [smolyak_vmap_xla_memory.md](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/notes/knowledge/smolyak_vmap_xla_memory.md)
  - [smolyak_runtime_cost_model_20260403.md](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/notes/knowledge/smolyak_runtime_cost_model_20260403.md)
  - [smolyak_vmap_idea_campaign_20260402.md](/workspace/.worktrees/work-smolyak-integrator-opt-20260328/notes/experiments/smolyak_vmap_idea_campaign_20260402.md)

片付けるときの優先順:

1. integrator 本体の採否を確定
2. 実験 runner 変更と baseline 結果を別 commit に切る
3. note を最終状態へ揃える
4. `results/latest.json` のような生成物更新を最後にまとめる

一時 HLO dump は `/tmp/xla_dump_*` にあり、repo 管理対象ではない。cleanup 時に repo 内の source/note だけ見ればよい。
