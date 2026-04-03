# Experiment Runner

## このページの役割

このページは、`experiment_runner` について branch ごとの試行錯誤を main 側の再利用知識へ直したものです。

ここでは履歴よりも、今の設計で何を固定し、何を避けるべきかを優先します。

## いまの正本

いまの標準経路は、`StandardRunner` が case ごとに fresh child process を起動する形です。

実験コードは、基本的に

- case 定義
- worker task
- resource estimate

だけを書き、timeout、child cleanup、diagnostics、JSONL 保存、GPU slot 割当は runner 側へ寄せます。

## Worked

### Fresh Child Process

JAX や CUDA は import-sensitive な state を持つため、case ごとに fresh process を立てる設計が安定でした。

process pool や同一 process 再利用より、`spawn` で毎回 child を分けるほうが case 間汚染を避けやすく、失敗の切り分けも容易でした。

### Scheduler-Driven Environment

GPU 可視性や allocator 系 env は、experiment script 側ではなく scheduler / runner 側で組み立てるのが有効でした。

実験コード側で `CUDA_VISIBLE_DEVICES` や `XLA_*` を直接組むと、resource 割当と import 順序の責務がぶつかります。

### Structured Diagnostics

child の Python 例外だけでなく、timeout、signal 終了、completion 欠落も parent 側で診断情報に寄せる形が扱いやすかったです。

`raw exitcode`、`failure_kind`、message、traceback を分けると、framework 固有の失敗解釈を core runner に埋め込まずに済みます。

### Single Execution Path

runner の実行経路を 1 本に揃えたのは正解でした。

`subprocess_scheduler` と `StandardRunner` の二系統を並立させるより、fresh child process 前提の 1 経路に揃えたほうが保守しやすく、テストも整理しやすくなりました。

## Did Not Work

### Runner の二重化

host が `Popen` を握る経路と `StandardRunner` 系を並行維持すると、monitor、GPU env、JSONL helper、failure handling が二重化しやすくなりました。

実験コードから見る入口は 1 本に保つべきです。

### 実験コード側の直接 env 制御

experiment script 側で JAX/XLA env を直接触る方式は長続きしませんでした。

JAX import 前に env を入れる必要がある一方で、どの GPU を使うかは scheduler が知っているからです。

### Parent Cleanup の後回し

child の異常終了を Python 例外だけで見ようとすると、native crash や hang の回収が弱くなります。

timeout と `terminate -> kill` の cleanup は runner core が持つ必要があります。

## Coding Pattern

- child task の先頭で env を適用し、その後に `jax` を import する
- resource 割当は `StandardFullResourceScheduler` に寄せる
- JSONL 逐次保存と final JSON 生成は runner 共通 helper を使う
- 実験コードでは case と比較対象を主語にし、process lifecycle は書かない

## Pitfall

- parent を中断したあと child が残ると、次の GPU run の frontier を汚します
- worktree 上の runner 改造は、テストと文書を同時に持ち帰らないと知識が分散します
- `PYTHONPATH` を明示しないと main と worktree で別の code path を踏みやすいです

## References

- [work_experiment_runner_module_20260316.md](/workspace/notes/branches/work_experiment_runner_module_20260316.md)
- [work_experiment_runner_refactor_20260330.md](/workspace/notes/branches/work_experiment_runner_refactor_20260330.md)
- [worktree_experiment_runner_module_2026-03-16.md](/workspace/notes/worktrees/worktree_experiment_runner_module_2026-03-16.md)
- [worktree_experiment_runner_refactor_2026-03-30.md](/workspace/notes/worktrees/worktree_experiment_runner_refactor_2026-03-30.md)
- [worktree_runner_diagnostics_2026-04-03.md](/workspace/notes/worktrees/worktree_runner_diagnostics_2026-04-03.md)
- [worktree_runner_single_path_2026-04-03.md](/workspace/notes/worktrees/worktree_runner_single_path_2026-04-03.md)
- [experiment_runner.md](/workspace/documents/experiment_runner.md)
