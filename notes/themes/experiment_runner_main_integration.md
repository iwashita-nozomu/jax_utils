# Experiment Runner Main Integration

この note は、`python/experiment_runner/` を `main` に取り込むときの受け入れ準備を整理するための theme note です。

## Goal

`experiment_runner` を「一部の実験だけが使う worktree 内ユーティリティ」ではなく、`main` 上で再利用できる standalone module として受け入れることが目的です。

ここでやりたいのは次です。

- どのファイルを正本として持ち帰るかを固定する
- 実験コードと runner の責務境界を固定する
- `main` 取り込み時のチェック項目を減らす

## Scope To Carry

`main` に持ち帰る中心は次です。

- `python/experiment_runner/`
- `python/tests/experiment_runner/`
- `documents/experiment_runner.md`
- `notes/experiments/experiment_runner_usage.md`
- `notes/themes/experiment_runner_realtime_monitor.md`

必要に応じて一緒に持ち帰るもの:

- `experiments/smolyak_experiment/` 側の integration patch
- monitor を呼ぶ run script 側の薄い wiring

## What Is Stable Enough

現時点で比較的安定している前提:

- `StandardRunner` は case ごとの fresh child process を使う
- GPU 実験の標準 scheduler は `StandardFullResourceScheduler`
- GPU 可視性の制御は scheduler 側の責務
- JAX allocator 設定は `GPUEnvironmentConfig` に集約する
- monitor は軽量 HTTP + JSON API に留める

## Integration Rules

`main` へ持ち帰るときに、利用ルールも一緒に固定した方がよいです。

### Rule 1: 実験コードは runtime env を直接セットしない

禁止寄りで扱う env:

- `CUDA_VISIBLE_DEVICES`
- `NVIDIA_VISIBLE_DEVICES`
- `JAX_PLATFORMS`
- `XLA_*`
- `TF_GPU_ALLOCATOR`

理由:

- scheduler の資源割当と衝突しやすい
- JAX import 前後の差が大きく、再現性が落ちる
- 問題発生時に責務が曖昧になる

### Rule 2: 実験コードは intent を宣言する

実験コードは次だけを宣言する形に寄せる。

- CPU か GPU か
- 必要 GPU 数
- 推定 GPU memory
- allocator profile を使うか
- 出力 path や task key

### Rule 3: runner 側が env を組み立てる

実際の env 反映は次へ寄せる。

- `StandardFullResourceScheduler`
- `GPUEnvironmentConfig`
- `apply_worker_environment()`

## Recommended Merge Order

`main` へは次の順で持ち帰るのが安全です。

1. core module
1. unit tests
1. usage note / design doc
1. monitor
1. experiment script integration

理由:

- まず core API を安定化したい
- 次に regression test を持ち帰りたい
- そのあとで docs を合わせる方が文言がぶれにくい
- experiment script 側の修正は最後に薄く入れる方が conflict が少ない

## Pre-Merge Checklist

`main` へ持ち帰る前に満たしたい項目:

- `pytest -q python/tests/experiment_runner`
- `pyright python/experiment_runner python/tests/experiment_runner`
- `git diff --check`
- usage note が current API と一致している
- `documents/experiment_runner.md` が current implementation と一致している
- monitor の endpoint 記述が実装と一致している
- `StandardRunner` の fresh process 前提が docs に明記されている
- GPU env の責務が docs / notes で一貫している

## Main-Side Carry-Over Work

`main` に取り込んだら最初にやること:

1. `experiment_runner` 導入前提の branch とぶつかる実験 script を洗う
1. `context_builder["environment_variables"]` に危険 env を入れている script を特定する
1. それらを `GPUEnvironmentConfig` または runner wiring に寄せる
1. `smolyak_experiment` を移行の先行例にする
1. monitor を使う script と使わない script の境界を明記する

## Migration Targets

移行対象として分かりやすいもの:

- `experiments/smolyak_experiment/run_smolyak_experiment_simple.py`

この script では現在、`context_builder` 内で CPU / GPU に応じた env を作っています。

`main` へ持ち帰るときは、これを次に分解した方がよいです。

- device 選択の意図
- resource estimate
- allocator profile
- output / metadata context

## Acceptance Criteria

`main` で次が成立したら、ひとまず受け入れ完了と見てよいです。

- CPU 実験が `StandardRunner` で普通に回る
- GPU 実験が `StandardFullResourceScheduler` で普通に回る
- 実験 script が runtime env を直接触らなくても動く
- `python/tests/experiment_runner` が `main` 上で通る
- usage note を見れば新しい script が書ける

## Nice To Have After Merge

`main` へ取り込んだあと、余裕があればやると良いもの:

- dangerous env を `context_builder` から弾く guard
- allocator profile の命名済み preset
- `smolyak_experiment` 側の monitor 起動オプション
- `main` 用の短い onboarding example
- 重い GPU test を nightly か manual job に分ける運用

## Open

まだ `main` 取り込み時に決める必要があるもの:

- dangerous env を warning にするか exception にするか
- allocator profile を fully explicit にするか preset を持つか
- monitor を v1 でどこまで public API 扱いにするか
- `subprocess_scheduler` を legacy helper として残すか、benchmark 用正式 API として残すか

## Suggested Next Step

次の実務的な一手はこれです。

1. `main` へ持ち帰る commit 境界を決める
1. `smolyak_experiment` の env 直設定を整理する patch を別 commit に分ける
1. `main` 側で docs / tests / experiment integration を順に当てる
