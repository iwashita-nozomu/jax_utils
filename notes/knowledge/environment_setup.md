# Environment Setup

## Python

- 基本の実行は `PYTHONPATH=/workspace/python` を前提にする。
- `main` と worktree で code path が違うときは、どの tree を使っているかを明示する。
- 実験 child process には、親と同じ Python path を明示的に渡す。

## JAX / GPU

- GPU を固定したい child では `CUDA_VISIBLE_DEVICES` を明示する。
- GPU の先取りを避けたいときは `XLA_PYTHON_CLIENT_PREALLOCATE=false` を使う。
- CPU に逃がしたいときは `JAX_PLATFORMS=cpu` を使う。
- GPU child は、親の見えている全 GPU を暗黙に引き継がないようにする。

## CPU thread

- 実験 worker で hidden thread が暴れないように、必要なら
  - `OMP_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
  - `NUMEXPR_NUM_THREADS=1`
    を使う。
- CPU affinity を切るときは、worker 数と logical CPU 数を先に確認する。

## 実験前チェック

- `main` が clean か確認する。
- 対象 results branch が `origin` と同期しているか確認する。
- 実験に使う worktree が clean か確認する。
- 出力先 directory が期待した branch 上にあるか確認する。

## よくある失敗

- child process 側で `cuda` backend 初期化に失敗する。
- 実験と編集で違う tree を見ていて、思ったコードが走っていない。
- CPU thread 数を放置して oversubscription が起きる。
