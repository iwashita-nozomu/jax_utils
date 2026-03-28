**実験ランナー使い方（サマリ）**

- **目的**: `StandardRunner` / `StandardScheduler` 系を使って実験ケースを並列実行するための軽量フレームワーク。
- **主なモジュール**: `python/jax_util/experiment_runner/protocols.py`, `python/jax_util/experiment_runner/runner.py`, `python/jax_util/experiment_runner/gpu_runner.py`, `python/jax_util/experiment_runner/resource_scheduler.py`。

**基本的な使い方**

- **Simple FIFO 実行（CPU）**:

```python
from jax_util.experiment_runner.runner import (
    StandardWorker, StandardScheduler, StandardRunner, StandardResourceCapacity
)

# ケースとタスク
cases = [1,2,3]

def task(case, ctx):
    # 実験ロジック
    return 0

scheduler = StandardScheduler(
    resource_capacity=StandardResourceCapacity(max_workers=4),
    cases=cases,
)
worker = StandardWorker(task)
runner = StandardRunner(scheduler)
runner.run(worker)
```yaml

- **GPU を使う場合**:

  - 環境変数 `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES` を設定して見える GPU を制御する。
  - `GPUResourceCapacity.from_environment()` と `StandardGPUScheduler` を使うと便利。
  - `gpu_runner.py` の `visible_gpu_ids_from_environment` が環境変数の解釈規則を提供する。
  - `StandardFullResourceScheduler` を使う場合は、GPU 関連の環境変数は `TaskContext["environment_variables"]` に入り、worker 側で `apply_environment_variables(context)` を呼んで反映する。

- **リソース自動検出**:

  - `FullResourceCapacity.from_system()` は `detect_max_workers()`, `detect_host_memory_bytes()`, `detect_gpu_devices()` を組み合わせて実行環境を推定する。
  - テスト時は `query_rows` / `environ` を注入して振る舞いを固定できます。

**テスト実行**

- 実験ランナー用テストは `python/tests/experiment_runner` にあります。
- 高負荷／重いテストは環境変数 `RUN_HEAVY_TESTS=1` で有効化されます。
- 既定のテスト実行:

```bash
pytest -q python/tests/experiment_runner
```yaml

**設計上の注意**

- モジュールはインポート時に副作用を持たないように設計されています（`nvidia-smi` 呼び出し等は検出関数で明示的に行う）。
- 公開 API は各サブモジュールの `__all__` で明示しています。
- `detect_max_workers()` は `os.cpu_count()` が不明な場合に例外を投げます。コンテナ等で不明な場合は明示的パラメータを渡してください。

**改善アイデア（短期〜中期）**

- **動的スケーリング**: 実行中に負荷や待ち行列を監視して `max_workers` を動的に拡大/縮小する（クラウド VM やコンテナオーケストレータ連携）。
- **優先度・プリエンプション**: 緊急ジョブを優先し、低優先度ジョブを一時停止または再スケジュールする機能。
- **予測ベース配置**: 過去の実行データからケースのリソース需要を学習し、より良い GPU/メモリ割当を行う。
- **GPU トポロジ対応**: NVLink/PCIe トポロジを考慮した割当で通信コストを低減する。
- **マルチノード分散スケジューラ**: 単一ノードではなくクラスタ上での分散スケジューリング（軽量マスター/ワーカープロトコル）。
- **シミュレーションモード / ドライラン**: スケジューリング決定を副作用なしに検証するモード（テスト／チューニング用）。
- **メトリクス & テレメトリ**: 実行ログ、リソース使用率、待ち時間を収集して可視化（Prometheus 等に吐く）。
- **プラグイン式リソース推定子**: ユーザ実装の `resource_estimate(case)` を簡単に差し替えられるインタフェース強化。
- **コンテナ / cgroup 対応のリソース検出**: コンテナ化された環境で実効メモリ/CPU 制限を正しく検出する実装。

**次のアクション提案**

- README や `docs/` に簡易チュートリアルを追加する。
- CI に重いテストを分離したワークフローを追加し、通常プルリクでは短時間テストのみ走らせる。
- 小さな PoC を一つ（例: GPU トポロジ考慮スケジューラのプロトタイプ）を実装して評価する。

______________________________________________________________________

ファイル: `notes/experiments/experiment_runner_usage.md`
