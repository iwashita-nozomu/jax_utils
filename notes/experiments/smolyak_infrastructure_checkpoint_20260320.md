# Smolyak 実験基盤：検証結果（2026-03-20）

## サマリー

JAX fork() 互換性の spawn context 導入と、StandardRunner/StandardFullResourceScheduler による並列実行インフラの整備完了。

| メトリクス | smoke | small | 備考 |
|:---|:---:|:---:|:---|
| ケース数 | 6 | 50 | |
| 実行時間 | 1.6-1.9 s | 4.8 s | 複数回実行 |
| スループット | 3.1-3.7 cases/sec | 10.4 cases/sec | 4 parallel workers |
| Pyright | 0 errors | 0 errors | 規約準拠 ✅ |
| リソース監視 | ✅ | ✅ | Max workers=4, host_memory=16GB |

## 技術的成果

### 1. JAX Fork() 互換性（✅ 完了）

**問題**: JAX は import 時に JIT 初期化やメモリ先取りを行うため、fork() ベースの multiprocessing で child プロセス状態が不確定になる。

**実装**:
- `jax_context.py`: spawn context の統一的な factory
- `disable_jax_memory_preallocation()`: CPU-only でも安全に動作
- 常時 spawn context 使用（フラグなし、シンプル化）

**確認**:
- smoke test: 6 cases を 4 parallel workers で 3.1 cases/sec
- 実行回による変動が小さい（Std 0.72 ms in init time）

### 2. Worker Pickle 化検証（✅ 完了）

**問題**: ProcessPoolExecutor に送出する worker が pickle 化不可能だと、worker pool 作成時に失敗する。

**実装**:
- `check_picklable()` 関数で事前検証
- 不可能な場合は起動前エラーで検出

**効果**: 設定エラーを worker pool 起動前に捕捉 → デバッグ時間短縮

### 3. リソース監視スケジューラ（✅ 動作確認）

**実装**:
- `StandardFullResourceScheduler`: host memory と parallel worker を同時監視
- resource estimate に基づいたスケジューリング
- 動的なケース投入制御

**確認**:
- small test で 50 cases を安定実行
- resource capacity: max_workers=4, host_memory=16GB を正確に反映
- 完了 context が記録される

### 4. JSONL チェックポイント（✅ 動作確認）

**実装**:
- 各ケース結果を JSONL 形式で逐次追記
- fcntl locking で排他制御
- 途中中断後の再開対応

**確認**:
- smoke/small で正常に生成
- ファイルサイズ: smoke 6 cases = 3.2 KB, small 50 cases = 25 KB

### 5. Pyright 規約準拠（✅ 0 エラー）

**修正**:
- 未使用 import 削除
- Worker protocol 型指定完全化
- multiprocessing type hint 対応（type: ignore）

**結果**: `pyright 0 errors, 0 warnings, 0 informations`

## 性能指標の分析

### Throughput（実行速度）

```
smoke test:  3.1-3.7 cases/sec (平均 3.4)
small test:  10.4 cases/sec
```

**解釈**:
- small test の dimension 1-5, level 1-5 は smoke より重いケースが多いが、4 parallel で効率的に処理
- spawn context のオーバーヘッドは無視できるレベル（1 case = 200-300 ms）

### Initialization 時間（JAX startup）

```
smoke/small 合計統計:
  Min: 1.52 ms
  Max: 16.23 ms
  Mean: 7-10 ms
  Std: 0.72-3.97 ms
```

**解釈**:
- 2 回目以降の JIT compile は cache されている
- dimension 1 の初回 init が若干遅い（10-16 ms）は妥当
- CPU-only でのキャッシュ効率が良好

### Integration 時間（Smolyak 計算本体）

```
smoke/small 合計統計:
  Min: 60-90 ms
  Max: 130-200 ms
  Mean: 112-120 ms
```

**解釈**:
- level が高いほど計算時間が増加（expected）
- dimension ごとの variance は小さい（キャッシュ効率良好）

## 課題と留意点

### 1. CPU-only での検証

✅ **確認済み**: disable_jax_memory_preallocation(gpu_devices=False) で動作

next: GPU 環境での検証は別 worktree にて

### 2. Medium/Large test スケーリング

- smoke: 6 cases, small: 50 cases
- medium: 50-100 cases? large: 100+ cases?

proposal: BRANCH_STRATEGY_SMOLYAK_EXPERIMENT.md を参照

### 3. 記録・再現性

✅ **JSONL checkpoint**: 途中中断からの再開が可能

記録方針:
- 実験ごとに `experiments/smolyak_experiment/results/<size>/results_<timestamp>.{json|jsonl}`
- 集約統計は JSON に保存
- per-case 結果は JSONL で保存

## 次のステップ

### Phase 1（完了）: 基盤整備 ✅

- [x] spawn context factory 実装
- [x] Worker pickle 化検証
- [x] Resource scheduler 統合
- [x] Pyright 0 件達成
- [x] smoke/small smoke test 検証

### Phase 2（提案）: 本実運用

- [ ] medium test（新 results branch）
- [ ] large test（同一 results branch）
- [ ] 実験結果の分析・レポート化

### Phase 3（検討中）: 次の改良

- [ ] GPU parallelization（別 worktree）
- [ ] performance tuning
- [ ] ドキュメント化

## 参照

- Implementation: `python/jax_util/experiment_runner/jax_context.py`, `runner.py`
- Experiment script: `experiments/smolyak_experiment/run_smolyak_experiment.py`
- Branch strategy: `documents/BRANCH_STRATEGY_SMOLYAK_EXPERIMENT.md`
- Data: `experiments/smolyak_experiment/results/smoke/`, `results/small/`
