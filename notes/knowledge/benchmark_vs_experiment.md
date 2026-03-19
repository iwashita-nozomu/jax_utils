"""Benchmark と Experiment の使い分け

このファイルは、開発過程で「いつどの測定を使うか」を明確にするための指針。

## 定義

### Benchmark（ベンチマーク）
- **目的**: 単一環境での性能比較（同じ機械で再現可能）
- **スコープ**: 特定の条件下での基本的な性能（init time, integral time）
- **実行時間**: 秒～分（全テスト合わせて数分以内）
- **実装**: `python/benchmark/functional/benchmark_*.py`
- **出力**: JSON（手軽に分析可能）
- **用途**: 
  - 実装変更の前後比較
  - 次元による性能スケーリング確認
  - CI/CD での性能劣化検知

### Experiment（実験）
- **目的**: 複数条件・複数 dtype での详细デ ータ収集
- **スコープ**: 精度・収束性・スケーリング限界の探索
- **実行時間**: 分～時間（Timeout 対応必須）
- **実装**: `experiments/functional/` 配下の Python + `experiment_runner`
- **出力**: JSONL（逐次保存、途中停止対応）
- **用途**:
  - 新しい実装方針の妥当性検証
  - 高次元での性能限界を探る
  - 誤差・精度傾向の詳細理解

## 使い分けチャート

```
┌─ パフォーマンスを確認したい
│  └─ 数秒以内に結果が欲しい ─→ **Benchmark** 使用
│     (次のステップの前後比較、簡易検証)
│
└─ 複数条件を詳しく調べたい
   └─ 数分～数時間かかっても OK ─→ **Experiment** 使用
      (実装方針の妥当性、限界探索)
```

## Smolyak 積分器での例

### ✅ Benchmark を使うべき例
- 「NumPy/JAX 変換を統一したら性能改善した？」
  → `python/benchmark/functional/benchmark_smolyak_integrator.py` で初期化時間を計測

- 「次元が増えると初期化時間はどうなる？」
  → `benchmark_initialization_scaling()` で d=1~10 を計測

### ✅ Experiment を使うべき例
- 「新しい term plan 遅延化設計は本当に改善するのか」
  → `experiments/functional/smolyak_scaling/` で多条件・多精度を探索

- 「float16 と float64 の精度失敗点を詳しく比較したい」
  → `experiment_runner` で dtype×dimension×level を組み合わせ

## 現在の Smolyak 改善での位置づけ

### Phase 1: 実装最適化（Benchmark）← 今ここ
```
baseline (commit 64fb00b) 
  ↓
init time 計測（benchmark）
  ↓
改善実装（遅延化 term plan など）
  ↓
改善後 benchmark で再計測 ← 性能向上確認
```

### Phase 2: 新設計の妥当性検証（Experiment）← 次のフェーズ
```
改善実装が OK なら
  ↓
高次元での限界を experiment で探索
  ↓
精度・収束性を詳細分析
  ↓
main に統合 OK か判断
```

## Notes への出力ポイント

### benchmark からの知見
- ベンチマーク結果 JSON を `notes/knowledge/` に軽く保存
- 傾向のスナップショット（「このコミットでこんなに改善した」）

### experiment からの知見
- `notes/themes/` に詳細な分析
- frontier, error scaling, 精度限界
- 文献との比較

## 参考

- Benchmark スクリプト: [python/benchmark/functional/](../../../python/benchmark/functional/)
- Experiment ツール: AGENT_USAGE.md を参照

"""
