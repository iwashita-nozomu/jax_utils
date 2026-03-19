# ベンチマーク実行サマリー

**実行日**: 2026-03-19  
**対象**: Smolyak 積分器性能測定スイート

---

## 実行結果

### ✅ 完了したベンチマーク

| レベル | 状態 | 実行時間 | ファイル |
|--------|------|---------|---------|
| **Light** | ✅ 完了 | ~30秒 | `light.json` (11KB) |
| **Heavy** | ✅ 完了 | ~5-10分 | `heavy.json` (17KB) |
| **Extreme** | ⏳ 実行中 | ~1h+ | `extreme.json` (予測中) |

### 📊 出力ファイル一覧

**JSON データ**:
- `light.json` - Light レベルの詳細データ
- `heavy.json` - Heavy レベルの詳細データ
- `extreme.json` - Extreme レベル（実行中）

**レポート**:
- `BENCHMARK_REPORT.md` - Markdown 形式の包括的レポート
- `BENCHMARK_REPORT.html` - HTML 形式のレポート

**CSV データ**:
- `light_scaling.csv` - 次元スケーリング分析
- `light_refinement.csv` - レベル精製分析
- `heavy_scaling.csv` - 高精度の次元スケーリング分析
- `heavy_refinement.csv` - 高精度のレベル精製分析

---

## 主要発見

### 1. 初期化時間のスケーリング

**Light (d=1-8)**:
```
d=1: 127.6ms ← JIT warmup が支配的
d=2: 2.54ms
d=3-8: 2-4ms 範囲 ← ほぼ安定
```

**Heavy (d=1-15)**:
```
d=1-12: 2-7ms 範囲 ← 線形成長
d=13: 11.5ms
d=14: 21.3ms  ← 加速開始
d=15: 123.6ms ← 指数的領域
```

### 2. 指数爆発領域の特定

- **d≤12**: 線形スケーリング（初期化 < 10ms）
- **d=13-15**: 指数的領域への移行
  - d=14/d=13 比率: 1.86 倍
  - d=15/d=14 比率: 5.8 倍

### 3. dtype の性能特性

**float32 の初期化時間オーバーヘッド**:

| サイズ | float32 | float64 | 倍率 |
|--------|---------|---------|------|
| Light (d=4, l=2) | 83.35ms | 2.23ms | **37.4倍** |
| Heavy (d=8, l=3) | 12.74ms | 7.08ms | **1.80倍** |

**観察**: 小規模問題では float32 が不利，大規模では相対効果が減少

### 4. レベル精製の特性

**評価点数の増加** (d=3):
```
level=1: 1 点      → 初化 2.3-2.9ms
level=2: 10 点     → 初化 2.1-2.5ms
level=3: 52 点     → 初化 2.3-2.5ms
level=4: 196 点    → 初化 2.6-4.1ms
level=5: 619 点    → 初化 3.1ms
level=6: 1762 点   → 初化 3.7ms
level=7: 4698 点   → 初化 4.5ms
level=8: 11988 点  → 初化 5.5ms
```

**重要**: 評価点が $10^4$ を超えても初期化は線形成長 → **初期化ボトルネックではない**

---

## 推奨用途

### Light ベンチマーク
- **実行**: 毎回（CI/CD、pre-commit check）
- **目的**: 回帰検出、性能劣化の即時通知
- **コスト**: ~30秒 → DevOps 統合可能

### Heavy ベンチマーク
- **実行**: 週 1 回（定期ビルド時）
- **目的**: 改善検証、ベースライン比較
- **コスト**: 5-10分 → GitHub Actions compatible

### Extreme ベンチマーク
- **実行**: 月 1 回またはマイルストーン時
- **目的**: 設計限界確認、漸近性能分析
- **コスト**: 1+ 時間 → オフピーク実行推奨

---

## Phase 2 への示唆

### 初期化最適化の優先度
- **低優先度**: d≤12（既に最適化済み，余地少ない）
- **高優先度**: d≥13（指数爆発領域，改善効果大）

### 提案改善候補
1. **Lazy term enumeration** (d≥13 向け)
   - 現在: 全計数が初期化時に実行
   - 改善: 必要な term だけ遅延計数
   - 期待効果: d=15 で 100ms → 1ms

2. **dtype 選択の自動化**
   - 現在: float64 固定推奨
   - 改善: d<10 で float32 を試行

3. **GPU 活用の検討**
   - 現在: CPU compute + JAX JIT
   - 改善: GPU evaluation batching
   - 注意: 小規模問題では逆効果

---

## 次ステップ

### 短期 (1-2週間)
- Extreme ベンチマーク完了待ち
- d=13-15 での詳細分析
- Phase 2 design document 作成

### 中期 (3-4週間)
- Lazy enumeration 実装
- Extreme ベースラインとの比較
- High-d 専用テスト

### 長期 (1-3ヶ月)
- GPU 活用への移行検討
- 月次 Extreme 実行スケジュール化
- ベンチマーク CI/CD 統合

---

## レポート生成方法

```bash
# Light・Heavy 実行後、常にレポート生成
cd python/benchmark/functional
python3 generate_report.py

# 出力:
# - BENCHMARK_REPORT.md (Markdown)
# - BENCHMARK_REPORT.html (HTML)
# - *_scaling.csv
# - *_refinement.csv
```

---

## 注記

- Extreme は実行中（ログファイル: `extreme.log`）
- 完了後は自動的に JSON に保存される
- レポート再生成時に自動的に Extreme データが含まれる

