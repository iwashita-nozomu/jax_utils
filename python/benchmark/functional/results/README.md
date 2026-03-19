# ベンチマーク結果ディレクトリ

このディレクトリには，Smolyak 積分器の性能ベンチマーク結果が格納されます。

---

## ファイル構成

### 📊 レポート

| ファイル | 形式 | 用途 |
|---------|------|------|
| **BENCHMARK_REPORT.md** | Markdown | GitHub 向け，版管理可能 |
| **BENCHMARK_REPORT.html** | HTML | ブラウザで即座に閲覧可能 |
| **BENCHMARK_SUMMARY.md** | Markdown | 実行サマリー＆推奨事項 |

### 📈 JSON データ

| ファイル | サイズ | 対象 | 内容 |
|---------|--------|------|------|
| `light.json` | ~10KB | d=1-8, level=1-5 | 初期化スケーリング，レベル精製，dtype 比較 |
| `heavy.json` | ~15KB | d=1-15, level=1-8 | 詳細な性能分析，指数爆発領域を含む |
| `extreme.json` | ~? | d=1-20, level=1-10 | 設計限界確認（実行中またはバックアップ） |

### 📋 CSV データ

| ファイル | 用途 |
|---------|------|
| `light_scaling.csv` | 次元スケーリング分析（Light） |
| `light_refinement.csv` | レベル精製分析（Light） |
| `heavy_scaling.csv` | 次元スケーリング分析（Heavy） |
| `heavy_refinement.csv` | レベル精製分析（Heavy） |

### 📝 ログ

| ファイル | 内容 |
|---------|------|
| `extreme.log` | Extreme ベンチマーク実行ログ（stdut/stderr） |

---

## 使用方法

### 1. レポートを閲覧

**HTML（推奨）**:
```bash
# ブラウザで開く
open BENCHMARK_REPORT.html
# または
firefox BENCHMARK_REPORT.html
```

**Markdown**:
```bash
# テキストエディタで開く
cat BENCHMARK_REPORT.md

# または GitHub/VS Code で表示
```

### 2. CSV データを分析

```bash
# Light の初期化スケーリング確認
cat light_scaling.csv

# Heavy の詳細レベル精製分析
head -20 heavy_refinement.csv
```

### 3. JSON から Python で分析

```python
import json

# Light を読み込み
with open("light.json") as f:
    light_data = json.load(f)

# 初期化スケーリング結果抽出
for benchmark in light_data["suite"]["benchmarks"]:
    if benchmark["benchmark"] == "initialization_scaling":
        for result in benchmark["results"]:
            d = result["dimension"]
            init_ms = result["init_time"]["mean_sec"] * 1000
            print(f"d={d}: {init_ms:.2f}ms")
```

### 4. レポート再生成

新しいベンチマークを追加実行後，レポートを再生成します：

```bash
cd /workspace/.worktrees/work-smolyak-improvement-20260318
python3 python/benchmark/functional/generate_report.py
```

---

## 主要な数値

### Light レベル
```
実行時間: ~30 秒
対象: d=1-8, level=1-5
初期化時間: mostly 2-4ms（d≥2）
目的: Quick verification, DevOps
```

### Heavy レベル
```
実行時間: ~5-10 分
対象: d=1-15, level=1-8  
初期化時間: d=1-12 は線形，d=13+ は指数爆発
目的: Detailed analysis, baseline comparison
```

### Extreme レベル
```
実行時間: ~1時間以上
対象: d=1-20, level=1-10
初期化時間: 超大規模問題での漸近性能
目的: Design limit confirmation
```

---

## 解釈ガイド

### dtype 比較の解釈

```
Light (d=4, level=2):
  float32: 83.35ms (83倍遅い!)
  float64: 2.23ms

Heavy (d=8, level=3):
  float32: 12.74ms (1.8倍遅い)
  float64: 7.08ms
```

**意味**: 小規模問題では float32 が顕著に遅い（理由未調査）

### 初期化スケーリングの解釈

```
d=1-12: 線形成長 ✓
d=13-15: 指数的領域 ⚠
```

**意味**: 
- d≤12 は十分に最適化
- d≥13 は term enumeration が支配的
- Phase 2 の lazy enumeration で改善可能

### レベル精製の解釈

```
level=1: 1 点, 初化 2.3ms
level=8: 11988 点, 初化 5.5ms
```

**意味**:
- 評価点数は指数成長 ($10^4$ まで観測)
- 初期化時間は線形成長 → 初期化ボトルネックではない
- ボトルネックは積分実行

---

## トラブルシューティング

### Q: Extreme がまだ実行中です

A: 1 時間以上かかる場合があります。進行状況を確認：
```bash
tail -f extreme.log
ps aux | grep benchmark
```

### Q: JSON ファイルが破損しています

A: ベンチマーク中に割り込まれた可能性があります。再実行：
```bash
python3 ../benchmark_smolyak_integrator.py --heavy --output heavy.json
```

### Q: CSV が空です

A: レポート生成スクリプトを再実行：
```bash
python3 ../generate_report.py
```

### Q: HTML にデータが表示されません

A: ブラウザキャッシュをクリアしてリロード

---

## 関連ドキュメント

- [ベンチマーク実行ガイド](../../documents/BENCHMARK_RUN_GUIDE.md)
- [ベンチマークレベル分析](../../notes/knowledge/benchmark_levels_analysis.md)
- [ベンチマーク vs 実験](../../notes/knowledge/benchmark_vs_experiment.md)
- [コーディング規約](../../documents/conventions/python/20_benchmark_policy.md)

---

## 履歴

- **2026-03-19 14:30**: Light・Heavy 完了，Extreme 実行開始
- **2026-03-19 14:34**: CSV レポート生成
- **2026-03-19 14:34**: HTML・Markdown レポート生成

