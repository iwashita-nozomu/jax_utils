# Smolyak 実験：ブランチ戦略の決定

## 背景

`work/smolyak-improvement-20260318` worktree で以下を達成した：

- 実験インフラ整備：spawn context、リソース監視、Pyright 0 件
- smoke test: 6 cases in 3.1 s/sec ✅
- small test: 50 cases in 10.4 cases/sec ✅

次フェーズで medium/large test を実行する際の branch 戦略を決定する。

## 方針：段階的実験 + 最後に統合

### ブランチ構成

| 段階 | test size | Branch | Worktree | 結果保管 | 用途 |
|:---|:---:|:---|:---|:---|:---|
| 検証 | smoke | `work/smolyak-improvement-20260318` | active | notes/diary | パフォーマンス基準確認 |
| 検証 | small | `work/smolyak-improvement-20260318` | active | notes/experimental/ | リソース監視確認 |
| 本実行 | medium | `results/smolyak-experiment-20260318` | 新規作成 | results branch | スケーリング測定（準本実） |
| 本実行 | large | `results/smolyak-experiment-20260318` | 同一 | results branch | スケーリング測定（本実） |

### 理由

1. **smoke/small は高速検証**
   - 数秒～数十秒で完了
   - パフォーマンス基準（スループット、リソース使用率）確認
   - 問題検出時に即座に修正できる
   - 旧版との性能比較用途として notes へ記録

2. **medium/large は本実運用**
   - 数分～数十分の長時間実行
   - results branch 所有で独立した記録保管
   - 複数回の本実運用（再チューニング等）に備える
   - 最終判断を経てから main へ統合

## 実行手順

### Phase 1: 検証（current worktree）

```bash
# smoke/small test は既に完了

# 今後の追加検証がある場合はこれで
timeout 120 python3 experiments/smolyak_experiment/run_smolyak_experiment.py --size smoke
timeout 180 python3 experiments/smolyak_experiment/run_smolyak_experiment.py --size small
```

結果は以下に記録：
- `experiments/smolyak_experiment/results/smoke/results_*.json`
- `experiments/smolyak_experiment/results/small/results_*.json`

### Phase 2: 本実運用（新規 worktree）

a) **results branch 作成**

```bash
# main に戻す
git checkout main
git pull origin main

# results branch を作成
git checkout -b results/smolyak-experiment-20260318

# .worktree 向け WORKTREE_SCOPE を追加
cat > WORKTREE_SCOPE.md << 'EOF'
# WORKTREE_SCOPE — results/smolyak-experiment-20260318

- **役割**: Smolyak 実験の medium/large test 本実運用
- **編集対象**: experiments/smolyak_experiment/results/
- **コミット対象**: JSON/JSONL 実験結果と集約レポート
- **実行環境**: Docker CPU-only, 4 parallel workers
- **期間**: 2026-03-20 本実開始
EOF

git add WORKTREE_SCOPE.md
git commit -m "docs: results branch WORKTREE_SCOPE 追加"
git push origin results/smolyak-experiment-20260318
```

b) **新 worktree を作成**

```bash
# 新 worktree を切る
git worktree add -b results/smolyak-experiment-20260318 \
  ./.worktrees/work-smolyak-experiment-results-20260318 \
  results/smolyak-experiment-20260318

cd ./.worktrees/work-smolyak-experiment-results-20260318
```

c) **medium/large test 実行**

```bash
# medium test
timeout 600 python3 experiments/smolyak_experiment/run_smolyak_experiment.py --size medium

# large test （必要に応じて）
# timeout 1800 python3 experiments/smolyak_experiment/run_smolyak_experiment.py --size large
```

d) **コミット・プッシュ**

```bash
# 結果ファイルとレポート
git add experiments/smolyak_experiment/results/
git commit -m "experiment: medium test results (smolyak-experiment-20260318)"
git push origin results/smolyak-experiment-20260318

# 集約レポート作成（オプション）
# experiments/smolyak_experiment/generate_report.py など
```

### Phase 3: main への統合（決定後）

実験結果が満足できるものになったら、以下の流れで main へ統合：

```bash
# results branch の最新コミット hash を取得
git log results/smolyak-experiment-20260318 --oneline -1

# main で結果を参照可能にする（note リンク等）
git checkout main
git pull origin main

# results branch の実験結果は notes へ記録
cat >> notes/experiments/smolyak_experiment_20260320.md << 'EOF'
## Results Summary (2026-03-20)

- branch: results/smolyak-experiment-20260318
- medium test: X cases, Y cases/sec
- large test: ...

参照: git log results/smolyak-experiment-20260318
EOF

git add notes/experiments/smolyak_experiment_20260320.md
git commit -m "docs: Smolyak 実験結果(20260320) を notes へ記録"
git push origin main
```

## 記録ポイント

### work/ worktree での記録（notes/experiments/）

- パフォーマンス基準
- smoke/small での問題・修正
- リソース監視の所見

例：`notes/experiments/smolyak_infrastructure_20260320.md`

### results/ branch での記録

- JSON/JSONL の完全結果
- 集約統計情報
- worktree 内 WORKTREE_SCOPE.md

## 判断基準

### main へ統合するかの判断

- [ ] medium test のスループットが smoke/small と一貫性を持つか確認
- [ ] リソース監視で想定外の問題（OOM、deadlock等）が発生していないか確認
- [ ] 結果の再現性が 2 回以上の実行で確認されたか
- [ ] 実験結果が当初の仮説（スケーリング特性）を支持するか

判断後、notes/experiments/ へ最終レポートを記録し、results branch を archived 状態へ。

## Next Step

1. **smoke/small test の結果を notes へ記録**
2. **medium test 用 worktree を新規作成** → results/smolyak-experiment-20260318
3. **medium test 実行**
4. **結果を notes へ記録 + 最終判断**
