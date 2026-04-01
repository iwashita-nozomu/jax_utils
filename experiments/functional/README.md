# 機能実験・テスト検証（Functional Experiments）

機能的な検証、単体テスト拡張、非機能要件の評価などを行う実験ディレクトリ。

______________________________________________________________________

## 目的

- **新機能の基本動作確認** — 実装直後の単体検証
- **回帰テスト実行** — 主要アルゴリズムの保湿性確認
- **性能基準の確認** — 速度・メモリ・精度の閾値確認
- **仕様適合性検証** — 要件書との一致確認

______________________________________________________________________

## ファイル構成

各実験は `<experiment_name>/` ディレクトリで管理：

```
functional/
├─ <experiment_name>/
│  ├─ experiment.py (実験メインコード)
│  ├─ config.yaml (パラメータ設定)
│  ├─ results.json (実行結果)
│  └─ analysis.md (分析・結論)
├─ README.md (ここ)
```

______________________________________________________________________

## 実験の記録

実験の詳細ログは [notes/experiments/](../../notes/experiments/) および [notes/themes/](../../notes/themes/) で管理。

実験を新規実施した際は、以下のコマンドからテンプレートを利用してください：

```bash
PYTHONPATH=/workspace/python python -m experiment_runner \
  --experiment-dir ./experiments/functional/<name> \
  --config config.yaml \
  --output-dir ./experiments/functional/<name>/results
```

詳細は [documents/research-workflow.md](../../documents/research-workflow.md) を参照。

______________________________________________________________________

**関連:** [experiments/smolyak_experiment/](../smolyak_experiment/README.md) • [notes/experiments/](../../notes/experiments/README.md)
