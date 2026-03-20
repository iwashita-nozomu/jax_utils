"""HLO ダンプ・解析ユーティリティパッケージ。

XLA Higher-level Optimizer (HLO) IR をダンプ・パース・解析し、
JAX プログラムの低レベル計算グラフを可視化・最適化します。

主要機能:
    dumper: HLO text/proto フォーマット抽出
    parser: HLO 解析・構造表現
    analyzer: 計算グラフの統計・最適化機会抽出
    visualizer: グラフ可視化

用途:
    - パフォーマンスプロファイリング
    - オペレーション融合の機会発見
    - メモリ使用量最適化
    - 並列化戦略の検討

参考資料:
    - [XLA HLO Documentation](https://www.tensorflow.org/xla)
    - [JAX Profiling Guide](https://jax.readthedocs.io/)
"""

from .dump import dump_hlo_jsonl

__all__ = [
    "dump_hlo_jsonl",
]
