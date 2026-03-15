# 責務分離

この章は、ディレクトリごとの責務分離だけを定めます。

## 要約

- この章では、どのディレクトリに何を置くかだけを定めます。
- クラスや Protocol の詳細設計は、この章では扱いません。
- API 詳細設計は、この章では扱いません。

## 規約

- `python/jax_util/base/`: 型・定数・作用素・環境設定を置きます。
- `python/jax_util/hlo/`: HLO ダンプと解析補助だけを置きます。
- `python/jax_util/solvers/`: 数値ソルバと、その計算に直接関わる補助を置きます。
- `python/jax_util/optimizers/`: `solvers` を利用する最適化アルゴリズムを置きます。
- `python/jax_util/neuralnetwork/`: 実験段階の実装を置き、安定 API 文書へは混ぜません。
- `python/jax_util/solvers/archive/`: 現在使わない実装の保管場所とし、新規実装や修正の対象外とします。
- `python/tests/`: テストだけを置きます。
- `scripts/`: 実行補助とログ整形だけを置きます。
- `documents/`: 規約と設計書の一次情報源とします。
