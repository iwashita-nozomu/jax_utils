# 設計ドキュメント

このディレクトリは、プロジェクトの高レベル設計と実装ガイドを集約します。

---

## 📐 階層構造

```
documents/design/
├── README.md (このファイル) ← 最上位ナビゲーション
├── protocols.md ← 全体の Protocol 設計
├── base_components.md ← BASE層の型・Protocol・クラス設計
├── experiment_runner.md ← standalone experiment runner の設計
│
└── jax_util/
    ├── README.md ← jax_util 全体ナビゲーション
    ├── base.md
    ├── solvers.md
    ├── optimizers.md
    ├── functional.md
    ├── neuralnetwork.md
    └── hlo.md
```

---

## 🔗 ナビゲーション（3参照ルール準拠）

### 最上位階層

- **[protocols.md](./protocols.md)** — Protocol の型パラメータ化戦略
  - 5つの Protocol レイヤ（base → functional/optimizers/neuralnetwork）
  - 依存関係図（Mermaid）
  - 実装ガイドライン

- **[base_components.md](./base_components.md)** — BASE層の基礎設計
  - 型エイリアス（Scalar, Vector, Matrix）
  - 共通 Protocol
  - クラス設計

### jax_util 実装ガイド

- **[jax_util/README.md](./jax_util/README.md)** ← 中間ハブ
  - 各モジュール（base, solvers, optimizers, functional, neuralnetwork）へのリンク
  - モジュール間依存関係

**個別モジュール設計** (jax_util/ 配下):

| モジュール | ファイル | 責務 |
|-----------|---------|------|
| base | [base.md](./jax_util/base.md) | 型定義・基盤 Protocol |
| solvers | [solvers.md](./jax_util/solvers.md) | 数値ソルバー実装 |
| optimizers | [optimizers.md](./jax_util/optimizers.md) | 最適化アルゴリズム |
| functional | [functional.md](./jax_util/functional.md) | 関数型最適化 |
| neuralnetwork | [neuralnetwork.md](./jax_util/neuralnetwork.md) | NN の層別訓練 |
| hlo | [hlo.md](./jax_util/hlo.md) | HLO 解析・最適化 |

**独立モジュール設計**:

| モジュール | ファイル | 責務 |
|-----------|---------|------|
| experiment_runner | [experiment_runner.md](../experiment_runner.md) | 実験実行制御 |

---

## 📊 参照ルール

本プロジェクトは**「3参照以内ルール」**を採用しています：

- **最上位** → **中間ハブ** → **個別設計** で最大 3 階層
- 各ハブは以下層への参照リストのみ提供
- 詳細設計は対応ファイルで自己完結

**例**:

```
README.md (top)
  ↓ (1参照)
documents/design/README.md (中間ハブ)
  ↓ (2参照)
documents/design/jax_util/README.md (下位ハブ)
  ↓ (3参照)
documents/design/jax_util/solvers.md (詳細)
```

---

## 🎯 各ドキュメントの更新ルール

### 参照パス修正時

実装変更が発生した際の対象ドキュメント：

| 変更内容 | 更新対象 | パス |
|---------|---------|------|
| BASE 型・Protocol 変更 | base_components.md | `documents/design/base_components.md` |
| サブモジュール API 変更 | モジュール設計書 | `documents/design/jax_util/<module>.md` |
| Protocol 戦略変更 | protocols.md | `documents/design/protocols.md` |

### 参照リンク記法

```markdown
# ✅ 正確な相対パス
[protocols.md](./protocols.md)
[base.md](./jax_util/base.md)
[TEAM_IMPLEMENTATION_GUIDE.md](../TEAM_IMPLEMENTATION_GUIDE.md)

# ❌ 古い参照（廃止）
documents/design/apis/solvers.md  ← 存在しない
```

---

## 📚 関連ドキュメント

### 実装ガイド（documents/）

- [TEAM_IMPLEMENTATION_GUIDE.md](../TEAM_IMPLEMENTATION_GUIDE.md) — 品質改善フェーズ 1-3
- [DOCSTRING_GUIDE.md](../DOCSTRING_GUIDE.md) — Docstring 実装方法
- [CODE_REVIEW_REPORT_2026-03-20.md](../CODE_REVIEW_REPORT_2026-03-20.md) — 品質スコア詳細

### コーディング規約（documents/coding-conventions-*.md）

- [coding-conventions-project.md](../coding-conventions-project.md) — プロジェクト全体方針
- [coding-conventions-python.md](../coding-conventions-python.md) — Python 規約
- [coding-conventions-testing.md](../coding-conventions-testing.md) — テスト規約

---

## ✅ このファイルの役割

- **最上位ナビゲーション**: すべての設計ドキュメントへの単一エントリポイント
- **参照ルール明示**: 「3参照ルール」の在り方を説明
- **リンク一覧**: IDE のクイックリンク・ホバー支援
- **メンテナンス性向上**: リンク死を防ぐため中央管理

---

**作成日**: 2026-03-20  
**目的**: 設計ドキュメント体系の統一化・参照ルール準拠  
**状態**: ✅ 現在稼働中

最新の設計方針は各モジュール設計書を参照してください。
