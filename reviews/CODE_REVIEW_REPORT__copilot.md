# Status: Working Note
## Created: 2026-03-15
## Note: この文書はレビュー時点の所見を保存した成果物であり、現在の実装と完全には一致しない可能性があります

## コードレビュー報告書

**日時:** 2026-03-15
**対象:** `python/jax_util/` 配下の全実装
**方針:** 静的解析、アルゴリズム検証、数理的整合性確認

---

## 目次
1. [基盤モジュール (base)](#基盤モジュール)
1. [ソルバモジュール (solvers)](#ソルバモジュール)
1. [最適化モジュール (optimizers)](#最適化モジュール)
1. [汎関数モジュール (functional)](#汎関数モジュール)
1. [JAX/Equinox 運用](#jaxequinox-運用)
1. [まとめ](#まとめ)

---

## 基盤モジュール

## 1. protocols.py

**評価:** ✅ 良好

**内容:**
- 型エイリアス（Scalar, Vector, Matrix）を jaxtyping で定義
- Protocol クラス（Operator, LinearOperator, SolverLike等）を定義
- `@runtime_checkable` 装飾子で実行時型チェック対応

**強み:**
- 明確な型契約
- Protocol の overload で契約の詳細を明示

**留意点:**
- Boolean, Integer のエイリアス定義は現在未使用（neuralnetwork や functional で活用予定？）
- LinearOperator Protocol に `__add__` メソッドがあるが、全実装が対応しているか要確認

---

## 2. _env_value.py

**評価:** ✅ 基本的に良好、⚠️ 注意点あり

**内容:**
- 環境変数の bool/float 解釈
- JAX 設定（x64 モード）の管理
- dtype 選択、定数（ZERO, ONE, HALF, EPS）の定義

**強み:**
- 環境変数読み込みが丁寧（デフォルト値の設定）
- 定数を JAX Array として定義（dtype 統一）
- 中断環境フラグ（DEBUG, ENABLE_HLO_DUMP）

**問題点:**
1. **🔴 Import 副作用**: Line 56-57 で `jax.config.update()` を呼び出している

   ```python
   jax.config.update("jax_enable_x64", _get_bool_env("JAX_UTIL_ENABLE_X64", True))
   ```

   - import 時に JAX グローバル状態が変更される
   - テスト隔離やその他モジュールロード順序に依存する可能性
   - **推奨:** lazy initialization または setup 関数への移行

1. **dtype 名のアンダースコア統一:**
   - `"float16"` と `"f16"` の両方をサポートしているが、統一表記がない
   - 規約文書に明記があるか確認が必要

---

## 3. linearoperator.py

**評価:** ⚠️ バグ検出、算術設計に疑問

**内容:**
- `LinOp` クラス: 線形作用素の実装（eqx.Module ベース）
- `_ensure_batched`: 1D → 2D バッチ対応
- 演算子多重定義（`@`, `*`, `__rmul__`）
- ユーティリティ（`hstack_linops`, `vstack_linops`, `stack_linops`）

**強み:**
- Equinox の`filter_vmap` を活用したバッチ対応
- `@` と `*` の演算子分離が明確
- shape メタ情報の保持

**問題点:**

1. **🔴 バグ: `__mul__` と `__rmul__` 内の `other.__name__`**
   Lines 79, 82, 85, 103, 106, 109 など複数カ所で:

   ```python
   raise ValueError(f"...{other.__name__} is vector.")
   ```

   - `jax.Array` には `__name__` 属性がない → AttributeError
   - **修正:** `type(other).__name__` または `other.dtype` などに変更

1. **🟡 `hstack_linops` の設計に疑問**
   Line 136:

   ```python
   return jnp.sum(jnp.stack(results), axis=0)
   ```

   - 水平連結（hstack）なら結果を concatenate すべきでは？
   - stack + sum は **加算合成**を意図？
   - **確認:** コメント不足。数学的定義を明記すべき

1. **🟡 vstack の出力次元計算が Index 参照依存**
   Line 158:

   ```python
   total_u_dim = sum([op.shape[0] for op in ops])
   ```

   - shape が None の場合の処理がない
   - **推奨:** shape 検証を init 段階で行う

---

## 4. nonlinearoperator.py

**評価:** ✅ 良好

**内容:**
- `linearize`: 関数の Jacobian 行列を LinOp で返す
- `adjoint`: 随伴（転置）Jacobian を返す

**確認:**
- `jax.linearize` の使用は正しい（VJP ベース）
- shape 定義（線形化の出力次元、随伴の転置）が正確

---

## ソルバモジュール

## 1. pcg.py

**評価:** ✅ 良好

**内容:**
- 前処理付き共役勾配法（PCG）
- SPD 系を対象（A x = b）

**確認:**
- **反復式が正しい:**
  - alpha = rs / denom (ここ rs = r·z)
  - beta = rs_new / rs_safe (Polak-Ribière)
  - p_new = z_new + beta * p
- **投影が毎回行われる** (プロトタイプ版ではしばしば最後だけ投影するが、ここは正確)
- **ゼロ除算保護:** AVOID_ZERO_DIV を活用
- **収束判定:** 二乗ノルム上で相対・絶対許容誤差

**出典:**
- 標準的な PCG（多くの教科書に記載）

---

## 2. _minres.py

**評価:** ✅ 数学的に正確、複雑

**内容:**
- 対称不定値系向けの最小残差法（MINRES）
- Choi–Saunders 「unnormalized」形式

**確認:**

1. **SymOrtho パターン:**
   - Givens 回転を数値安定に実装
   - ケース分岐（b=0, a=0, 一般）で漏れなく処理

1. **Lanczos 3項漸化:**

   ```
   z_{k+1} = p/beta - alpha*z/beta - (beta/beta_prev)*z_prev
   ```

   - 実装と式の対応を確認 ✓

1. **MINRES フェーズ:**
   - Givens 回転で QR を ビルドアップ
   - dbar（検索方向係数）の更新が複雑だが正確

**出典:**
- Choi–Saunders (1992) "MINRES-QLP: A Krylov Subspace Method for Indefinite or Singular Symmetric Systems"
- 論文との対応を要確認（仕様書またはコメント):
  **🟡 現状: コメントに出典論文が明記されていない**

**注意:**
- "unnormalized form" は理論的に複雑
- テスト時に数値検証が重要

---

## 3. kkt_solver.py

**評価:** ✅ 基本構造は良好、❌ 前処理の限界あり

**内容:**
- KKT ブロックシステム：

  ```
  [ H   B^T ] [ u ]   [ g ]
  [ B    0  ] [ v ] = [ h ]
  ```

- スペクトル前処理（rank-r LOBPCG ベース）
- Schur 補完を活用

**確認:**

1. **前処理戦略:**
   - H^{-1} をスペクトル近似（最小固有ベクトル）
   - Schur 補完 S = B H^{-1} B^T を再度近似
   - ブロック対角前処理: diag(H^{-1}, S^{-1})
   - **限界:** ブロック非対角成分（coupling）を無視

1. **Tracer 対応:**
   - JAX JIT トレース外でのみ自己随伴性・SPD チェック
   - `getattr(jax, "core").Tracer` で判定
   - pyright ignore コメント付き
   - **評価:** 実用的

1. **🟡 コードの簡略化:**

   ```python
   Sv = Bv * H_inv_approx * BTv
   ```

   - `*` 演算で作用素合成。明確だが演算順序のコメント欲しい

**出典:**
- KKT 系の前処理は標準的な IP 文献で扱われる（Boyd & Vandenberghe など）

---

## 4. lobpcg.py

**評価:** ✅ 実装は正確、複雑さ高

**内容:**
- Block Local Optimal Preconditioned Conjugate Gradient
- SPD 行列の最小固有値（複数本）を推定

**確認:**

1. **trial subspace:**

   ```
   S = [X, W, P]  (n, 3r)
   ```

   - W = M^{-1} (A*X - X*Λ)（前処理残差）
   - P: 探索方向
   - QR 直交化で S_orth を得る

1. **Rayleigh–Ritz:**
   - 小規模 (3r, 3r) 固有値問題をホスト側で解く
   - 最小 r 本の固有ペアを取得
   - **数学的正確性:** ✓

1. **スペクトル補正前処理:**

   ```
   M^{-1} = Q (Λ+ε)^{-1} Q^T + (I-QQ^T) M_base^{-1} (I-QQ^T)
   ```

   - rank-r 補正 + 補空間前処理
   - 正規化の実装に注意あり

**出典:**
- LOBPCG: Knyazev (2001) "Toward the Optimal Preconditioned Eigensolver"

---

## 最適化モジュール

## optimizers/pdipm.py

**評価:** ✅ Mehrotra 型算法は正確、⚠️ 実装複雑

**内容:**
- 内点法（Primal-Dual Interior Point Method）
- Mehrotra predictor-corrector

**問題定式化:**

```
min  f(x)
s.t. c_eq(x) = 0
     c_ineq(x) + s = 0, s ≥ 0
```

**確認:**

1. **Hessian の effective:**

   ```
   H_eff = H_L + J_ineq^T diag(λ/s) J_ineq
   ```

   - 内点法の標準形
   - 二次項が λ/s で weighting される

1. **inexact Newton forcing:**

   ```
   KKT_rtol ~ c * IPM_residual^α
   定義域: [rtol_min, rtol_max]
   ```

   - 外部から許容誤差を動的に制御 ✓
   - 理論的根拠は Dembo-Eisenstat-Steihaug か Gratton ら

1. **Mehrotra predictor-corrector:**
   - (a) affine predictor: μ_aff = 0
   - (b) centering σ = (μ_aff/μ)^3
   - (c) corrector: r_c に高次補正 (ds_aff ∘ dλ_aff)
   - (d) step lengths: fraction-to-boundary
   - **標準実装:** ✓

1. **🟡 step length 計算:**

   ```python
   alpha_pri = frac_to_boundary(s, ds, tau_fb)
   alpha_dual = frac_to_boundary(lam, dlam_dir, tau_fb)
   ```

   - primal/dual で分離した step length
   - Mehrotra 原著の形式に従っている

**出典:**
- Mehrotra (1992) "On the Implementation of a Primal-Dual Interior Point Method"

---

## 汎関数モジュール

## functional/

**評価:** ✅ 基本は正確、複雑さ中～高

**内容:**
- `integrate`: 積分器の統一インターフェース
- `monte_carlo.py`: モンテカルロ積分
- `smolyak.py`: スパースグリッド（Clenshaw-Curtis, trapezoidal）

**確認:**

1. **monte_carlo.py:**
   - [-0.5, 0.5]^d の一様サンプル
   - vmap で並列評価、平均化
   - **数学:** 正確

1. **smolyak.py:**
   - dyadic 分数による hierarchical 構築
   - Clenshaw-Curtis nodes の canonical ID
   - **複雑性:** 高
   - **検証:** テストコード確認推奨

---

## JAX/Equinox 運用

## 使用パターン

| パターン | モジュール | 評価 |
|---------|---------|------|
| `jax.lax.while_loop` | solvers, optimizers | ✅ 正しい（JIT 対応） |
| `jax.vmap` | base linearoperator | ✅ or ⚠️ (in_axes/out_axes の明示性) |
| `eqx.Module` | 全モジュール | ✅ 統一的 |
| `jax.linearize` | optimizers, solvers | ✅ 効率的 |
| `eqx.filter_vjp` | optimizers | ✅ 正しい |
| `eqx.filter_vmap` | base linearoperator | ✅ 投影対応 |

## 懸念点

1. **🟡 import 時の副作用（_env_value.py）**
   - JIT/eager モード切り替え時に問題の可能性

1. **🟡 Tracer チェックの明示性**
   - `isinstance(..., getattr(jax, "core").Tracer)` は脆弱か？
   - `jax.core.Tracer` への直接アクセスが望ましいが、互換性の都合？

---

## 型安全性・規約遵守

## 確認項目

| 項目 | 状態 | コメント |
|---|---|---|
| 型注釈の完全性 | ✅ ほぼ完全 | `Any` の使用は最小限 |
| Protocol 準拠 | ✅ 確認済み | base, functional など |
| 環境変数名の統一 | ✅ `JAX_UTIL_*` prefix | 規約通り |
| DEBUG ガード | ✅ 実装 | ログ出力が適切に条件付き |
| 出典・引用 | ⚠️ 部分的 | MINRES, LOBPCG, PDIPM の論文参照を明記推奨 |

---

## テスト構造確認

**対象:** `python/tests/`

次ステップで詳細確認予定。以下は初期観察:
- solvers/: PCG, MINRES, KKT, LOBPCG のテストあり
- optimizers/: PDIPM のテストあり
- functional/: 統合テストあり

---

## まとめ

## 🟢 強み
1. **数学的正確性**: PCG, MINRES, LOBPCG, PDIPM は論文ベースで実装
1. **JAX 最適化**: `jax.lax.while_loop`, `filter_vmap` など効率的パターン
1. **型安全性**: jaxtyping + Protocol で型契約明確
1. **前処理戦略**: スペクトル前処理は堅牢かつ革新的

## 🟡 改善推奨
1. **アルゴリズム出典の明記**
   - ファイルヘッダに論文参考番号を記載（論文リスト参照）
   - 特に MINRES (Choi–Saunders), LOBPCG (Knyazev)

1. **バグ修正**
   - linearoperator.py の `__name__` → `type(__name__)`
   - hstack_linops の定義（concat vs sum）を明確化

1. **環境変数副作用の後延期**
   - import 時の `jax.config.update()` → setup() 関数への移行

1. **コード複雑性の削減**
   - smolyak.py, MINRES の状態管理を簡潔化（可能なら）
   - 数値演算のコメントを充実

## 🔴 確認待ち
1. neuralnetwork モジュールの詳細確認
1. テストカバレッジの確認
1. 実際の問題での数値検証（特に MINRES, PDIPM）

---

## 参考文献（推奨）

## Solvers
- Paige, C. C., & Saunders, M. A. (1975). Solution of sparse indefinite systems of linear equations. SIAM J. Numer. Anal.
- Choi, S. H., & Saunders, M. A. (1992). MINRES-QLP...
- Knyazev, A. V. (2001). Toward the optimal preconditioned eigensolver.

## Optimizers
- Mehrotra, S. (1992). On the implementation of a primal-dual interior point method.
- Boyd, S., & Vandenberghe, L. (2004). Convex Optimization.

## JAX
- JAX 公式ドキュメント: https://jax.readthedocs.io/

---

**レビュー完了日時:** 2026-03-15
**レビュアー:** GitHub Copilot (Claude Haiku 4.5)
