# NN DF Zero Worktree Log

## 2026-04-04

- `work/nn-df-zero-20260404` を current `main` (`6030e1e`) から作成した。
- 目的は、ラグランジアンの一回微分に基づく最急法のうち、parameter update を $DF=0$ の解として与える枠組みを、別サブモジュールとして調査・検証・設計・実装すること。
- 作業対象は `python/jax_util/neuralnetwork/lagrangian_df_zero/` を仮の正本とし、既存の `neuralnetwork.py`、`train.py`、`sequential_train.py`、`dynamic_programming.py` は読み取り中心とする。
- 初期段階では、文献調査、数理的検証、設計文書、reference 整理を優先し、本実装はその後に入る。
- `Question:` $DF=0$ 型の layer update を、reduced problem の exact stationarity として定式化できるか。
- `Change:` 必読の workflow / coding convention と、既存 `neuralnetwork` 実装、`bellman_bp` 系 note を読み直した。
- `Change:` `references/lagrangian_df_zero/` に layer-wise、auxiliary coordinates、lifted training、implicit differentiation の一次資料パックを作成した。
- `Decision:` 主線は exact equality-constraint + Lagrangian とし、lifted / penalty 系は comparison baseline として扱う。
- `Decision:` 実装前の正本として `notes/themes/06_lagrangian_df_zero.md` と `documents/design/jax_util/lagrangian_df_zero.md` を追加した。
- `Decision:` 詳細設計では suffix 変数を $(\theta_{k:N}, z_{k:N}, p_{k:N})$ としてまとめて解く。
- `Decision:` 初回 carry-over の solver は dense Jacobian を使う toy Newton に限定し、operator-form / KKT 再利用は次段とする。
- `Decision:` 初回 carry-over の対象 network は `standard` のみとし、`icnn` は後段へ送る。
- `Decision:` 数式の正本記法は `bellman_bp_chat_summary.md` に合わせ、`z` ではなく $\phi$、loss ではなく $V_k$、parameter 微分の前に function-space 側の微分と誘導計量を書く。
- `Change:` `python/jax_util/neuralnetwork/lagrangian_df_zero/` に `types.py`、`parameterization.py`、`prefix.py`、`constraints.py`、`lagrangian.py`、`stationary_solve.py`、`toy.py` を追加した。
- `Change:` `python/tests/neuralnetwork/` に parameterization / constraints / toy solver 向けの targeted tests を追加した。
- `Expected Effect:` suffix stationary problem を既存 `neuralnetwork` 公開面を壊さずに構築し、toy residual と dense Newton solve を end-to-end で検証できるようにする。
- `Validation Plan:` `ruff`、`pyright`、targeted `pytest`、文書の `mdformat`、`git diff --check` を通す。
- `Validation Command:` `python3 -m ruff check python/jax_util/neuralnetwork/lagrangian_df_zero python/tests/neuralnetwork/test_lagrangian_df_zero_parameterization.py python/tests/neuralnetwork/test_lagrangian_df_zero_constraints.py python/tests/neuralnetwork/test_lagrangian_df_zero_toy.py --select E,F,I,D,UP`
- `Validation Command:` `python3 -m pyright /workspace/.worktrees/nn-df-zero-20260404/python/jax_util/neuralnetwork/lagrangian_df_zero`
- `Validation Command:` `PYTHONPATH=/workspace/.worktrees/nn-df-zero-20260404/python JAX_PLATFORMS=cpu python3 -m pytest -q python/tests/neuralnetwork/test_lagrangian_df_zero_parameterization.py python/tests/neuralnetwork/test_lagrangian_df_zero_constraints.py python/tests/neuralnetwork/test_lagrangian_df_zero_toy.py`
- `Validation Command:` `python3 -m mdformat documents/design/README.md documents/design/jax_util/README.md documents/design/jax_util/lagrangian_df_zero.md notes/themes/README.md notes/themes/06_lagrangian_df_zero.md notes/worktrees/worktree_nn_df_zero_2026-04-04.md references/README.md references/lagrangian_df_zero/README.md`
- `Validation Command:` `git diff --check`
- `Result Summary:` `standard` network 限定で、packed suffix variables、primal / adjoint / theta residual、warm-start multiplier、dense Newton toy solver、layer update extraction が動作した。
- `Result Summary:` `ruff`、`pyright`、targeted `pytest` は green になった。
- `Critical Review:` 現時点で確認できたのは residual consistency と toy residual decrease までであり、exact reduced stationary equation の紙上検算、baseline comparison、loss decrease の評価は未了である。
- `Critical Review:` $DF=0$ の主張はまだ `standard` + small toy + dense Jacobian の範囲に限るべきで、既存 trainer 置換や scalability claim を出してはいけない。
- `Change:` `documents/design/jax_util/lagrangian_df_zero.md` と `notes/themes/06_lagrangian_df_zero.md` に、旧 `D_f \widetilde F = 0` 記法から function-level reduced functional、parameter pullback、`theta_residual` への式展開を追加した。
- `Change:` さらに、$\theta_k$ が exact reduced stationarity では nonlinear root $R_k^{\mathrm{red}}(\theta_k)=0$ の解であり、Stage 0 では full KKT root $R_k(u_k)=0$ の先頭 block として得られることを書き分けた。
- `Change:` さらに、exact reduced root は一般に Newton / Gauss-Newton 向き、induced-metric update は SPD 条件つきで CG 向き、解析解は linear-quadratic toy に限る、という solver choice も明記した。
- `Change:` `notes/themes/06_lagrangian_df_zero.md` を `Question / Scope / Search Log / Primary Sources / Contrary Or Narrowing Sources / Known / Contested / Open / Implications` を持つ survey report 形式に再構成した。
- `Change:` `references/lagrangian_df_zero/README.md` と `catalog.md` を survey の分類に合わせて更新し、`Lau et al.` を optimizer-baseline line として取り込んだ。
- `Critical Review:` survey 初稿は設計メモ寄りで英語見出しが多く、そのまま reader-facing な文献レビューとしては弱かった。日本語の要旨、文献地図、批判的比較、結論を持つ論文調の構成へ組み替えた。
- `Critical Review:` さらに、`最も近い`、`最も重要` といった評価が deep learning 全体の一般論に読まれないよう、focused survey であることと worktree-specific な相対評価であることを明示した。
- `Critical Review:` survey 本文と worktree の設計判断が混ざらないように、Section 8 冒頭で「ここから先は文献からの自動的帰結ではなく設計方針である」と明記した。
- `Decision:` この worktree では `D_f \widetilde F = 0` を global function identity ではなく、固定した prefix state $\phi_{k-1}$ における fiberwise stationarity として読む。
- `Research Perspective Review:`
  - `reproducibility`: 参考文献、設計正本、validation command、targeted tests を残したので rerun 導線は確保できた。
  - `scientific-computing`: 既存 trainer を触らず別サブモジュールとして小さく実装した点は良い。一方で exact reduced solve の解析的照合は follow-up です。
  - `benchmark`: baseline comparison はまだ未着手なので、性能や優位性の主張は禁止する。
  - `artifact`: code、tests、design、notes、references は揃った。削除挙動が不安定な stray PDF は索引から外し、artifact の正本には含めない。
  - `fair-data`: reference pack と note / design のリンクは揃えた。命名規則も topic-local に統一した。
  - `ml-science-reporting`: claim は toy implementation と residual validation の範囲に限定し、未確認事項を明示した。
- `Decision:` Stage 0 の toy implementation は承認する。
- `Decision:` 既存 trainer への統合、benchmark、operator-form solver、`icnn` 対応は次段の課題として残す。
- `Branch Reflection:` この worktree 内で文献調査、設計、toy 実装、validation を完結させた。既存 `neuralnetwork.py`、`train.py`、`sequential_train.py`、`dynamic_programming.py` は直接変更していない。

## 2026-04-05

- `Change:` `notes/themes/06_lagrangian_df_zero.md` に「このプロジェクトの記法による式展開」を追加し、`D_f \widetilde F = 0` を Bellman 記法、function-level reduced functional、parameter pullback、`R_k^{\mathrm{red}}(\theta_k)=0`、`G_k \Delta \theta_k = -\tilde p_k` の順で読み下せるようにした。
- `Change:` 同ノートの節番号を繰り下げ、文献レビューから project-local な数式解釈へ自然につながる構成に整理した。
- `Critical Review:` survey 本文に project-local な数式節を入れると設計メモ化しやすいので、Section 5 は「文献の自動的帰結」ではなく「この project が採用する読み方の明文化」として限定した。
- `Critical Review:` exact reduced stationarity と局所更新方程式を同一視しないことを再確認し、`theta` が解く方程式、Stage 0 の full KKT root、CG が妥当になる範囲を同じ節の中で書き分けた。
- `Validation Command:` `python3 -m mdformat notes/themes/06_lagrangian_df_zero.md notes/worktrees/worktree_nn_df_zero_2026-04-04.md`
- `Validation Command:` `python3 scripts/tools/check_markdown_lint.py notes/themes/06_lagrangian_df_zero.md notes/worktrees/worktree_nn_df_zero_2026-04-04.md`
- `Validation Command:` `git diff --check`
- `Result Summary:` survey note だけ読んでも、この project における `DF=0`、`theta_residual`、full KKT root、局所更新方程式の関係が追えるようになった。
