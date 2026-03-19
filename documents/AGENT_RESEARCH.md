# コーディング Agent に関する調査メモ（拡張版）

このドキュメントは、本プロジェクト向けに「コーディング Agent」の研究知見を実運用へ接続するための整理です。特に以下を重視します。

- Agent 設計原則（Plan / Execute / Verify）
- ファイル探索順（どの順でどのファイルを読むと失敗率が下がるか）
- ディレクトリ構成（安全性・検証性・監査性）

## 1. 主要な外部知見（論文・実装ノート）

### 1.1 ReAct（Reason + Act）

- 出典: https://arxiv.org/abs/2210.03629
- 要点: 推論と行動を交互に行うと、環境フィードバックを取り込みやすくなり、幻覚や探索失敗を減らせる。
- 本プロジェクトへの示唆:
	- 「読む → 仮説 → 実行（検索/検証） → 更新」のループを標準化する。
	- 1ターンで大きく編集するより、小さな検証単位で進める。

### 1.2 Reflexion（言語的フィードバック学習）

- 出典: https://arxiv.org/abs/2303.11366
- 要点: 重み更新なしでも、失敗理由をテキスト記録して次試行に反映すると性能が上がる。
- 本プロジェクトへの示唆:
	- `reports/agents/<run-id>/retrospective.md` のような反省ログを必須化。
	- 「同じ失敗を繰り返さない」ための再試行ポリシーを定義する。

### 1.3 SWE-bench（評価ベンチマーク）

- 出典: https://arxiv.org/abs/2310.06770
- 要点: 実運用に近い GitHub issue 解決には、複数ファイル横断理解・長文脈処理・テスト実行が必要。
- 本プロジェクトへの示唆:
	- 単一ファイル最適化ではなく、変更波及（import / API / docs）を同時追跡する。
	- 成功判定は「編集できたか」ではなく「テスト・静的解析を通したか」に置く。

### 1.4 SWE-agent（Agent-Computer Interface, ACI）

- 出典: https://arxiv.org/abs/2405.15793
- 要点: Agent の性能はモデル本体だけでなく、ツール I/F（ACI）設計で大きく変わる。
- 本プロジェクトへの示唆:
	- 絶対パス利用、差分編集、明示的エラー取得など、ツール契約を厳格化する。
	- `planner/executor/verifier` を分け、I/O 形式を単純化する。

### 1.5 AutoCodeRover（構造化探索 + 局所化）

- 出典: https://arxiv.org/abs/2404.05427
- 要点: ファイル列挙だけでなく、構文構造（クラス・関数）ベース探索と fault localization が有効。
- 本プロジェクトへの示唆:
	- ファイル探索は「ディレクトリ順」より「シンボル順（定義→利用）」を優先する。
	- 失敗テストがある場合は、そのテストから逆向きに依存を辿る。

### 1.6 実装ノート（運用観点）

- SWE-agent リポジトリ: https://github.com/SWE-agent/SWE-agent
- OpenHands リポジトリ: https://github.com/All-Hands-AI/OpenHands
- Anthropic Engineering Note: https://www.anthropic.com/engineering/building-effective-agents

共通傾向:

- 「単純なワークフローを合成」する構成が、複雑なフレームワーク依存より運用しやすい。
- Agent の失敗原因は、モデル能力不足よりも探索順・ツール仕様・停止条件の未整備であることが多い。

## 2. 本プロジェクト向けの設計原則

1. **最小編集**: 1 回の編集は小さく、検証可能であること。
2. **観測優先**: 先に探索・計測し、根拠なしで編集しないこと。
3. **検証駆動**: 変更直後に静的解析/テストを回すこと。
4. **監査可能性**: 実行ログ・diff・判断根拠を `reports/` に残すこと。

## 3. Agent のファイル探索順（重要）

本プロジェクトでは、以下の探索順を標準にします。

### Phase 0: スコープ固定

- `WORKTREE_SCOPE.md`
- issue / task 文
- 影響対象ディレクトリ（例: `python/jax_util/`, `documents/`）

### Phase 1: 高シグナル入口を先読む

1. `README.md`
2. `documents/README.md`
3. 関連する規約（`documents/coding-conventions-*.md`）
4. 対象サブモジュール設計（`documents/design/jax_util/*.md`）

### Phase 2: 実装入口（定義側）

1. `__init__.py`
2. `core.py` / API 入口
3. 型定義や共通ユーティリティ（`_type_aliaces.py`, `_linop_utils.py` など）

### Phase 3: 利用側（呼び出し側）

- `tests/` → `python/` 利用箇所 → `scripts/` 依存箇所 の順で確認。

### Phase 4: 変更→検証ループ

- 小さい差分を適用し、静的解析・テスト・レポート更新を行う。

探索優先度スコア（実装ガイド）:

$$
score(file) = 0.40\,R + 0.30\,D + 0.20\,T + 0.10\,C
$$

- $R$: 要求との関連度（キーワード一致）
- $D$: 依存中心性（import / 呼び出し関係）
- $T$: テスト失敗との近さ
- $C$: 最近の変更頻度（churn）

このスコアで上位ファイルから読むと、無駄な探索を減らせます。

## 4. 失敗しやすいパターンと対策

- **いきなり広範囲編集** → 対策: 3ファイル以内の小差分に分割。
- **規約未確認で修正** → 対策: `documents/coding-conventions-*.md` を先読む。
- **検証不足の commit** → 対策: verifier で静的解析 + 最低1テストを必須化。
- **再現不能な意思決定** → 対策: `reports/agents/<run-id>/decision_log.md` を保存。

## 5. 本プロジェクトへの適用結論

- Agent の性能改善は、モデル変更よりも「探索順」「ツール仕様」「検証導線」の改善が効きます。
- よって、まずは `Agent` 用ディレクトリを分離し、探索順と検証契約をコード化する方針が妥当です。
- 具体構成は `documents/AGENT_DIRECTORY_PROPOSAL.md` を参照してください。
