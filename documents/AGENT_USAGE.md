# Agent Usage（統合版）

この文書は、`Agent` の調査結果・運用方針・探索順ポリシー・推奨ディレクトリ構成を一体化した正本です。

## 1. 目的

- 文書の分散を防ぎ、参照先を最小化する
- Agent の安全運用（Plan / Execute / Verify）を定着させる
- 探索順を標準化して、調査漏れや無駄な編集を減らす

## 2. 研究知見サマリ（要点のみ）

### ReAct

- 出典: https://arxiv.org/abs/2210.03629
- 推論と行動の交互ループで環境フィードバックを活用。
- 適用: 「読む→仮説→検証→更新」の反復を標準化。

### Reflexion

- 出典: https://arxiv.org/abs/2303.11366
- 失敗の言語的反省を次試行に活かす設計が有効。
- 適用: `reports/agents/<run-id>/retrospective.md` を必須化。

### SWE-bench

- 出典: https://arxiv.org/abs/2310.06770
- 実問題では複数ファイル横断理解とテスト駆動が必要。
- 適用: 成功判定は「編集完了」ではなく「検証通過」。

### SWE-agent（ACI）

- 出典: https://arxiv.org/abs/2405.15793
- Agent 性能はツール I/F 設計（ACI）に強く依存。
- 適用: 絶対パス、差分編集、明示的検証を標準化。

### AutoCodeRover

- 出典: https://arxiv.org/abs/2404.05427
- 構文構造ベース探索と fault localization が有効。
- 適用: 「定義→利用」の探索順を優先。

### 実装ノート

- SWE-agent: https://github.com/SWE-agent/SWE-agent
- OpenHands: https://github.com/All-Hands-AI/OpenHands
- Building effective agents: https://www.anthropic.com/engineering/building-effective-agents

## 3. 本プロジェクトでの設計原則

1. 最小編集: 1 回の変更は小さく保つ
1. 観測優先: 先に探索・根拠収集、後で編集
1. 検証駆動: 編集直後に静的解析・テストを実行
1. 監査可能: 実行ログ・差分・判断根拠を保存

## 4. 標準ファイル探索順（重要）

### Phase 0: スコープ固定

- `WORKTREE_SCOPE.md`
- issue / task 文
- 影響対象ディレクトリ

### Phase 1: 高シグナル入口

1. `README.md`
1. `documents/README.md`
1. 関連規約（`documents/coding-conventions-*.md`）
1. 設計（`documents/design/jax_util/*.md`）

### Phase 2: 実装の定義側

1. `python/**/__init__.py`
1. `python/**/core.py`
1. 型・共通ユーティリティ

### Phase 3: 実装の利用側

- `tests/**` → 呼び出し元モジュール

### Phase 4: 編集と検証

- 小差分編集 → 静的解析/テスト → レポート保存

探索優先度スコア:

$$
score(file) = 0.40\,R + 0.30\,D + 0.20\,T + 0.10\,C
$$

- $R$: 要求との関連度
- $D$: 依存中心性
- $T$: テスト失敗との近さ
- $C$: 変更頻度

## 5. 推奨ディレクトリ構成

- `agents/`

  - `planner/`: タスク分解・計画
  - `executor/`: 差分適用・worktree 操作
  - `verifier/`: 解析・テスト・安全判定
  - `retriever/`: 探索順制御（ランキング）
  - `policies/`: 変更制約、承認条件
  - `agents_config.json`: 実行設定
  - `logs/`: 実行ログ

- `scripts/agent_tools/`

  - 文書整形、リンク監査、類似検出、探索ランキング計算

- `reports/agents/<run-id>/`

  - `decision_log.md`
  - `edits.patch`
  - `verification.txt`
  - `retrospective.md`

## 6. 運用ルール

- 原則として Agent は `main` へ直接 push しない（ドキュメント更新例外は運用判断）。
- 変更は `agent/*` ブランチで PR 化し、CI + 人間レビューを通す。
- すべての Agent 実行は `reports/agents/` に証跡を保存する。

## 7. 停止条件

- 検証が通り、変更目的を満たしたら終了
- 同種エラーが3回連続したらエスカレーション
- スコープ外波及が発生したら、先に設計文書を更新

## 8. 失敗しやすいパターンと対策

- いきなり広範囲編集 → 3ファイル以内の小差分へ分割
- 規約未確認で編集 → 規約・設計を先読みに固定
- 検証不足で commit → verifier の必須ゲート化
- 意思決定が再現不能 → `decision_log.md` を保存

## 9. 参考

- ReAct: https://arxiv.org/abs/2210.03629
- Reflexion: https://arxiv.org/abs/2303.11366
- SWE-bench: https://arxiv.org/abs/2310.06770
- SWE-agent: https://arxiv.org/abs/2405.15793
- AutoCodeRover: https://arxiv.org/abs/2404.05427
