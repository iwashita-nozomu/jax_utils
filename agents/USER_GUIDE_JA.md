# 🤖 エージェント用ガイド（日本語）

> GitHub Copilot や Cursor などの AI エージェントが jax_util プロジェクトで作業するためのガイド集

______________________________________________________________________

## 📚 3 回のクリックで目的到達

### **あなたは何をしたいですか？**

#### 🔹 **「新しい実装を追加したい」**
1. [agents/README.md](./README.md) — チームロール・権限を確認
2. [TASK_WORKFLOWS.md](./TASK_WORKFLOWS.md) — 実装タスクの workflow を選択
3. [documents/AGENT_TASK_MAP.md](../documents/AGENT_TASK_MAP.md#💻-実装者向けtask-11-20) — Task 11-20 を実行

#### 🔹 **「実験・検証をしたい」**
1. [agents/README.md](./README.md) — experimenter ロール確認
2. [TASK_WORKFLOWS.md](./TASK_WORKFLOWS.md) — 実験タスクの workflow を選択  
3. [documents/research-workflow.md](../documents/research-workflow.md) — 実験設計・比較・改造ガイド

#### 🔹 **「コードレビューをしたい」**
1. [agents/COMMUNICATION_PROTOCOL.md](./COMMUNICATION_PROTOCOL.md) — review メッセージの書き方
2. [documents/REVIEW_PROCESS.md](../documents/REVIEW_PROCESS.md) — レビュー項目・チェックリスト
3. [reviews/](../reviews/) — 既存レビュー報告を参照

#### 🔹 **「タスク全体を管理したい」**
1. [agents/README.md](./README.md) — manager ロール確認
2. [TASK_WORKFLOWS.md](./TASK_WORKFLOWS.md) — タスク引き継ぎ workflow
3. [documents/AGENT_TASK_CHECKLIST.md](../documents/AGENT_TASK_CHECKLIST.md) — 管理者向けチェックリスト

#### 🔹 **「全タスク 50 個を把握したい」**
1. [documents/AGENT_TASK_INDEX_GUIDE.md](../documents/AGENT_TASK_INDEX_GUIDE.md) — ドキュメント 3つ の役割説明
2. [documents/AGENT_TASK_MAP.md](../documents/AGENT_TASK_MAP.md) — 50 タスク詳細マップ
3. [documents/AGENT_TASK_VALIDATION_REPORT.md](../documents/AGENT_TASK_VALIDATION_REPORT.md) — テスト・リソース検証

______________________________________________________________________

## 🎯 エージェント向けスキル別ガイド

### 📋 **コーディング・実装スキル**

| タスク | ファイル | ロール | 時間 |
|--------|---------|--------|------|
| 新規モジュール実装 | [TASK_WORKFLOWS.md](./TASK_WORKFLOWS.md#-モジュール実装workflow) | implementer | 2-3h |
| テスト追加 | [coding-conventions-testing.md](../documents/coding-conventions-testing.md) | implementer | 1-2h |
| ドキュメント更新 | [TASK_WORKFLOWS.md](./TASK_WORKFLOWS.md#-ドキュメント更新workflow) | implementer | 30min-1h |
| 型エラー削減 | [task.md](../task.md) | implementer | 2-3h |

**参考規約:**
- [Python コーディング規約](../documents/coding-conventions-python.md)
- [テスト規約](../documents/coding-conventions-testing.md)
- [Markdown 記法](../documents/coding-conventions.md)

______________________________________________________________________

**最後に更新:** 2026-04-01  
**対象:** GitHub Copilot, Cursor, Claude, その他 AI エージェント  
**言語:** 日本語 (Japanese)
