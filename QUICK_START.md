# 🚀 作業開始ガイド

## 現在の状態

✅ **mainブランチに最新の変更をPush完了**

- Markdown規約修正（MD029/MD060対応）
- Protocol型アノテーション改善
- ドキュメント整理

## 📚 規約について

このプロジェクトは **20種類以上の詳細な規約** に従って実装されています：

### Python規約（15章）

- 型アノテーション、責務分離、命名規則、JAX運用等

### 共通規約（5章）

- ドキュメント形式、コメント方針、演算子記法等

## 🎯 3ステップで作業開始

### ステップ1: 規約確認

```bash
bash scripts/view_conventions.sh
```

**主要な規約（最初に確認すべき）:**

- `documents/conventions/python/04_type_annotations.md` - 型安全性
- `documents/conventions/python/09_file_roles.md` - 責務分離
- `documents/conventions/common/05_docs.md` - ドキュメント形式

### ステップ2: ワークツリーを作成

```bash
bash scripts/setup_worktree.sh <branch-name> [description]
```

**例:**

```bash
# Protocol型アノテーションの改善
bash scripts/setup_worktree.sh protocol-improvements "Protocol型アノテーション改善"

# train.pyのAPI改善
bash scripts/setup_worktree.sh train-api-refactor "train.pyのAPI改善" 

# テスト追加
bash scripts/setup_worktree.sh neuralnetwork-tests "ニューラルネットテスト拡充"
```

自動作成される内容：

- ブランチ: `work/<branch-name>-<YYYYMMDD>`
- ワークツリー: `.worktrees/<branch-name>-<YYYYMMDD>/`

### ステップ3: ワークツリーで作業

```bash
# ワークツリーに移動
cd .worktrees/<branch-name>-<YYYYMMDD>

# 作業実施
# ... ファイル編集 ...
make test

# コミット・プッシュ
git add -A
git commit -m "category: 説明"
git push origin work/<branch-name>-<YYYYMMDD>
```

## 📋 コミットメッセージの規約

```bash
feat:      新機能追加
fix:       バグ修正
docs:      ドキュメント更新
refactor:  コード改善（機能変更なし）
test:      テスト追加/修正
chore:     ビルド等の設定変更
```

## 🧹 ワークツリーのクリーンアップ

（マージ完了後）

```bash
# ワークツリーを削除
git worktree remove .worktrees/<branch-name>-<YYYYMMDD>

# ローカルブランチを削除
git branch -d work/<branch-name>-<YYYYMMDD>
```

## 📊 現在のブランチ・ワークツリー確認

```bash
# 全ワークツリー表示
git worktree list

# ブランチ一覧
git branch -v

# リモートブランチ
git branch -r
```

## 🔗 参考資料

### 全体ガイド（インタラクティブ）

```bash
bash scripts/guide.sh
```

### 規約一覧

```bash
bash scripts/view_conventions.sh
```

### 特定の規約を確認

```bash
# 型アノテーション規約
less documents/conventions/python/04_type_annotations.md

# Markdown形式規約
less documents/conventions/common/05_docs.md

# プロジェクト全体のコーディング規則
less documents/coding-conventions-project.md
```

## 💡 Tips

### よくある作業パターン

#### 1. 新機能追加（複数章に渡る大型タスク）

```bash
bash scripts/setup_worktree.sh major-feature-name "大型機能の実装"
cd .worktrees/major-feature-name-20260318
# ... 大きな変更 ...
# 複数回コミット可能
```

#### 2. バグ修正（緊急・小規模）

```bash
bash scripts/setup_worktree.sh quick-fix "バグ修正のタイトル"
cd .worktrees/quick-fix-20260318
# ... 修正 ...
git add -A
git commit -m "fix: バグ詳細"
```

#### 3. ドキュメント更新（規約改善含む）

```bash
bash scripts/setup_worktree.sh docs-update "規約更新・ドキュメント改善"
cd .worktrees/docs-update-20260318
# ... ファイル編集 ...
```

## ⚠️ 注意点

1. **ワークツリーは独立している**
   - mainで作業しない
   - 各ワークツリーは完全に独立

2. **マージ前にレビューを通す**
   - GitHub PR作成時に自動でレビュー

3. **規約に従わない場合**
   - CI/CD が失敗するので、必ず規約を確認

## 🆘 トラブルシューティング

### ワークツリーが作成できない

```bash
# origin/mainが古い可能性
git fetch origin main
bash scripts/setup_worktree.sh ...
```

### ブランチ削除に失敗

```bash
# マージされていない場合は -D で強制削除
git branch -D work/branch-name-20260318
```

### ワークツリー登録状態を確認

```bash
git worktree prune  # 無効なエントリを削除
git worktree list   # 確認
```

---

**質問や問題がある場合は、docs/ 内の規約を参照するか、メンターに相談してください。**
