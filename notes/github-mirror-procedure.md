# GitHub Mirror Procedure

## Current Status

**2026-03-17 完全同期完了** ✅

- ローカル main: `387cb03` (81 commits)
- origin/main: `387cb03` (81 commits)
- GitHub/main: `387cb03` (81 commits)

**全ブランチ同期状態:**
- main ✅
- results/functional-smolyak-scaling ✅
- results/functional-smolyak-scaling-tuned ✅
- work/editing-20260316 ✅
- work/experiment-runner-module-20260316 ✅
- work/smolyak-tuning-20260316 ✅

## Executed Sync Procedure

2026-03-17 に実行した同期手順：

### Phase 1: ローカルの未コミット変更をコミット

```bash
# 2ファイルの変更を検出
git status --short
# M .github/copilot-instructions.md
# M notes/github-mirror-procedure.md

# 全変更をコミット
git add -A
git commit -m "Update GitHub mirror procedure and Copilot instructions - sync state before GitHub push"
```

### Phase 2: Main ブランチの 44 個未プッシュコミットを GitHub へプッシュ

```bash
# 同期前の状態
# origin/main: 80 commits
# GitHub/main: 36 commits
# 差分: 44 commits ⚠️

# GitHub へプッシュ
git push GitHub main
# 結果: 51c7d2e..387cb03 main -> main (成功)
```

### Phase 3: 6 個の Worktree ブランチを GitHub へプッシュ

```bash
# 各ブランチをプッシュ
git push GitHub results/functional-smolyak-scaling
git push GitHub results/functional-smolyak-scaling-tuned
git push GitHub work/editing-20260316
git push GitHub work/experiment-runner-module-20260316
git push GitHub work/smolyak-tuning-20260316

# 結果: すべてのブランチが GitHub に追加/更新された
```

### Phase 4: ローカルコミットを Origin へプッシュ

```bash
# ローカルで作成したコミット (387cb03) を origin へも反映
git push origin main
# 結果: 7488c07..387cb03 main -> main (成功)
```

### Phase 5: 完全同期の検証

```bash
# コミット数確認
echo "origin/main: $(git rev-list --count origin/main) commits"
echo "GitHub/main: $(git rev-list --count GitHub/main) commits"
echo "ローカル: $(git rev-list --count main) commits"

# 結果:
# origin/main: 81 commits ✅
# GitHub/main: 81 commits ✅
# ローカル: 81 commits ✅

# ブランチ別確認（全ブランチがハッシュ一致）
for branch in main results/functional-smolyak-scaling ...
  git rev-parse origin/$branch == git rev-parse GitHub/$branch  # ✅ 完全同期
done
```

## Synchronization Method

**実装完了: HTTPS Direct Push**

/bin/python3  GitHub リモートへ直接プッシュする方法で同期しています。/Workspace/Python/Tests/Neuralnetwork/Test_Neuralnetwork.

### 設定内容

```bash
# GitHub リモート（既設定）
git remote add GitHub https://github.com/iwashita-nozomu/jax_utils.git

# リモート確認
git remote -v
# GitHubhttps://github.com/iwashita-nozomu/jax_utils.git (fetch)
# GitHubhttps://github.com/iwashita-nozomu/jax_utils.git (push)
# origin/mnt/git/jax_util.git (fetch)
# origin/mnt/git/jax_util.git (push)
```

### 推奨ワークフロー

**方式 A: 明示的プッシュ（現在の方法）**

```bash
# Step 1: ローカルで作業・コミット
git add <files>
git commit -m "Message"

# Step 2: Origin にプッシュ
git push origin main

# Step 3: GitHub にもプッシュ
git push GitHub main
```

**方式 B: デフォルト Push ターゲット設定（自動化オプション）**

1回だけ設定：

```bash
# main ブランチの pushRemote を GitHub に設定
git config branch.main.pushRemote GitHub
```

`git push` のみで両方のリモートに同期されます：

```bash
git push  # → origin と GitHub 両方に自動送信
```

/bin/python3 /workspace/python/tests/neuralnetwork/test_neuralnetwork.py

```bash
git config core.pushDefault GitHub
```

## Optional: Automatic Mirror via Bare Repository

**推奨度: 低** - 現在の HTTPS Direct Push で十分に機能しています。

`git push origin main` のみで **自動的に** GitHub へも同期したい場合は、 Bare Repository の post-receive hook を使用できます。

### 前提条件

- SSH 接続が可能
- GitHub に SSH 公開鍵が登録されている
- Bare repository に GitHub remote が設定されている

### 設定手順

#### Step 1: Bare Repository に GitHub Remote を追加

```bash
cd /mnt/git/jax_util.git
git remote add github git@github.com:iwashita-nozomu/jax_utils.git
```

#### Step 2: Post-Receive Hook を設定

```bash
cat > /mnt/git/jax_util.git/hooks/post-receive << 'HOOK_EOF'
#!/bin/bash
# Mirror all branches to GitHub on every push
logger -t git-mirror "Starting mirror to GitHub..."
git push --mirror github 2>&1 | logger -t git-mirror
logger -t git-mirror "Mirror completed"
HOOK_EOF

chmod +x /mnt/git/jax_util.git/hooks/post-receive
```

#### Step 3: Hook 動作確認

```bash
# ローカルでコミット
git add .
git commit -m "Test message"

# Origin へプッシュ（Bare Repo の hook が自動実行される）
git push origin main

# GitHub へ反映を確認
git fetch GitHub
git log GitHub/main | head -3
```

## Verification Commands

/bin/python3 /workspace/python/tests/neuralnetwork/test_neuralnetwork.py

```bash
# 各リポジトリの HEAD コミット確認
git rev-parse HEAD                    # ローカル
git rev-parse origin/main             # Origin
git rev-parse GitHub/main             # GitHub

# コミット数比較
git rev-list --count origin/main      # Origin
git rev-list --count GitHub/main      # GitHub

# 全ブランチの同期状態
for branch in main results/functional-smolyak-scaling \
              results/functional-smolyak-scaling-tuned \
              work/editing-20260316 work/experiment-runner-module-20260316 \
              work/smolyak-tuning-20260316; do
  origin_hash=$(git rev-parse origin/$branch 2>/dev/null || echo "N/A")
  github_hash=$(git rev-parse GitHub/$branch 2>/dev/null || echo "N/A")
  if [ "$origin_hash" = "$github_hash" ]; then
    echo "✅ $branch"
  else
    echo "⚠️ $branch (diverged)"
  fi
done
```

## Summary

| 方式 | 状態 | 導入難度 | 推奨度 |
|------|------|----------|--------|
| **HTTPS Direct Push** | ✅ 実装完了 | 低 | ⭐⭐⭐ |
| Bare Repo Auto Mirror (SSH) | ℹ️ オプション | 中 | ⭐ |

**推奨: HTTPS Direct Push**
- 実装が完了している
- セットアップが簡単
- トラブル時の対応も簡単

**選択するなら: Bare Repo Auto Mirror**
- `git push origin main` のみで十分
- ユーザー側の明示的なアクションが不要
- Bare repository 側の設定が必要

## Related Notes

- [Git Mirroring Knowledge](/workspace/notes/knowledge/git_mirroring.md)
