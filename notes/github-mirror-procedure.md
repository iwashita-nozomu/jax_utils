# GitHub Mirror Procedure - Auto Mirror Setup

## Current Status

**2026-03-17 自動ミラー設定完了** ✅

ローカルリポジトリから `origin` へプッシュすると、**自動的に GitHub へもミラーされます**。

```
ローカル → origin (push) → Hook 実行 → GitHub (自動同期)
```

## Recommended Workflow

**推奨: 1つのコマンドだけで OK**

```bash
# Step 1: ローカルで作業・コミット
git add <files>
git commit -m "Feature description"

# Step 2: Origin へプッシュ（hook が自動実行 → GitHub も同期）
git push origin main
```

これだけで **origin と GitHub 両方が同期** されます。

## Implementation Details

### Bare Repository Hook Setup

Bare repository (`/mnt/git/jax_util.git`) に post-receive hook を設定しました。

**Hook の内容:**

```bash
#!/bin/sh
set -e
# Automatically mirror to GitHub after every push to origin
git push --mirror github 2>&1 | sed 's/^/[git-mirror] /'
```

**特徴:**
- 相対パスで記述（移植性確保）
- エラーログは `[git-mirror]` プレフィックス付きで表示

### Bare Repository Remote Configuration

```bash
# Bare repo の GitHub remote
$ cd /mnt/git/jax_util.git && git remote -v
github  git@github.com:iwashita-nozomu/jax_utils.git (fetch)
github  git@github.com:iwashita-nozomu/jax_utils.git (push)
```

**重要:** SSH 接続を使用しています（credential の煩雑さを回避）

## Prerequisites

GitHub への SSH 接続が必要です。以下の手順で設定してください：

### Step 1: SSH 公開鍵を GitHub に登録

ローカルの公開鍵情報：

```
SSH Key Type: Ed25519
Location: /root/.ssh/id_ed25519_github_mirror.pub
```

**GitHub への登録手順:**

1. https://github.com/settings/keys にアクセス
2. **"New SSH key"** をクリック
3. 以下の公開鍵をペースト：

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIK+L6Gxakq8DKvU/7ISMIuxA/pS4MRgnPjDLRC0Pvhcb
```

4. **"Add SSH key"** をクリック

### Step 2: SSH 接続を確認

```bash
ssh -T git@github.com
# 期待される出力: Hi <username>! You've successfully authenticated...
```

### Step 3: Hook が動作することを確認

```bash
# ローカルでコミット
cd /workspace
echo "test" > test.txt
git add test.txt
git commit -m "Test: Verify mirror hook"

# Origin へプッシュ（hook が自動実行）
git push origin main

# GitHub に反映を確認
git fetch GitHub --all
git log GitHub/main | head -3
```

## Synchronization Verification

全ブランチの同期状態を確認：

```bash
# 簡単確認
git rev-parse origin/main
git rev-parse GitHub/main
# 同じハッシュが表示されれば ✅ OK

# 詳細確認（全ブランチ）
for branch in main results/functional-smolyak-scaling \
              results/functional-smolyak-scaling-tuned \
              work/editing-20260316 work/experiment-runner-module-20260316 \
              work/smolyak-tuning-20260316; do
  origin_hash=$(git rev-parse origin/$branch 2>/dev/null || echo "N/A")
  github_hash=$(git rev-parse GitHub/$branch 2>/dev/null || echo "N/A")
  local_hash=$(git rev-parse $branch 2>/dev/null || echo "N/A")
  
  if [ "$origin_hash" = "$github_hash" ] && [ "$origin_hash" = "$local_hash" ]; then
    echo "✅ $branch"
  else
    echo "⚠️ $branch (diverged)"
  fi
done
```

## Troubleshooting

### Hook が実行されても GitHub に反映されない

**症状:**

```
remote: [git-mirror] git@github.com: Permission denied (publickey).
```

**原因:** GitHub に SSH 公開鍵が登録されていない

**解決策:** [Prerequisites](#prerequisites) の Step 1 を実施

### Hook 実行ログを確認

```bash
# Bare repo の hook ログ
tail -f /var/log/syslog | grep git-mirror

# または、手動実行してテスト
cd /mnt/git/jax_util.git
git push --mirror github
```

### Manual Force Sync（緊急時）

origin と GitHub を手動で同期：

```bash
cd /workspace
git push GitHub --all  # 全ブランチを GitHub へプッシュ
```

## Configuration Summary

**ローカルリポジトリ:**
- `origin` remote: `/mnt/git/jax_util.git`
- `GitHub` remote: `https://github.com/iwashita-nozomu/jax_utils.git` (fetch only)

**Bare Repository (`/mnt/git/jax_util.git`):**
- `github` remote: `git@github.com:iwashita-nozomu/jax_utils.git` (SSH)
- Hook: `hooks/post-receive` (自動ミラー実行)

**GitHub:**
- SSH Deploy Key 登録: ✅ Required (see Prerequisites)

## Related Notes

- [Git Mirroring Knowledge](../knowledge/git_mirroring.md)
