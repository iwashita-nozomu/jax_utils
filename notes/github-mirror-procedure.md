# GitHub Mirror Procedure

## Current Status

2026-03-17 時点では、`origin` と GitHub は一致していません。

- `HEAD` と `origin/main`
  - 検証時点で一致
- `GitHub/main`
  - `51c7d2e799c75281327c97186475942da40dfc85`

つまり、`origin/main` は進んでいますが、GitHub mirror は止まっています。

## Current Route

現在の mirror 経路は HTTPS です。

- local repo `GitHub` remote
  - `https://github.com/iwashita-nozomu/jax_utils.git`
- bare repo `/mnt/git/jax_util.git` の `github` remote
  - `https://github.com/iwashita-nozomu/jax_utils.git`
- bare repo hook
  - `/mnt/git/jax_util.git/hooks/post-receive`
  - `git push --mirror github`

通常は `git push origin main` のあとに bare repo が GitHub へ mirror する構成です。

## What Was Carefully Confirmed

次の点は実際に確認しました。

1. `main` repo の local config が欠けていて、`core.filemode=false` が落ちていました。
2. その結果、`git status` が大量変更に見える状態になっていました。
3. `.git/config` を復元すると、`git status` は正常化しました。
4. `GitHub` remote と bare repo 側 `github` remote は HTTPS に戻しました。
5. その状態で direct push と bare repo mirror の両方を試しました。

## Current Failure

local repo からの direct push:

```text
$ git push GitHub main
fatal: could not read Username for 'https://github.com': terminal prompts disabled
fatal: could not read Username for 'https://github.com': terminal prompts disabled
fatal: could not read Username for 'https://github.com': No such device or address
```

bare repo からの mirror:

```text
$ git --git-dir=/mnt/git/jax_util.git push --mirror github
fatal: could not read Username for 'https://github.com': terminal prompts disabled
fatal: could not read Username for 'https://github.com': terminal prompts disabled
fatal: could not read Username for 'https://github.com': No such device or address
```

つまり、失敗は bare hook 固有ではありません。local repo でも bare repo でも、GitHub 用の HTTPS 資格情報が供給されていません。

## Verification Commands

`origin` との一致確認:

```bash
git rev-parse HEAD
git rev-parse origin/main
```

GitHub 側の公開先頭確認:

```bash
git ls-remote https://github.com/iwashita-nozomu/jax_utils.git refs/heads/main
```

local remote の確認:

```bash
git remote -v
```

bare repo 側の mirror 設定確認:

```bash
sed -n '1,120p' /mnt/git/jax_util.git/config
sed -n '1,120p' /mnt/git/jax_util.git/hooks/post-receive
```

direct push の確認:

```bash
git push GitHub main
```

bare repo 側 mirror の確認:

```bash
git --git-dir=/mnt/git/jax_util.git push --mirror github
```

## What Is Missing

今足りていないのは、GitHub 用の HTTPS 資格情報です。

この環境では `credential.helper` 自体は設定されていますが、実際には GitHub の username / token を返せていません。そのため、`push` の段階で止まります。

## Practical Next Step

1. GitHub 用の HTTPS token を、この環境の credential helper が読める形で供給する。
2. `git push GitHub main` が通ることを確認する。
3. その後 `git --git-dir=/mnt/git/jax_util.git push --mirror github` を再実行して、bare repo mirror も通ることを確認する。
4. 最後に `git ls-remote https://github.com/iwashita-nozomu/jax_utils.git refs/heads/main` で先頭一致を見る。

## Related Note

- [Git Mirroring Knowledge](/workspace/notes/knowledge/git_mirroring.md)
