# GitHub Mirror Procedure

## Current Status

2026-03-17 時点では、`origin` と `GitHub` は一致していません。

- `HEAD`
  - `1dafb18c7347ab24af2cfaf3c3f693cf6e186b6f`
- `origin/main`
  - `1dafb18c7347ab24af2cfaf3c3f693cf6e186b6f`
- `GitHub/main`
  - `51c7d2e799c75281327c97186475942da40dfc85`

つまり、`origin/main` は更新されていますが、GitHub mirror はまだ追いついていません。

## Current Route

現在の mirror 経路は SSH です。

- local repo `GitHub` remote
  - `git@github.com:iwashita-nozomu/jax_utils.git`
- bare repo `/mnt/git/jax_util.git` の `github` remote
  - `git@github.com:iwashita-nozomu/jax_utils.git`
- bare repo hook
  - `/mnt/git/jax_util.git/hooks/post-receive`
  - `git push --mirror github`

つまり、通常は `git push origin main` のあとに bare repo が GitHub へ mirror します。

## Current SSH Setup

このマシンでは GitHub 用の鍵を用意済みです。

- private key
  - `/root/.ssh/id_ed25519_github_mirror`
- public key
  - `/root/.ssh/id_ed25519_github_mirror.pub`
- fingerprint
  - `SHA256:6K+GKmfKcS6mezj05+SMYjWe8XuR+bJ2vyuOzR1T/hU`

`/root/.ssh/config` には次の設定を入れています。

```sshconfig
Host github.com
  HostName github.com
  User git
  IdentityFile /root/.ssh/id_ed25519_github_mirror
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
```

## What Is Blocking The Mirror

SSH 経路への切り替え自体は終わっていますが、GitHub 側に公開鍵がまだ登録されていません。

そのため、現時点の接続確認は次の状態です。

```text
$ ssh -T git@github.com
git@github.com: Permission denied (publickey).
```

この状態では `git push GitHub main` も bare repo の mirror も成功しません。

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

SSH 接続確認:

```bash
ssh -T git@github.com
```

bare repo mirror 設定確認:

```bash
sed -n '1,120p' /mnt/git/jax_util.git/config
sed -n '1,120p' /mnt/git/jax_util.git/hooks/post-receive
```

## How To Finish The Mirror

1. GitHub の SSH keys に `/root/.ssh/id_ed25519_github_mirror.pub` を登録する。
2. `ssh -T git@github.com` が成功することを確認する。
3. `git push GitHub main` を一度 direct に試す。
4. その後 `git push origin main` を行い、bare repo の hook 経由 mirror も動くことを確認する。

## Last Attempt

今回の再試行では、SSH remote への切り替えまでは完了しましたが、公開鍵未登録のため mirror 自体は未完了でした。

## Related Note

- [Git Mirroring Knowledge](/workspace/notes/knowledge/git_mirroring.md)
