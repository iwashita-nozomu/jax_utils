# Git Mirroring

## このリポジトリの経路

- 通常の push 先は `origin=/mnt/git/jax_util.git`
- GitHub mirror は `GitHub=https://github.com/iwashita-nozomu/jax_utils.git`
- bare repo 側の `post-receive` hook は `git push --mirror github` を呼ぶ

つまり、通常は `main` から `origin` へ push し、その後 bare repo が GitHub へ mirror する構成です。

## どこを見るか

- local repo の remote:
  - `git remote -v`
- bare repo の config:
  - `/mnt/git/jax_util.git/config`
- bare repo の hook:
  - `/mnt/git/jax_util.git/hooks/post-receive`
- SSH config:
  - `~/.ssh/config`
- GitHub 用公開鍵:
  - `~/.ssh/id_ed25519_github_mirror.pub`

## 今回確認できたこと

- `origin` への push 自体は成功する。
- bare repo の hook は実際に走っている。
- ただし GitHub 側の認証情報が無いため、mirror push は失敗する。
- 直接 `git push GitHub main` しても同じ認証エラーになる。

## 典型的な失敗

- `origin` への push は通るが、GitHub mirror だけ失敗する。
- bare repo の hook に認証情報が無い。
- HTTPS remote を使っているが、username / token が供給されない。
- `git credential fill` 自体が GitHub の資格情報を返せない。

## 通すときの確認

- `git push origin main`
- `git rev-parse HEAD`
- `git rev-parse origin/main`
- bare repo の hook を確認
- 必要なら `git push GitHub main` を直接試す

## 現状の結論

- `origin` への反映はできている。
- GitHub mirror は認証待ちである。
- mirror を安定させるには、HTTPS の token か SSH 鍵のどちらかを supply する必要がある。

## Addendum: 2026-03-16 の再確認

2026-03-16 に再確認した時点では、GitHub mirror はまだ最新ではありませんでした。

- `HEAD`:
  - `62bca1cecc385d864b2d29408d6be0ef87f82bc6`
- `origin/main`:
  - `62bca1cecc385d864b2d29408d6be0ef87f82bc6`
- `GitHub/main`:
  - `51c7d2e799c75281327c97186475942da40dfc85`

つまり、`origin/main` と `GitHub/main` は一致していません。確認時点では、GitHub は `39` commits 分だけ遅れています。

同日に、mirror 経路は HTTPS から SSH へ切り替えました。

- local repo `GitHub` remote:
  - `git@github.com:iwashita-nozomu/jax_utils.git`
- bare repo `github` remote:
  - `git@github.com:iwashita-nozomu/jax_utils.git`
- SSH config:
  - `~/.ssh/config`
- generated key:
  - `~/.ssh/id_ed25519_github_mirror`
  - `~/.ssh/id_ed25519_github_mirror.pub`

この時点では、SSH 接続自体は公開鍵認証待ちで止まっています。

```text
git@github.com: Permission denied (publickey).
```text

つまり、経路は SSH に切り替わっていますが、GitHub 側に公開鍵がまだ登録されていません。

## 実際の確認手順

まず、local と `origin` が一致しているかを見ます。

```bash
git rev-parse HEAD
git rev-parse origin/main
```text

次に、GitHub 側の実際の先頭を見ます。fetch 済みの local tracking ref ではなく、remote 本体を見るのが安全です。

```bash
git ls-remote GitHub refs/heads/main
```text

差分件数は、次で見られます。

```bash
git rev-list --count GitHub/main..origin/main
```text

ただし、このコマンドは local の `GitHub/main` tracking ref を使うので、厳密に見たいときは先に `git fetch GitHub` するか、`ls-remote` の結果を優先します。

## mirror が遅れているときの通し方

### 1. remote 設定を確認する

```bash
git remote -v
git config --get-all remote.GitHub.pushurl
git --git-dir=/mnt/git/jax_util.git config --get remote.github.url
```text

少なくとも、local repo と bare repo の両方で GitHub remote が正しい必要があります。

SSH を使うなら、どちらも次の形にそろえるのが分かりやすいです。

```text
git@github.com:iwashita-nozomu/jax_utils.git
```text

### 2. まずは direct push を試す

```bash
git push GitHub main
```text

これで通れば、mirror 経路の問題ではなく bare repo hook 側の問題です。

### 3. bare repo mirror を確認する

```bash
sed -n '1,120p' /mnt/git/jax_util.git/hooks/post-receive
sed -n '1,120p' /mnt/git/jax_util.git/config
```text

この repo では bare repo 側で `git push --mirror github` を呼ぶ設計なので、hook と bare repo の remote 設定が一致している必要があります。

### 4. 認証方式を決める

この環境で一番詰まりやすいのは認証です。症状が

```text
fatal: could not read Username for 'https://github.com': terminal prompts disabled
```text

であれば、HTTPS remote は見えていても資格情報が渡っていません。

対策は次のどちらかです。

- HTTPS token を credential helper 経由で渡す
- SSH remote に切り替えて、bare repo と local repo の両方で鍵を使う

SSH を使う場合は、次も確認します。

```bash
ssh -T git@github.com
```text

ここで

```text
Permission denied (publickey)
```text

が出るなら、鍵ファイルはあっても GitHub 側への登録がまだです。

### 4.1 SSH 鍵を作る

```bash
ssh-keygen -t ed25519 -C "niwashita@users.noreply.github.com" -f ~/.ssh/id_ed25519_github_mirror
```text

```bash
cat ~/.ssh/id_ed25519_github_mirror.pub
```text

この公開鍵を GitHub の SSH keys に登録してから、再度 `ssh -T git@github.com` を試します。

### 4.2 SSH config をそろえる

```sshconfig
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github_mirror
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
```text

この設定があると、local repo と bare repo の両方で同じ鍵を使いやすくなります。

### 5. 通ったあとに再確認する

```bash
git ls-remote GitHub refs/heads/main
git rev-parse HEAD
```text

この 2 つが一致すれば、少なくとも `main` の mirror は完了です。

## References

- [Environment Setup](/workspace/notes/knowledge/environment_setup.md)
- [Experiment Operations](/workspace/notes/knowledge/experiment_operations.md)
