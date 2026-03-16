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

## References

- [Environment Setup](/workspace/notes/knowledge/environment_setup.md)
- [Experiment Operations](/workspace/notes/knowledge/experiment_operations.md)
