# プロジェクト全体の運用規約

この文書は、Docker 環境・ドキュメント運用・サブモジュールの位置づけをまとめる高レベル文書です。
文書は参照の連鎖で読ませず、この文書単体でも全体方針が分かるように保ちます。

## 1. 対象

- 対象は `/workspace` 配下の全体構成です。
- 安定サブモジュールは `base` / `solvers` / `optimizers` / `hlo` とします。
- `neuralnetwork` は実験段階、`solvers/archive` は保管領域として扱います。

## 2. 文書構造

- `documents/README.md` を `documents/` 全体の入口とします。
- general なコーディング方針は `coding-conventions*.md` と `conventions/` に置きます。
- base の型・Protocol・共通クラスは `design/base_components.md` を正本とします。
- 安定サブモジュールの API 詳細は `design/apis/` にサブモジュール単位で置きます。
- 補助文書が必要な場合でも、正本は 1 つに定めます。
- レビュー報告と実装進捗報告は `./reviews/` に置きます。
- 新規のレビュー文書は `./reviews/` に置き、ファイル名に作成エージェント識別子を含めます。
- 軽量なメモと実験の考察は `./notes/` に置きます。
- branch ごとの要約と note 入口は `./notes/branches/` に置きます。
- topic ごとの知識整理は `./notes/themes/` に置きます。
- 日付ごとの作業ログは `./diary/` に置きます。
- `./notes/experiments/` のメモは実験ごとに分けます。
- `./notes/experiments/results/` には、`main` に持ち帰る最小限の final JSON を置きます。
- 削除予定の worktree から吸い出したメモは `./notes/worktrees/` に置きます。
- `./diary/` は日付ファイルへ逐次追記する運用を基本とします。
- `./notes/` と `./diary/` では、文献由来の記述に対象文献を明示し、アイデアや考察はラベルで区別します。
- 一度 `main` に載せた過去の `./notes/` と `./diary/` の本文は、原則として書き換えません。
- 過去の記述へ補足や訂正が必要な場合は、既存本文を差し替えず、追記で `Correction:` / `Addendum:` を明示して残します。
- `./notes/` のタイトルは日付ではなくトピックを主にし、日付は副情報として扱います。

## 3. Docker 環境の方針

- 既定の実行環境は `docker/Dockerfile` を基準にします。
- 追加パッケージが必要な場合は、**`docker/Dockerfile` と `docker/requirements.txt` を同時に更新**します。
- 仮想環境の作成は行いません。`venv` / `virtualenv` / `conda` の導入は対象外です。
- テストやスクリプトは、repo root から `PYTHONPATH=./python` を基本とします。
- import パスは `jax_util.*` で統一します。

## 4. サブモジュールの位置づけ

| パス | 状態 | 役割 |
| --- | --- | --- |
| `python/jax_util/base` | 安定 | 型・定数・作用素の基盤。 |
| `python/jax_util/solvers` | 安定 | 数値ソルバと関連ユーティリティ。 |
| `python/jax_util/optimizers` | 安定 | `solvers` を利用する最適化アルゴリズム。 |
| `python/jax_util/hlo` | 安定 | HLO ダンプと解析補助。 |
| `python/jax_util/neuralnetwork` | 実験段階 | forward / train の整理中。 |
| `python/jax_util/solvers/archive` | 保管 | 現在使わないアルゴリズムの退避先。 |
| `python/tests` | 安定 | 検証とログ出力。 |
| `scripts` | 安定 | テスト・ログ・依存解析の補助。 |
| `experiments` | 実験運用 | 長時間実験、結果整理、レポート生成。 |
| `reviews` | 安定 | Git 管理するレビュー報告と進捗文書。 |
| `notes` | 安定 | 実験メモ、考察。 |
| `diary` | 安定 | 日付ごとの作業ログ。 |
| `documents` | 安定 | 規約と設計書の一次情報源。 |

## 5. 共通ルール

- `Scalar` / `Vector` / `Matrix` を優先します。
- `Matrix` は `(n, batch)` を原則とします。
- 線形作用素の適用は `@`、合成は `*` を原則とします。
- 反復は `jax.lax.scan` / `jax.lax.while_loop` / `jax.lax.fori_loop` を優先します。
- ログ出力は `DEBUG` ガードの内側でのみ行います。

## 6. ドキュメント更新ルール

- 実装変更が入った場合は、該当する `documents/` の文書を同時に更新します。
- 規約を増やすときは `documents/conventions/` に追記します。
- base の型・Protocol・クラスを変えるときは `documents/design/base_components.md` を更新します。
- 安定サブモジュールの API を変えるときは `documents/design/apis/` の対応ファイルを更新します。
- 文書は参照の一覧ではなく、責務ごとのまとまりとして編成します。
- 実装パスへの参照は、実装上の制約を明示する必要がある場合に限ります。

## 7. タスク運用

- `task.md` は実装の進行管理に使います。
- 粒度はファイル単位で統一します。
- 完了した項目は `task.md` から削除し、仕様や判断は `documents/` に残します。

## 8. ブランチ運用

- 実装コードと生成物は、必要に応じてブランチを分けます。
- worktree の作成、`WORKTREE_SCOPE.md`、削除前の吸い出し、削除条件は `documents/worktree-lifecycle.md` を正本とします。
- 実験を実行するときは、実験専用の worktree を作ってその worktree 上で動かします。
- `main` の worktree ではコード編集・文書更新・通常テストのみを行い、長時間走る実験は開始しません。
- worktree は VS Code やホスト OS から見える共有パスに置くことを原則とし、既定では `./.worktrees/<name>` を使います。
- 実験コードそのものは `main` に載せてもよいですが、生成された実験結果は専用の results ブランチへ分離することを原則とします。
- results ブランチ名は `results/<topic>` 形式を原則とし、実験ごとに固有の名前を使います。
- 実験用 worktree は results ブランチに対応づけ、`main` と同じ worktree のまま branch を切り替えて使い回しません。
- 実験を始める前に、`main` と対象 results ブランチを `origin` と同期し、両方の worktree が clean であることを確認します。
- 実験固有のコード調整は、まず対応する results ブランチの worktree 上で行い、実験で確認した後に `main` へ統合します。
- `main` で先に変更するのは、実験固有ではない共通コードや規約更新に限ります。
- 実験スクリプトの先頭には、対応する results ブランチ名をコメントで明記します。
- `main` には再生成可能なコード・文書・最小限の雛形だけを置き、大きな JSON・画像・ログを常設しません。
- 長く使う branch や results branch は、`./notes/branches/README.md` に登録し、関連 note へ main から一段で辿れるようにします。
- リポジトリ内の Markdown は `markdownlint` に準拠させ、repo root の `.markdownlint.json` を既定設定として扱います。
- Markdown の変更時は、`README.md`, `documents/`, `reviews/`, `notes/` など触れた範囲に対して `markdownlint` を実行してから共有します。
- Markdown 文書中の JSON・画像・補助データへのリンクは、必ずその文書から見た相対パスで書きます。
- 絶対パスを Markdown の恒久リンクとして使いません。
