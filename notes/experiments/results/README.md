# Experiment Result JSON Archive

このディレクトリには、`main` に持ち帰る最小限の final JSON を置きます。

- 目的は、後から別の図や集計を再生成できるようにすることです。
- raw な JSONL、巨大ログ、途中経過の全ファイルまでは置きません。
- branch を代表する final JSON、あるいは partial でも再解析価値の高い JSON を選んで置きます。

各 JSON について、対応する note から

- branch 名
- 元の results branch
- 元データの所在
- その JSON を持ち帰った理由

が辿れるようにします。
